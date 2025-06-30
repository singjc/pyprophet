import os
import re
import pickle
from shutil import copyfile
import sqlite3
from typing import Literal, List
import duckdb
import pandas as pd
import click
from loguru import logger
from ..util import check_sqlite_table, check_duckdb_table, unimod_to_codename
from .._base import BaseOSWReader, BaseOSWWriter
from ..._config import IPFIOConfig


class OSWReader(BaseOSWReader):
    """
    Class for reading and processing data from an OpenSWATH workflow OSW-sqlite based file.

    The OSWReader class provides methods to read different levels of data from the file and process it accordingly.
    It supports reading data for semi-supervised learning, IPF analysis, context level analysis.

    Attributes:
        infile (str): Input file path.
        outfile (str): Output file path.
        classifier (str): Classifier used for semi-supervised learning.
        level (str): Level used in semi-supervised learning (e.g., 'ms1', 'ms2', 'ms1ms2', 'transition', 'alignment'), or context level used peptide/protein/gene inference (e.g., 'global', 'experiment-wide', 'run-specific').
        glyco (bool): Flag indicating whether analysis is glycoform-specific.

    Methods:
        read(): Read data from the input file based on the alogorithm.
    """

    def __init__(self, config: IPFIOConfig):
        super().__init__(config)

        self.prec_peptide_mapping = self._fetch_precursor_peptide_mapping()

    def read(
        self,
        level: Literal["peakgroup_precursor", "transition", "alignment", "peptide_ids"],
        peptide_ids: List[int] = None,
    ) -> pd.DataFrame:
        self._create_indexes()
        try:
            con = duckdb.connect()
            con.execute("INSTALL sqlite_scanner;")
            con.execute("LOAD sqlite_scanner;")
            con.execute(f"ATTACH DATABASE '{self.infile}' AS osw (TYPE sqlite);")
            return self._read_using_duckdb(con, level, peptide_ids)
        except ModuleNotFoundError as e:
            logger.warning(
                f"Warn: DuckDB sqlite_scanner failed, falling back to SQLite. Reason: {e}",
            )
            con = sqlite3.connect(self.infile)
            return self._read_using_sqlite(con, level, peptide_ids)

    def _create_indexes(self):
        """
        Always use a temporary SQLite connection to create indexes directly on the .osw file,
        since DuckDB doesn't seem to currently support creating indexes on attached SQLite databases.
        """
        try:
            sqlite_con = sqlite3.connect(self.infile)

            index_statements = [
                "CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);",
                "CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);",
                "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);",
                "CREATE INDEX IF NOT EXISTS idx_score_transition_feature_id ON SCORE_TRANSITION (FEATURE_ID);",
                "CREATE INDEX IF NOT EXISTS idx_score_transition_transition_id ON SCORE_TRANSITION (TRANSITION_ID);",
            ]

            for stmt in index_statements:
                try:
                    sqlite_con.execute(stmt)
                except sqlite3.OperationalError as e:
                    logger.warning(f"Warn: SQLite index creation failed: {e}")

            sqlite_con.commit()
            sqlite_con.close()

        except Exception as e:
            raise click.ClickException(
                f"Failed to create indexes via SQLite fallback: {e}"
            )

    def _fetch_precursor_peptide_mapping(self):
        """
        Fetch precursor to peptide mapping.
        """

        with sqlite3.connect(self.config.infile) as con:
            query = """SELECT DISTINCT PEPTIDE_ID AS ID_codename, UNMODIFIED_SEQUENCE, MODIFIED_SEQUENCE
            FROM SCORE_TRANSITION
            INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
            INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
            INNER JOIN PEPTIDE ON PEPTIDE.ID = TRANSITION_PEPTIDE_MAPPING.PEPTIDE_ID
            WHERE TRANSITION.TYPE != ''
            AND TRANSITION.DECOY = 0"""
            peptidoform_df = pd.read_sql_query(query, con)

            balanced = r"\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)"
            peptidoform_df["mods_list"] = peptidoform_df["MODIFIED_SEQUENCE"].apply(
                lambda s: re.findall(balanced, s)
            )
            peptidoform_df["bare_seq"] = peptidoform_df["MODIFIED_SEQUENCE"].apply(
                lambda s: re.sub(balanced, "", s)
            )
            peptidoform_df["isoform_signature"] = peptidoform_df[
                "bare_seq"
            ] + peptidoform_df["mods_list"].apply(lambda L: "".join(sorted(L)))

            peptidoform_df["isoform_id"] = peptidoform_df.groupby(
                "isoform_signature"
            ).ngroup()

            peptide_df = pd.read_sql_query(
                "SELECT ID, MODIFIED_SEQUENCE FROM PEPTIDE", con
            )
            precursor_df = pd.read_sql_query(
                "SELECT PRECURSOR_ID as precursor_id, PEPTIDE_ID AS ID_unimod FROM PRECURSOR_PEPTIDE_MAPPING",
                con,
            )

        peptide_df["codename"] = peptide_df["MODIFIED_SEQUENCE"].apply(
            unimod_to_codename
        )

        unimod_mask = peptide_df["MODIFIED_SEQUENCE"].str.contains("UniMod")
        merged_df = pd.merge(
            peptide_df[unimod_mask][["codename", "ID"]],
            peptide_df[~unimod_mask][["codename", "ID"]],
            on="codename",
            suffixes=("_unimod", "_codename"),
            how="outer",
        )

        # Fill NaN values in the 'ID_codename' column with the 'ID_unimod' values
        merged_df["ID_codename"] = merged_df["ID_codename"].fillna(
            merged_df["ID_unimod"]
        )
        # Fill NaN values in the 'ID_unimod' column with the 'ID_codename' values
        merged_df["ID_unimod"] = merged_df["ID_unimod"].fillna(merged_df["ID_codename"])

        merged_df["ID_unimod"] = merged_df["ID_unimod"].astype(int)
        merged_df["ID_codename"] = merged_df["ID_codename"].astype(int)

        prec_peptide_mapping = pd.merge(
            precursor_df,
            merged_df[["ID_unimod", "ID_codename"]],
            on="ID_unimod",
            how="left",
        )

        prec_peptide_mapping = pd.merge(
            prec_peptide_mapping,
            peptidoform_df[
                [
                    "ID_codename",
                    "isoform_signature",
                    "isoform_id",
                ]
            ],
            left_on="ID_codename",
            right_on="ID_codename",
            how="outer",
        )

        # Drop NaN values in isoform_id
        prec_peptide_mapping = prec_peptide_mapping.dropna(subset=["isoform_id"])

        return prec_peptide_mapping.rename(
            columns={
                "ID_unimod": "peptide_id",
                "ID_codename": "ipf_peptide_id",
            }
        )

    def _read_using_duckdb(
        self,
        con,
        level: Literal["peakgroup_precursor", "transition", "alignment"],
        peptide_ids: List[int] = None,
    ):
        if level == "peakgroup_precursor":
            return self._read_pyp_peakgroup_precursor_duckdb(con, peptide_ids)
        elif level == "transition":
            return self._read_pyp_transition_duckdb(con, peptide_ids)
        elif level == "alignment":
            return self._fetch_alignment_features_duckdb(con)
        elif level == "peptide_ids":
            return (
                self.prec_peptide_mapping[["ipf_peptide_id", "isoform_id"]]
                .rename(columns={"ipf_peptide_id": "peptide_id"})
                .drop_duplicates()
                .reset_index(drop=True)
            )
        else:
            raise click.ClickException(f"Unsupported level: {level}")

    def _read_using_sqlite(
        self,
        con,
        level: Literal["peakgroup_precursor", "transition", "alignment"],
        peptide_ids: List[int] = None,
    ):
        if level == "peakgroup_precursor":
            return self._read_pyp_peakgroup_precursor_sqlite(con, peptide_ids)
        elif level == "transition":
            return self._read_pyp_transition_sqlite(con, peptide_ids)
        elif level == "alignment":
            return self._fetch_alignment_features_sqlite(con)
        elif level == "peptide_ids":
            return (
                self.prec_peptide_mapping[["ipf_peptide_id"]]
                .rename(columns={"ipf_peptide_id": "peptide_id"})
                .drop_duplicates()
                .reset_index(drop=True)
            )
        else:
            raise click.ClickException(f"Unsupported level: {level}")

    # ----------------------------
    # DuckDB Queries
    # ----------------------------

    def _fetch_tables_duckdb(self, con):
        tables = con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables"
        ).fetchdf()
        return tables

    def _fetch_precursor_ids_duckdb(self, con, peptide_ids: List[int] = None):
        """
        Fetch precursor IDs based on peptide IDs.
        If peptide_ids is None, fetch all precursor IDs.
        """
        # use self.prec_peptide_mapping to get the precursor IDs
        ipf_peptide_id_mask = self.prec_peptide_mapping["ipf_peptide_id"].isin(
            peptide_ids
        )
        return (
            self.prec_peptide_mapping.loc[ipf_peptide_id_mask, "precursor_id"]
            .unique()
            .tolist()
        )

    def _read_pyp_peakgroup_precursor_duckdb(self, con, peptide_ids: List[int] = None):
        cfg = self.config  # IPFIOConfig instance
        ipf_ms1 = cfg.ipf_ms1_scoring
        ipf_ms2 = cfg.ipf_ms2_scoring
        pep_threshold = cfg.ipf_max_peakgroup_pep

        # precursors are restricted according to ipf_max_peakgroup_pep to exclude very poor peak groups
        logger.info("Reading precursor-level data ...")

        precursor_ids_filter_query = ""
        if peptide_ids is not None:
            precursor_ids = self._fetch_precursor_ids_duckdb(con, peptide_ids)
            precursor_ids_str = ",".join(map(str, precursor_ids))
            precursor_ids_filter_query = (
                f"AND osw.FEATURE.PRECURSOR_ID IN ({precursor_ids_str})"
            )

        if not ipf_ms1 and ipf_ms2:  # only use MS2 precursors
            if not check_duckdb_table(
                con, "main", "SCORE_MS2"
            ) or not check_duckdb_table(con, "main", "SCORE_TRANSITION"):
                raise click.ClickException(
                    f"Apply scoring to MS2 and transition-level data before running IPF.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
                )
            query = f"""
                SELECT FEATURE.ID AS FEATURE_ID,
                    SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
                    NULL AS MS1_PRECURSOR_PEP,
                    SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PEP
                FROM osw.PRECURSOR
                INNER JOIN osw.FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN osw.SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                INNER JOIN (
                    SELECT FEATURE_ID, PEP
                    FROM osw.SCORE_TRANSITION
                    INNER JOIN osw.TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                    WHERE TRANSITION.TYPE='' AND TRANSITION.DECOY=0
                ) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID
                WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < {pep_threshold}
                {precursor_ids_filter_query}
            """

        elif ipf_ms1 and not ipf_ms2:  # only use MS1 precursors
            if not check_duckdb_table(
                con, "main", "SCORE_MS1"
            ) or not check_duckdb_table(con, "main", "SCORE_TRANSITION"):
                raise click.ClickException(
                    f"Apply scoring to MS1 and transition-level data before running IPF.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
                )

            query = f"""
                SELECT FEATURE.ID AS FEATURE_ID,
                    SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
                    SCORE_MS1.PEP AS MS1_PRECURSOR_PEP,
                    NULL AS MS2_PRECURSOR_PEP
                FROM osw.PRECURSOR
                INNER JOIN osw.FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN osw.SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
                INNER JOIN osw.SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < {pep_threshold}
                {precursor_ids_filter_query}
            """

        elif ipf_ms1 and ipf_ms2:  # use both MS1 and MS2 precursors
            if (
                not check_duckdb_table(con, "main", "SCORE_MS1")
                or not check_duckdb_table(con, "main", "SCORE_MS2")
                or not check_duckdb_table(con, "main", "SCORE_TRANSITION")
            ):
                raise click.ClickException(
                    f"Apply scoring to MS1, MS2 and transition-level data before running IPF.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
                )

            query = f"""
                SELECT FEATURE.ID AS FEATURE_ID,
                    SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
                    SCORE_MS1.PEP AS MS1_PRECURSOR_PEP,
                    SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PEP
                FROM osw.PRECURSOR
                INNER JOIN osw.FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN osw.SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
                INNER JOIN osw.SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                INNER JOIN (
                    SELECT FEATURE_ID, PEP
                    FROM osw.SCORE_TRANSITION
                    INNER JOIN osw.TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                    WHERE TRANSITION.TYPE='' AND TRANSITION.DECOY=0
                ) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID
                WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < {pep_threshold}
                {precursor_ids_filter_query}
            """

        else:  # do not use any precursor information
            if not check_duckdb_table(
                con, "main", "SCORE_MS2"
            ) or not check_duckdb_table(con, "main", "SCORE_TRANSITION"):
                raise click.ClickException(
                    f"Apply scoring to MS2 and transition-level data before running IPF.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
                )

            query = f"""
                SELECT FEATURE.ID AS FEATURE_ID,
                    SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
                    NULL AS MS1_PRECURSOR_PEP,
                    NULL AS MS2_PRECURSOR_PEP
                FROM osw.PRECURSOR
                INNER JOIN osw.FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN osw.SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < {pep_threshold}
                {precursor_ids_filter_query}
            """

        df = con.execute(query).fetchdf()
        return df.rename(columns=str.lower)

    def _read_pyp_transition_duckdb(self, con, peptide_ids: List[int] = None):
        rc = self.config
        ipf_h0 = rc.ipf_h0
        pep_threshold = rc.ipf_max_transition_pep

        peptide_ids_filter_query = ""
        if peptide_ids is not None:
            peptide_ids_str = ",".join(map(str, peptide_ids))
            peptide_ids_filter_query = (
                f"AND osw.TRANSITION_PEPTIDE_MAPPING.PEPTIDE_ID IN ({peptide_ids_str})"
            )

        # only the evidence is restricted to ipf_max_transition_pep, the peptidoform-space is complete
        logger.info("Info: Reading peptidoform-level data ...")

        queries = {
            "evidence": f"""
                SELECT FEATURE_ID, TRANSITION_ID, PEP
                FROM osw.SCORE_TRANSITION
                INNER JOIN osw.TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                WHERE TRANSITION.TYPE != ''
                AND TRANSITION.DECOY = 0
                AND PEP < {pep_threshold}
            """,
            "bitmask": f"""
                SELECT DISTINCT TRANSITION.ID AS TRANSITION_ID, PEPTIDE_ID, 1 AS BMASK
                FROM osw.SCORE_TRANSITION
                INNER JOIN osw.TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                INNER JOIN osw.TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
                WHERE TRANSITION.TYPE != ''
                AND TRANSITION.DECOY = 0
                {peptide_ids_filter_query}
            """,
            "num_peptidoforms": f"""
                SELECT FEATURE_ID, COUNT(DISTINCT PEPTIDE_ID) AS NUM_PEPTIDOFORMS
                FROM osw.SCORE_TRANSITION
                INNER JOIN osw.TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                INNER JOIN osw.TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
                WHERE TRANSITION.TYPE != ''
                AND TRANSITION.DECOY = 0
                {peptide_ids_filter_query}
                GROUP BY FEATURE_ID
            """,
            "peptidoforms": f"""
                SELECT DISTINCT FEATURE_ID, PEPTIDE_ID
                FROM osw.SCORE_TRANSITION
                INNER JOIN osw.TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                INNER JOIN osw.TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
                WHERE TRANSITION.TYPE != ''
                AND TRANSITION.DECOY = 0
                {peptide_ids_filter_query}
            """,
        }

        # Execute
        evidence = con.execute(queries["evidence"]).fetchdf().rename(columns=str.lower)
        bitmask = con.execute(queries["bitmask"]).fetchdf().rename(columns=str.lower)
        num_peptidoforms = (
            con.execute(queries["num_peptidoforms"]).fetchdf().rename(columns=str.lower)
        )
        peptidoforms = (
            con.execute(queries["peptidoforms"]).fetchdf().rename(columns=str.lower)
        )

        # Add null hypothesis (peptide_id = -1)
        if ipf_h0:
            peptidoforms = pd.concat(
                [
                    peptidoforms,
                    pd.DataFrame(
                        {
                            "feature_id": peptidoforms["feature_id"].unique(),
                            "peptide_id": -1,
                        }
                    ),
                ],
                ignore_index=True,
            )

        # Merge
        trans_pf = pd.merge(evidence, peptidoforms, how="outer", on="feature_id")
        trans_pf_bm = pd.merge(
            trans_pf, bitmask, how="left", on=["transition_id", "peptide_id"]
        ).fillna(0)
        data = pd.merge(trans_pf_bm, num_peptidoforms, how="inner", on="feature_id")

        return data

    def _fetch_alignment_features_duckdb(self, con):
        pep_threshold = self.config.ipf_max_alignment_pep

        if not check_duckdb_table(
            con, "main", "FEATURE_MS2_ALIGNMENT"
        ) or not check_duckdb_table(con, "main", "SCORE_ALIGNMENT"):
            raise click.ClickException(
                f"Perform feature alignment using ARYCAL, and apply scoring to alignment-level data before running IPF.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )

        query = f"""
            SELECT  
                DENSE_RANK() OVER (ORDER BY PRECURSOR_ID, ALIGNMENT_ID) AS ALIGNMENT_GROUP_ID,
                ALIGNED_FEATURE_ID AS FEATURE_ID 
            FROM (
                SELECT DISTINCT * FROM osw.FEATURE_MS2_ALIGNMENT
            ) AS FEATURE_MS2_ALIGNMENT
            INNER JOIN (
                SELECT 
                    FEATURE_ID,
                    MIN(QVALUE) AS QVALUE,
                    ANY_VALUE(SCORE) AS SCORE,
                    ANY_VALUE(PEP) AS PEP
                FROM osw.SCORE_ALIGNMENT 
                GROUP BY FEATURE_ID
            ) AS SCORE_ALIGNMENT 
            ON SCORE_ALIGNMENT.FEATURE_ID = FEATURE_MS2_ALIGNMENT.ALIGNED_FEATURE_ID
            WHERE LABEL = 1
            AND SCORE_ALIGNMENT.PEP < {pep_threshold}
            ORDER BY ALIGNMENT_GROUP_ID
        """

        df = con.execute(query).fetchdf()
        return df.rename(columns=str.lower)

    # ----------------------------
    # SQLite fallback
    # ----------------------------

    def _read_pyp_peakgroup_precursor_sqlite(self, con, peptide_ids: List[int] = None):
        cfg = self.config
        ipf_ms1 = cfg.ipf_ms1_scoring
        ipf_ms2 = cfg.ipf_ms2_scoring
        pep_threshold = cfg.ipf_max_peakgroup_pep

        if not ipf_ms1 and ipf_ms2:  # only use MS2 precursors
            if not check_sqlite_table(con, "SCORE_MS2") or not check_sqlite_table(
                con, "SCORE_TRANSITION"
            ):
                raise click.ClickException(
                    "Apply scoring to MS2 and transition-level data before running IPF."
                )

            query = """
                SELECT FEATURE.ID AS FEATURE_ID,
                    SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
                    NULL AS MS1_PRECURSOR_PEP,
                    SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PEP
                FROM PRECURSOR
                INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                INNER JOIN (
                    SELECT FEATURE_ID, PEP
                    FROM SCORE_TRANSITION
                    INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                    WHERE TRANSITION.TYPE='' AND TRANSITION.DECOY=0
                ) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID
                WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < ?
            """

        elif ipf_ms1 and not ipf_ms2:  # only use MS1 precursors
            if not check_sqlite_table(con, "SCORE_MS1") or not check_sqlite_table(
                con, "SCORE_TRANSITION"
            ):
                raise click.ClickException(
                    "Apply scoring to MS1 and transition-level data before running IPF."
                )

            query = """
                SELECT FEATURE.ID AS FEATURE_ID,
                    SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
                    SCORE_MS1.PEP AS MS1_PRECURSOR_PEP,
                    NULL AS MS2_PRECURSOR_PEP
                FROM PRECURSOR
                INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
                INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < ?
            """

        elif ipf_ms1 and ipf_ms2:
            if (
                not check_sqlite_table(con, "SCORE_MS1")
                or not check_sqlite_table(con, "SCORE_MS2")
                or not check_sqlite_table(con, "SCORE_TRANSITION")
            ):
                raise click.ClickException(
                    "Apply scoring to MS1, MS2 and transition-level data before running IPF."
                )
            query = """
                SELECT FEATURE.ID AS FEATURE_ID,
                    SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
                    SCORE_MS1.PEP AS MS1_PRECURSOR_PEP,
                    SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PEP
                FROM PRECURSOR
                INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
                INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                INNER JOIN (
                    SELECT FEATURE_ID, PEP
                    FROM SCORE_TRANSITION
                    INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                    WHERE TRANSITION.TYPE='' AND TRANSITION.DECOY=0
                ) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID
                WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < ?
            """

        else:
            query = """
                SELECT FEATURE.ID AS FEATURE_ID,
                    SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
                    NULL AS MS1_PRECURSOR_PEP,
                    NULL AS MS2_PRECURSOR_PEP
                FROM PRECURSOR
                INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < ?
            """

        df = pd.read_sql_query(query, con, params=[pep_threshold])
        return df.rename(columns=str.lower)

    def _read_pyp_transition_sqlite(self, con, peptide_ids: List[int] = None):
        rc = self.config
        ipf_h0 = rc.ipf_h0
        pep_threshold = rc.ipf_max_transition_pep

        # only the evidence is restricted to ipf_max_transition_pep, the peptidoform-space is complete
        logger.info("Info: Reading peptidoform-level data ...")

        queries = {
            "evidence": """
                SELECT FEATURE_ID, TRANSITION_ID, PEP
                FROM SCORE_TRANSITION
                INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                WHERE TRANSITION.TYPE != ''
                AND TRANSITION.DECOY = 0
                AND PEP < ?
            """,
            "bitmask": """
                SELECT DISTINCT TRANSITION.ID AS TRANSITION_ID, PEPTIDE_ID, 1 AS BMASK
                FROM SCORE_TRANSITION
                INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
                WHERE TRANSITION.TYPE != ''
                AND TRANSITION.DECOY = 0
            """,
            "num_peptidoforms": """
                SELECT FEATURE_ID, COUNT(DISTINCT PEPTIDE_ID) AS NUM_PEPTIDOFORMS
                FROM SCORE_TRANSITION
                INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
                WHERE TRANSITION.TYPE != ''
                AND TRANSITION.DECOY = 0
                GROUP BY FEATURE_ID, SCORE_TRANSITION.TRANSITION_ID
            """,
            "peptidoforms": """
                SELECT DISTINCT FEATURE_ID, PEPTIDE_ID
                FROM SCORE_TRANSITION
                INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
                WHERE TRANSITION.TYPE != ''
                AND TRANSITION.DECOY = 0
            """,
        }

        # Execute
        evidence = pd.read_sql_query(
            queries["evidence"], con, params=[pep_threshold]
        ).rename(columns=str.lower)
        bitmask = pd.read_sql_query(queries["bitmask"], con).rename(columns=str.lower)
        num_peptidoforms = pd.read_sql_query(queries["num_peptidoforms"], con).rename(
            columns=str.lower
        )
        peptidoforms = pd.read_sql_query(queries["peptidoforms"], con).rename(
            columns=str.lower
        )

        # Add null hypothesis
        if ipf_h0:
            peptidoforms = pd.concat(
                [
                    peptidoforms,
                    pd.DataFrame(
                        {
                            "feature_id": peptidoforms["feature_id"].unique(),
                            "peptide_id": -1,
                        }
                    ),
                ],
                ignore_index=True,
            )

        # Merge
        trans_pf = pd.merge(evidence, peptidoforms, how="outer", on="feature_id")
        trans_pf_bm = pd.merge(
            trans_pf, bitmask, how="left", on=["transition_id", "peptide_id"]
        ).fillna(0)
        data = pd.merge(trans_pf_bm, num_peptidoforms, how="inner", on="feature_id")

        return data.drop_duplicates()

    def _fetch_alignment_features_sqlite(self, con):
        pep_threshold = self.config.ipf_max_alignment_pep

        if not check_sqlite_table(
            con, "FEATURE_MS2_ALIGNMENT"
        ) or not check_sqlite_table(con, "SCORE_ALIGNMENT"):
            raise click.ClickException(
                "Perform feature alignment using ARYCAL, and apply scoring to alignment-level data before running IPF."
            )

        query = f"""
            SELECT  
                DENSE_RANK() OVER (ORDER BY PRECURSOR_ID, ALIGNMENT_ID) AS ALIGNMENT_GROUP_ID,
                ALIGNED_FEATURE_ID AS FEATURE_ID 
            FROM (
                SELECT DISTINCT * FROM FEATURE_MS2_ALIGNMENT
            ) AS FEATURE_MS2_ALIGNMENT
            INNER JOIN (
                SELECT DISTINCT *, MIN(QVALUE) 
                FROM SCORE_ALIGNMENT 
                GROUP BY FEATURE_ID
            ) AS SCORE_ALIGNMENT 
            ON SCORE_ALIGNMENT.FEATURE_ID = FEATURE_MS2_ALIGNMENT.ALIGNED_FEATURE_ID
            WHERE LABEL = 1
            AND SCORE_ALIGNMENT.PEP < {pep_threshold}
            ORDER BY ALIGNMENT_GROUP_ID
        """

        df = pd.read_sql_query(query, con)
        return df.rename(columns=str.lower)


class OSWWriter(BaseOSWWriter):
    """
    Class for writing OpenSWATH results to an OSW-sqlite based file.

    Attributes:
        infile (str): Input file path.
        outfile (str): Output file path.
        classifier (str): Classifier used for semi-supervised learning.
        level (str): Level used in semi-supervised learning (e.g., 'ms1', 'ms2', 'ms1ms2', 'transition', 'alignment'), or context level used peptide/protein/gene inference (e.g., 'global', 'experiment-wide', 'run-specific').
        glyco (bool): Flag indicating whether analysis is glycoform-specific.

    Methods:
        save_results(result, pi0): Save the results to the output file based on the module using this class.
        save_weights(weights): Save the weights to the output file.
    """

    def __init__(self, config: IPFIOConfig):
        super().__init__(config)

    def save_results(self, result):
        """
        Save the results to the output file based on the module using this class.
        """
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        con = sqlite3.connect(self.outfile)
        result.to_sql("SCORE_IPF", con, index=False, if_exists="replace")
        con.close()
