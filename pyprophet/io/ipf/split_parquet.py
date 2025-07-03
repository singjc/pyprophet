import os
import glob
import re
from shutil import copyfile
from typing import Literal, List
import pandas as pd
import pyarrow as pa
import duckdb
import click
from loguru import logger

from ..util import get_parquet_column_names
from .._base import BaseSplitParquetReader, BaseSplitParquetWriter
from ..._config import IPFIOConfig


class SplitParquetReader(BaseSplitParquetReader):
    """
    Class for reading and processing data from OpenSWATH results stored in a directoy containing split Parquet files.

    The ParquetReader class provides methods to read different levels of data from the split parquet files and process it accordingly.
    It supports reading data for semi-supervised learning, IPF analysis, context level analysis.

    This assumes that the input infile path is a directory containing the following files:
    - precursors_features.parquet
    - transition_features.parquet
    - feature_alignment.parquet (optional)

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

    def read(
        self,
        level: Literal["peakgroup_precursor", "transition", "alignment", "peptide_ids"],
        peptide_ids: List[int] = None,
    ) -> pd.DataFrame:
        con = duckdb.connect()
        try:
            self._init_duckdb_views(con)

            if level == "peakgroup_precursor":
                return self._read_pyp_peakgroup_precursor(con, peptide_ids)
            elif level == "transition":
                return self._read_pyp_transition(con, peptide_ids)
            elif level == "alignment":
                return self._fetch_alignment_features(con)
            elif level == "peptide_ids":
                return self._fetch_peptide_ids(con)
            else:
                raise click.ClickException(f"Unsupported level: {level}")
        finally:
            con.close()

    def _read_pyp_peakgroup_precursor(
        self, con, peptide_ids: List[int] = None
    ) -> pd.DataFrame:
        cfg = self.config
        ipf_ms1 = cfg.ipf_ms1_scoring
        ipf_ms2 = cfg.ipf_ms2_scoring
        pep_threshold = cfg.ipf_max_peakgroup_pep

        logger.info("Reading precursor-level data ...")

        if cfg.file_type == "parquet_split_multi":
            precursor_files = glob.glob(
                os.path.join(self.infile, "*.oswpq", "precursors_features.parquet")
            )
            transition_files = glob.glob(
                os.path.join(self.infile, "*.oswpq", "transition_features.parquet")
            )
        else:
            precursor_files = [os.path.join(self.infile, "precursors_features.parquet")]
            transition_files = [
                os.path.join(self.infile, "transition_features.parquet")
            ]

        all_precursor_cols = get_parquet_column_names(precursor_files[0])
        all_transition_cols = get_parquet_column_names(transition_files[0])

        if peptide_ids is not None:
            peptide_ids_filter_query = (
                f" AND IPF_PEPTIDE_ID IN ({','.join(map(str, peptide_ids))})"
            )
        else:
            peptide_ids_filter_query = ""

        # con.execute(
        #     f"CREATE VIEW precursors AS SELECT * FROM read_parquet({precursor_files})"
        # )
        # con.execute(
        #     f"CREATE VIEW transitions AS SELECT * FROM read_parquet({transition_files})"
        # )

        if not ipf_ms1 and ipf_ms2:
            if not any(
                c.startswith("SCORE_MS2_") for c in all_precursor_cols
            ) or not any(
                c.startswith("SCORE_TRANSITION_") for c in all_transition_cols
            ):
                raise click.ClickException("Apply MS2 + transition scoring before IPF.")
            query = f"""
            SELECT p.FEATURE_ID, p.SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                NULL AS MS1_PRECURSOR_PEP, t.SCORE_TRANSITION_PEP AS MS2_PRECURSOR_PEP
            FROM precursors p
            INNER JOIN (
                SELECT FEATURE_ID, SCORE_TRANSITION_PEP
                FROM transition
                WHERE TRANSITION_TYPE = '' AND TRANSITION_DECOY = 0
            ) t ON p.FEATURE_ID = t.FEATURE_ID
            WHERE p.PRECURSOR_DECOY = 0 AND p.SCORE_MS2_PEP < {pep_threshold}
            {peptide_ids_filter_query}
            """

        elif ipf_ms1 and not ipf_ms2:
            if not any(
                c.startswith("SCORE_MS1_") for c in all_precursor_cols
            ) or not any(c.startswith("SCORE_MS2_") for c in all_precursor_cols):
                raise click.ClickException("Apply MS1 + MS2 scoring before IPF.")
            query = f"""
            SELECT p.FEATURE_ID, p.SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                p.SCORE_MS1_PEP AS MS1_PRECURSOR_PEP, NULL AS MS2_PRECURSOR_PEP
            FROM precursors p
            WHERE p.PRECURSOR_DECOY = 0 AND p.SCORE_MS2_PEP < {pep_threshold}
            {peptide_ids_filter_query}
            """

        elif ipf_ms1 and ipf_ms2:
            if not all(
                [
                    any(c.startswith("SCORE_MS1_") for c in all_precursor_cols),
                    any(c.startswith("SCORE_MS2_") for c in all_precursor_cols),
                    any(c.startswith("SCORE_TRANSITION_") for c in all_transition_cols),
                ]
            ):
                raise click.ClickException(
                    "Apply MS1 + MS2 + transition scoring before IPF."
                )
            query = f"""
            SELECT p.FEATURE_ID, p.SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                p.SCORE_MS1_PEP AS MS1_PRECURSOR_PEP, t.SCORE_TRANSITION_PEP AS MS2_PRECURSOR_PEP
            FROM precursors p
            INNER JOIN (
                SELECT FEATURE_ID, SCORE_TRANSITION_PEP
                FROM transition
                WHERE TRANSITION_TYPE = '' AND TRANSITION_DECOY = 0
            ) t ON p.FEATURE_ID = t.FEATURE_ID
            WHERE p.PRECURSOR_DECOY = 0 AND p.SCORE_MS2_PEP < {pep_threshold}
            {peptide_ids_filter_query}
            """

        else:
            if not any(
                c.startswith("SCORE_MS2_") for c in all_precursor_cols
            ) or not any(
                c.startswith("SCORE_TRANSITION_") for c in all_transition_cols
            ):
                raise click.ClickException("Apply MS2 + transition scoring before IPF.")
            query = f"""
            SELECT p.FEATURE_ID, p.SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                NULL AS MS1_PRECURSOR_PEP, NULL AS MS2_PRECURSOR_PEP
            FROM precursors p
            WHERE p.PRECURSOR_DECOY = 0 AND p.SCORE_MS2_PEP < {pep_threshold}
            {peptide_ids_filter_query}
            """

        df = con.execute(query).df()
        df.columns = [col.lower() for col in df.columns]
        return df

    def _read_pyp_transition(
        self, con, peptide_ids: List[int] = None, chunk_size: int = 100_000
    ) -> pd.DataFrame:
        """
        Read and merge transition-peptidoform data in chunks to limit peak memory usage.

        Splits work by feature_id, processing a subset of features at a time.
        """
        cfg = self.config
        ipf_h0 = cfg.ipf_h0
        pep_threshold = cfg.ipf_max_transition_pep

        logger.info("Reading peptidoform-level data ...")

        # Resolve transition file paths
        if cfg.file_type == "parquet_split_multi":
            transition_files = glob.glob(
                os.path.join(self.infile, "*.oswpq", "transition_features.parquet")
            )
        else:
            transition_files = [
                os.path.join(self.infile, "transition_features.parquet")
            ]

        if not transition_files:
            raise click.ClickException("No transition_features.parquet files found.")

        # Use first file for column check
        all_transition_cols = get_parquet_column_names(transition_files[0])

        if "IPF_PEPTIDE_ID" not in all_transition_cols:
            raise click.ClickException(
                "IPF_PEPTIDE_ID column is required in transition features."
            )

        if peptide_ids is not None:
            peptide_ids_filter_query = (
                f" AND peptide_id IN ({','.join(map(str, peptide_ids))})"
            )
        else:
            peptide_ids_filter_query = ""

        # --------------------------------------------------------------------------
        # Step 1: Create or refresh DuckDB view for all transition features
        # --------------------------------------------------------------------------
        # con.execute(
        #     f"""
        #     CREATE OR REPLACE VIEW transitions AS
        #     SELECT
        #     FEATURE_ID,
        #     TRANSITION_ID,
        #     IPF_PEPTIDE_ID AS peptide_id,
        #     SCORE_TRANSITION_PEP  AS pep,
        #     SCORE_TRANSITION_SCORE,
        #     TRANSITION_TYPE,
        #     TRANSITION_DECOY
        #     FROM read_parquet({transition_files})
        #     """
        # )
        filter_where = f"""
        WHERE TRANSITION_TYPE       != ''
            AND TRANSITION_DECOY       = 0
            AND SCORE_TRANSITION_SCORE IS NOT NULL
            AND SCORE_TRANSITION_PEP   < {pep_threshold}
            {peptide_ids_filter_query}
        """

        con.execute(f"""
        PRAGMA threads=8;  -- enable multicore parquet scans

        CREATE OR REPLACE VIEW transitions AS
        SELECT
            FEATURE_ID,
            TRANSITION_ID,
            IPF_PEPTIDE_ID    AS peptide_id,
            SCORE_TRANSITION_PEP   AS pep
        FROM read_parquet({transition_files})
        {filter_where};
        """)

        # --------------------------------------------------------------------------
        # Step 2: Single SQL query with CTEs to build evidence, peptidoforms, bitmask, and counts
        # --------------------------------------------------------------------------

        # # Prepare optional decoy union for peptidoforms
        # union_sql = (
        #     "UNION ALL SELECT feature_id, -1 AS peptide_id FROM peptidoforms_real"
        #     if ipf_h0
        #     else ""
        # )

        # sql = f"""
        # WITH
        # -- evidence: transitions passing pep threshold
        # evidence AS (
        #     SELECT feature_id, transition_id, pep
        #     FROM transitions
        #     WHERE TRANSITION_TYPE       != ''
        #     AND TRANSITION_DECOY       = 0
        #     AND SCORE_TRANSITION_SCORE IS NOT NULL
        #     AND pep < {pep_threshold}
        # ),

        # -- real peptidoforms: all peptide_ids per feature
        # peptidoforms_real AS (
        #     SELECT DISTINCT feature_id, peptide_id
        #     FROM transitions
        #     WHERE TRANSITION_TYPE       != ''
        #     AND TRANSITION_DECOY       = 0
        #     AND SCORE_TRANSITION_SCORE IS NOT NULL
        #     AND peptide_id IS NOT NULL
        #     {peptide_ids_filter_query}
        # ),

        # -- include optional decoys
        # peptidoforms AS (
        #     SELECT * FROM peptidoforms_real
        #     {union_sql}
        # ),

        # -- bitmask: observed transition-peptidoform pairs
        # bitmask AS (
        #     SELECT DISTINCT t.transition_id, t.peptide_id, 1 AS bmask
        #     FROM transitions t
        #     WHERE t.TRANSITION_TYPE       != ''
        #     AND t.TRANSITION_DECOY       = 0
        #     AND t.SCORE_TRANSITION_SCORE IS NOT NULL
        #     AND t.peptide_id IS NOT NULL
        #     {peptide_ids_filter_query}
        # ),

        # -- counts: number of real peptidoforms per feature
        # counts AS (
        #     SELECT feature_id, COUNT(DISTINCT peptide_id) AS num_peptidoforms
        #     FROM peptidoforms_real
        #     {peptide_ids_filter_query.replace("AND", "WHERE", 1)}
        #     GROUP BY feature_id
        # )

        # SELECT
        # COALESCE(e.feature_id, p.feature_id) AS feature_id,
        # e.transition_id,
        # e.pep,
        # p.peptide_id,
        # COALESCE(b.bmask, 0)      AS bmask,
        # c.num_peptidoforms       AS num_peptidoforms
        # FROM evidence e
        # FULL OUTER JOIN peptidoforms p
        # ON e.feature_id = p.feature_id
        # LEFT JOIN bitmask b
        # ON e.transition_id = b.transition_id
        # AND p.peptide_id   = b.peptide_id
        # JOIN counts c
        # ON COALESCE(e.feature_id, p.feature_id) = c.feature_id
        # """
        sql = f"""
        WITH
        peptidoforms_real AS (
            SELECT DISTINCT feature_id, peptide_id
            FROM transitions
            WHERE peptide_id IS NOT NULL
        ),

        peptidoforms AS (
            SELECT * FROM peptidoforms_real
            {"UNION ALL SELECT feature_id, -1 AS peptide_id FROM peptidoforms_real" if ipf_h0 else ""}
        ),

        bitmask AS (
            SELECT transition_id, peptide_id, 1 AS bmask
            FROM transitions
            WHERE peptide_id IS NOT NULL
        ),

        counts AS (
            SELECT feature_id, COUNT(*) AS num_peptidoforms
            FROM peptidoforms_real
            GROUP BY feature_id
        )

        SELECT
        COALESCE(e.feature_id, p.feature_id) AS feature_id,
        e.transition_id,
        e.pep,
        p.peptide_id,
        COALESCE(b.bmask, 0)    AS bmask,
        c.num_peptidoforms     AS num_peptidoforms
        FROM
        -- evidence is now just the view itself
        transitions AS e
        FULL OUTER JOIN peptidoforms AS p USING (feature_id)
        LEFT JOIN bitmask AS b
        ON e.transition_id = b.transition_id
        AND p.peptide_id   = b.peptide_id
        JOIN counts AS c USING (feature_id);
        """

        result = con.execute(sql).df().rename(columns=str.lower)

        result = result.drop_duplicates()
        logger.info(f"Loaded total {len(result)} transition-peptidoform entries")

        return result

    def _fetch_alignment_features(self, con) -> pd.DataFrame:
        logger.info("Reading Across Run Feature Alignment Mapping ...")

        pep_threshold = self.config.ipf_max_alignment_pep
        alignment_file = os.path.join(self.infile, "feature_alignment.parquet")

        if not os.path.exists(alignment_file):
            raise click.ClickException(f"Alignment file not found: {alignment_file}")

        # Read alignment file into DuckDB
        con.execute(
            f"""
            CREATE VIEW alignment_features AS 
            SELECT * FROM read_parquet('{alignment_file}')
        """
        )

        query = f"""
            SELECT
                DENSE_RANK() OVER (ORDER BY a.PRECURSOR_ID, a.ALIGNMENT_ID) AS ALIGNMENT_GROUP_ID,
                a.FEATURE_ID AS FEATURE_ID
            FROM alignment_features AS a
            WHERE DECOY = 1
            AND a.SCORE_ALIGNMENT_PEP < {pep_threshold}
            ORDER BY ALIGNMENT_GROUP_ID
        """

        df = con.execute(query).df()
        df.columns = [col.lower() for col in df.columns]

        logger.info(f"Loaded {len(df)} aligned feature group mappings.")
        logger.debug(f"Unique alignment groups: {df['alignment_group_id'].nunique()}")

        return df

    def _fetch_peptide_ids(self, con) -> pd.DataFrame:
        logger.info("Reading Peptidoform Group IDs ...")

        if self.config.file_type == "parquet_split_multi":
            precursor_files = glob.glob(
                os.path.join(self.infile, "*.oswpq", "precursors_features.parquet")
            )
        else:
            precursor_files = [os.path.join(self.infile, "precursors_features.parquet")]

        if not precursor_files:
            raise click.ClickException("No precursors_features.parquet files found.")

        # Read all peptide IDs from the first file
        con.execute(
            f"""
            CREATE OR REPLACE VIEW peptides AS 
            SELECT DISTINCT IPF_PEPTIDE_ID AS PEPTIDE_ID, MODIFIED_SEQUENCE 
            FROM read_parquet({precursor_files})
        """
        )

        df = con.execute("SELECT * FROM peptides").df()

        balanced = r"\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)"
        df["mods_list"] = df["MODIFIED_SEQUENCE"].apply(
            lambda s: re.findall(balanced, s)
        )
        df["bare_seq"] = df["MODIFIED_SEQUENCE"].apply(
            lambda s: re.sub(balanced, "", s)
        )
        df["isoform_signature"] = df["bare_seq"] + df["mods_list"].apply(
            lambda L: "".join(sorted(L))
        )

        df["isoform_id"] = df.groupby("isoform_signature").ngroup()

        df.columns = [col.lower() for col in df.columns]

        logger.info(f"Loaded {len(df)} unique peptide IDs.")
        return df


class SplitParquetWriter(BaseSplitParquetWriter):
    """
    Class for writing OpenSWATH results to a directory containing split Parquet files.

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
        df = result

        # Rename score columns
        df = df.rename(
            columns={
                "PRECURSOR_PEAKGROUP_PEP": "SCORE_IPF_PRECURSOR_PEAKGROUP_PEP",
                "QVALUE": "SCORE_IPF_QVALUE",
                "PEP": "SCORE_IPF_PEP",
            }
        )

        # Identify columns to merge
        score_cols = [
            "SCORE_IPF_PRECURSOR_PEAKGROUP_PEP",
            "SCORE_IPF_QVALUE",
            "SCORE_IPF_PEP",
        ]
        join_cols = ["FEATURE_ID", "PEPTIDE_ID"]

        # Make sure the input dataframe has these columns
        if not all(col in df.columns for col in join_cols + score_cols):
            raise click.ClickException("Missing required columns in result dataframe.")

        # Determine output files to modify
        if self.file_type == "parquet_split_multi":
            run_dirs = [
                os.path.join(self.outfile, d)
                for d in os.listdir(self.outfile)
                if d.endswith(".oswpq") and os.path.isdir(os.path.join(self.outfile, d))
            ]
        else:
            run_dirs = [self.outfile]

        for run_dir in run_dirs:
            file_path = os.path.join(run_dir, "precursors_features.parquet")

            if not os.path.exists(file_path):
                logger.warning(f"File not found, skipping: {file_path}")
                continue

            # Read FEATURE_IDs from current file
            try:
                con = duckdb.connect()
                feature_ids = con.execute(
                    f"SELECT DISTINCT FEATURE_ID FROM read_parquet('{file_path}')"
                ).fetchall()
                con.close()
            except Exception as e:
                logger.error(f"Error reading FEATURE_IDs from {file_path}: {e}")
                continue

            feature_ids = set(f[0] for f in feature_ids)
            subset = df[df["FEATURE_ID"].isin(feature_ids)]

            if subset.empty:
                logger.warning(
                    f"No matching FEATURE_IDs found for {run_dir}, skipping."
                )
                continue

            # Identify columns to keep from original parquet file
            existing_cols = get_parquet_column_names(file_path)
            score_ipf_cols = [
                col for col in existing_cols if col.startswith("SCORE_IPF")
            ]
            if score_ipf_cols:
                logger.warning(
                    "Warn: There are existing SCORE_IPF_ columns, these will be dropped."
                )
            existing_cols = [
                col for col in existing_cols if not col.startswith("SCORE_IPF_")
            ]
            select_old = ", ".join([f"p.{col}" for col in existing_cols])
            new_score_sql = ", ".join([f"s.{col}" for col in score_cols])

            con = duckdb.connect()
            con.register("scores", pa.Table.from_pandas(subset))

            # Validate input row entry count and joined entry count remain the same
            self._validate_row_count_after_join(
                con,
                file_path,
                "p.FEATURE_ID, p.IPF_PEPTIDE_ID",
                "p.FEATURE_ID = s.FEATURE_ID AND p.IPF_PEPTIDE_ID = s.PEPTIDE_ID",
                "p",
            )

            con.execute(
                f"""
                COPY (
                    SELECT {select_old}, {new_score_sql}
                    FROM read_parquet('{file_path}') p
                    LEFT JOIN scores s
                    ON p.FEATURE_ID = s.FEATURE_ID AND p.IPF_PEPTIDE_ID = s.PEPTIDE_ID
                ) TO '{file_path}'
                (FORMAT 'parquet', COMPRESSION 'ZSTD', COMPRESSION_LEVEL 11)
                """
            )

            logger.debug(
                f"After appendings scores, {file_path} has {self._get_parquet_row_count(con, file_path)} entries"
            )

            con.close()
            logger.success(f"Updated: {file_path}")
