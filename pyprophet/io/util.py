"""
This module provides utility functions for handling tsv files, SQLite databases, Parquet files,
and directory structures. It includes functions for file validation, schema inspection,
and logging of file structures, which are commonly used in workflows involving
data processing and analysis.

Functions:
    - is_tsv_file(file_path): Checks if a file is likely a TSV file based on its extension and content.
    - is_sqlite_file(filename): Checks if a file is a valid SQLite database.
    - check_sqlite_table(con, table): Verifies if a table exists in a SQLite database.
    - write_scores_sql_command(con, score_sql, feature_name, var_replacement):
      Constructs an SQL command to select specific scores from a feature table.
    - check_duckdb_table(con, schema, table): Checks if a table exists in a DuckDB-attached SQLite schema.
    - create_index_if_not_exists(con, index_name, table_name, column_name):
      Creates an index on a table if it does not already exist.
    - is_parquet_file(file_path): Validates if a file is a Parquet file.
    - is_valid_single_split_parquet_dir(path): Checks if a directory contains a valid single-run split Parquet structure.
    - is_valid_multi_split_parquet_dir(path): Checks if a directory contains multiple subdirectories with split Parquet files.
    - get_parquet_column_names(file_path): Retrieves column names from a Parquet file without reading the entire file.
    - print_parquet_tree(root_dir, precursors, transitions, alignment=None, max_runs=10):
      Prints the structure of Parquet files in a tree-like format.
    - unimod_to_codename(seq): Converts a sequence with unimod modifications to a codename.

Key Features:
    - SQLite Utilities: Functions for validating SQLite files, checking table existence, and creating indexes.
    - Parquet Utilities: Functions for validating Parquet files, retrieving schema information, and inspecting directory structures.
    - Logging: Provides detailed logging of file structures and validation results using the `loguru` logger.

Dependencies:
    - os
    - collections (defaultdict)
    - click
    - pandas
    - pyopenms
    - pyarrow.parquet
    - loguru

Usage:
    This module is designed to be used as a helper library for workflows involving
    SQLite and Parquet file processing. It can be imported and its functions called
    directly in scripts or pipelines.
"""

import os
from collections import defaultdict
import sqlite3
import importlib
from typing import Type
import duckdb
import click
import pandas as pd
import pyopenms as poms
from loguru import logger


def _ensure_pyarrow():
    """
    Avoid importing pyarrow at module import time; import lazily in functions that need it.
    """
    try:
        import pyarrow as pa  # pylint: disable=C0415

        # Ensure the parquet submodule is loaded and available as pa.parquet
        try:
            import pyarrow.parquet as pq  # pylint: disable=C0415
        except Exception:
            pq = None
        from pyarrow.lib import ArrowInvalid, ArrowIOError  # pylint: disable=C0415

        # Some pyarrow installs don't automatically expose the parquet submodule
        # as an attribute on the top-level module until it's imported. Attach it
        # so callers can use pa.parquet.*
        if pq is not None and not hasattr(pa, "parquet"):
            setattr(pa, "parquet", pq)

        return pa, ArrowInvalid, ArrowIOError
    except ImportError as exc:
        raise click.ClickException(
            "Parquet support requires 'pyarrow'. Install with 'pip install pyarrow' or 'pip install pyprophet[parquet]'."
        ) from exc


def is_tsv_file(file_path):
    """
    Checks if a file is likely a TSV file based on extension and content.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file is likely a TSV file, False otherwise.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return False

    if not file_path.lower().endswith(".tsv") and not file_path.lower().endswith(
        ".txt"
    ):
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if "\t" in line:
                    return True  # Found tab character, likely a TSV
        return False  # No tab character found
    except Exception as e:
        print(f"Error reading file: {e}")
        return False


def is_sqlite_file(filename):
    """
    Check if the given file is a SQLite database file by examining the file header.
    :param filename: The path to the file to be checked.
    :return: True if the file is a SQLite database file, False otherwise.
    """
    # https://stackoverflow.com/questions/12932607/how-to-check-with-python-and-sqlite3-if-one-sqlite-database-file-exists
    from os.path import getsize, isfile

    if not isfile(filename):
        return False
    if getsize(filename) < 100:  # SQLite database file header is 100 bytes
        return False

    with open(filename, "rb") as fd:
        header = fd.read(100)

    if "SQLite format 3" in str(header):
        return True
    else:
        return False


def check_sqlite_table(con, table):
    """
    Check if a table exists in a SQLite database.

    Parameters:
    - con: SQLite connection object
    - table: Name of the table to check for existence

    Returns:
    - table_present: Boolean indicating if the table exists in the database
    """
    table_present = False
    c = con.cursor()
    c.execute(
        f'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="{table}"'
    )
    if c.fetchone()[0] == 1:
        table_present = True
    else:
        table_present = False
    c.fetchall()

    return table_present


def write_scores_sql_command(con, score_sql, feature_name, var_replacement):
    """
    Write SQL command to select specific scores from a given feature table.

    extracts the scores and writes it into an SQL command
    in some cases some post processing has to be performed depending on which
    position the statement should be inserted (e.g. export_compounds.py)

    Parameters:
    - con: Connection object to the database.
    - score_sql: SQL command to select scores.
    - feature_name: Name of the feature table.
    - var_replacement: Replacement string for "VAR" in score names.

    Returns:
    - Updated SQL command with selected scores.
    """
    feature = pd.read_sql_query(f"""PRAGMA table_info({feature_name})""", con)
    score_names_sql = [
        name for name in feature["name"].tolist() if name.startswith("VAR")
    ]
    score_names_lower = [
        name.lower().replace("var_", var_replacement) for name in score_names_sql
    ]
    for i, score_name_sql in enumerate(score_names_sql):
        score_sql = score_sql + str(
            feature_name + "." + score_name_sql + " AS " + score_names_lower[i] + ", "
        )
    return score_sql


def get_table_columns(sqlite_file: str, table: str) -> list:
    """
    Retrieve the column names of a table in a SQLite database file.

    Args:
        sqlite_file (str): Path to the SQLite database file.
        table (str): Name of the table to retrieve column names from.

    Returns:
        list: List of column names in the specified table.
    """
    with sqlite3.connect(sqlite_file) as conn:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def get_table_columns_with_types(sqlite_file: str, table: str) -> list:
    """
    Get the columns and their types for a given table in a SQLite database file.

    Args:
        sqlite_file (str): The path to the SQLite database file.
        table (str): The name of the table to retrieve columns and types from.

    Returns:
        list: A list of tuples where each tuple contains the column name and its data type.
    """
    with sqlite3.connect(sqlite_file) as conn:
        return [(row[1], row[2]) for row in conn.execute(f"PRAGMA table_info({table})")]


def check_table_column_exists(sqlite_file: str, table: str, column: str) -> bool:
    """
    Check if a specific column exists in a table of a SQLite database file.

    Args:
        sqlite_file (str): Path to the SQLite database file.
        table (str): Name of the table to check.
        column (str): Name of the column to check for existence.

    Returns:
        bool: True if the column exists, False otherwise.
    """
    with sqlite3.connect(sqlite_file) as conn:
        columns = get_table_columns(sqlite_file, table)
        return column in columns


def check_duckdb_table(con, schema: str, table: str) -> bool:
    """
    Check if a table exists in a DuckDB-attached SQLite schema (case-insensitive).

    Args:
        con: DuckDB connection.
        schema (str): The schema name (e.g., 'osw').
        table (str): The table name to check.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    query = f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE LOWER(table_schema) = LOWER('{schema}') 
          AND LOWER(table_name) = LOWER('{table}')
    """
    result = con.execute(query).fetchone()[0]
    return result == 1


def create_index_if_not_exists(con, index_name, table_name, column_name):
    """
    Create an index on a table if it does not already exist. For duckdb connections to sqlite files
    """
    res = con.execute(
        f"""
        SELECT count(*) 
        FROM duckdb_indexes() 
        WHERE index_name = '{index_name}' 
        AND table_name = '{table_name}'
    """
    ).fetchone()

    if res[0] == 0:
        con.execute(f"CREATE INDEX {index_name} ON {table_name} ({column_name})")


def _lazy_parquet_class(module_path: str, class_name: str) -> Type:
    """
    Import the given module (relative to this package) and return the class.
    Raises a ClickException with a friendly message if the import fails (e.g. missing pyarrow).
    """
    try:
        mod = importlib.import_module(module_path, package=__package__)
        return getattr(mod, class_name)
    except ModuleNotFoundError as exc:
        # Likely pyarrow or the module itself is missing; user should install the parquet extra.
        raise click.ClickException(
            "Parquet support requires the 'pyarrow' package. "
            "Install it with 'pip install pyarrow' or 'pip install pyprophet[parquet]'."
        ) from exc
    except Exception:
        # Propagate other exceptions (syntax errors, attribute errors) to surface the real problem.
        raise


def _area_from_config(config) -> str:
    """
    Map a config instance to its package area name used in the io package.
    """
    # Avoid importing config classes here to prevent circular imports.
    cname = type(config).__name__
    if cname == "RunnerIOConfig":
        return "scoring"
    if cname == "IPFIOConfig":
        return "ipf"
    if cname == "LevelContextIOConfig":
        return "levels_context"
    if cname == "ExportIOConfig":
        return "export"
    raise ValueError(f"Unsupported config context: {type(config).__name__}")


def _get_parquet_reader_class_for_config(config, split: bool = False) -> Type:
    _, _, _ = _ensure_pyarrow()
    area = _area_from_config(config)
    module = f".{area}.split_parquet" if split else f".{area}.parquet"
    return _lazy_parquet_class(
        module, "SplitParquetReader" if split else "ParquetReader"
    )


def _get_parquet_writer_class_for_config(config, split: bool = False) -> Type:
    _, _, _ = _ensure_pyarrow()
    area = _area_from_config(config)
    module = f".{area}.split_parquet" if split else f".{area}.parquet"
    return _lazy_parquet_class(
        module, "SplitParquetWriter" if split else "ParquetWriter"
    )


def is_parquet_file(file_path):
    """
    Check if the file is a valid Parquet file.
    """

    # First check extension
    if os.path.splitext(file_path)[1].lower() not in (".parquet", ".pq"):
        return False

    # Then verify it's actually a parquet file
    try:
        pa, ArrowInvalid, ArrowIOError = _ensure_pyarrow()  # pylint: disable=C0103
        pa.parquet.read_schema(file_path)
        return True
    except (ArrowInvalid, ArrowIOError, OSError):
        return False


def is_valid_single_split_parquet_dir(path):
    """Check if directory contains single-run split parquet structure."""
    required_files = ["precursors_features.parquet", "transition_features.parquet"]
    return os.path.isdir(path) and all(
        os.path.isfile(os.path.join(path, f)) and is_parquet_file(os.path.join(path, f))
        for f in required_files
    )


def is_valid_multi_split_parquet_dir(path):
    """Check if directory contains multiple subdirectories with split parquet files."""
    if not os.path.isdir(path):
        return False

    required_files = ["precursors_features.parquet", "transition_features.parquet"]
    subdirs = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if d.endswith(".oswpq") and os.path.isdir(os.path.join(path, d))
    ]
    if not subdirs:
        return False

    for subdir in subdirs:
        if not all(
            os.path.isfile(os.path.join(subdir, f))
            and is_parquet_file(os.path.join(subdir, f))
            for f in required_files
        ):
            return False

    return True


def load_sqlite_scanner(conn: duckdb.DuckDBPyConnection):
    """
    Ensures the `sqlite_scanner` extension is installed and loaded in DuckDB.
    """
    try:
        conn.execute("LOAD sqlite_scanner")
    except Exception as e:
        if "Extension 'sqlite_scanner' not found" in str(e):
            try:
                conn.execute("INSTALL sqlite_scanner")
                conn.execute("LOAD sqlite_scanner")
            except Exception as install_error:
                if "already installed but the origin is different" in str(
                    install_error
                ):
                    conn.execute("FORCE INSTALL sqlite_scanner")
                    conn.execute("LOAD sqlite_scanner")
                else:
                    raise install_error
        else:
            raise e


def get_parquet_column_names(file_path):
    """
    Retrieves column names from a Parquet file without reading the entire file.
    """
    try:
        pa, _, _ = _ensure_pyarrow()
        table_schema = pa.parquet.read_schema(file_path)
        return table_schema.names
    except Exception as e:
        print(f"An error occurred while reading schema from '{file_path}': {e}")
        return None


def print_parquet_tree(root_dir, precursors, transitions, alignment=None, max_runs=10):
    """
    Prints the structure of Parquet files in a tree-like format based on the provided root directory, precursor files, transition files, alignment file, and maximum number of runs to display.
    The function groups precursor and transition files by run, sorts the runs, and prints the file structure for each run up to the specified maximum number of runs.
    If there are more runs than the maximum allowed, it indicates the number of collapsed runs.
    If an alignment file is provided, it is also displayed in the structure.
    """

    def group_by_run(files):
        grouped = defaultdict(list)
        for f in files:
            parts = f.strip("/").split(os.sep)
            if len(parts) >= 2:
                grouped[parts[-2]].append(parts[-1])
            else:
                grouped["_root"].append(parts[-1])
        return grouped

    precursor_runs = group_by_run(precursors)
    transition_runs = group_by_run(transitions)

    all_runs = sorted(set(precursor_runs.keys()) | set(transition_runs.keys()))
    logger.info(f"Detected {len(all_runs)} split_parquet run files")
    logger.info("Input Parquet Structure:")
    click.echo(f"└── 📁 {root_dir}")

    runs_to_print = all_runs[:max_runs]
    skipped = len(all_runs) - max_runs

    for run in runs_to_print:
        click.echo(f"    ├── 📁 {run}")
        printed = set()
        if run in precursor_runs:
            for f in sorted(precursor_runs[run]):
                click.echo(f"    │   ├── 📄 {f}")
                printed.add(f)
        if run in transition_runs:
            for f in sorted(transition_runs[run]):
                if f not in printed:
                    click.echo(f"    │   └── 📄 {f}")

    if skipped > 0:
        click.echo(f"    │   ... ({skipped} more run(s) collapsed)")

    if alignment:
        click.echo(f"    └── 📄 {os.path.basename(alignment)}")


def unimod_to_codename(seq):
    """
    Convert a sequence with unimod modifications to a codename.
    """
    seq_poms = poms.AASequence.fromString(seq)
    codename = seq_poms.toString()
    return codename


def compute_adjusted_pep_and_rerank(data: "pd.DataFrame") -> "pd.DataFrame":
    """
    Compute adjusted PEP from MS2 PEP and alignment PEP, then re-rank peak groups.
    
    This function implements the alignment-adjusted posterior error probability calculation:
    1. For reference features (where feature is the alignment reference), set alignment_pep = 1.0
       and skip combining (use pep_ms2 directly as pep_adj)
    2. For aligned features with alignment evidence, compute pep_adj = 1 - (1 - pep_ms2) * (1 - alignment_pep)
    3. For features without alignment evidence, use pep_ms2 as pep_adj
    4. Re-rank peak groups within each (run_id, transition_group_id) by pep_adj
    5. Compute model-based FDR (qvalues) using compute_model_fdr on top-1 pep_adj per group
    
    The function preserves original columns by renaming:
    - m_score -> ms2_m_score (original qvalues from MS2 scoring)
    - peak_group_rank -> ms2_peak_group_rank (original ranking)
    
    And adds new columns:
    - ms2_aligned_adj_pep: The adjusted PEP combining MS2 and alignment evidence
    - m_score: New qvalues computed from adjusted PEPs (model-based FDR)
    - peak_group_rank: New ranking based on adjusted PEPs
    
    Args:
        data: DataFrame with columns: pep, alignment_pep (optional), alignment_reference_feature_id (optional),
              id, run_id, transition_group_id, m_score, peak_group_rank
              
    Returns:
        DataFrame with adjusted PEP, new rankings, and new qvalues
    """
    import pandas as pd
    import numpy as np
    from loguru import logger
    
    # Import compute_model_fdr from ipf module
    from ..ipf import compute_model_fdr
    
    # Check if we have alignment data
    has_alignment = "alignment_pep" in data.columns
    
    if not has_alignment:
        logger.debug("No alignment data present, skipping PEP adjustment")
        return data
    
    # Make a copy to avoid modifying the original
    data = data.copy()
    
    # Rename original columns to preserve them
    if "m_score" in data.columns:
        data.rename(columns={"m_score": "ms2_m_score"}, inplace=True)
    if "peak_group_rank" in data.columns:
        data.rename(columns={"peak_group_rank": "ms2_peak_group_rank"}, inplace=True)
    
    # Step 1: Identify reference features
    # Reference features are those where id appears in alignment_reference_feature_id
    is_reference = pd.Series(False, index=data.index)
    if "alignment_reference_feature_id" in data.columns:
        reference_ids = data[data["alignment_reference_feature_id"].notna()]["alignment_reference_feature_id"].unique()
        is_reference = data["id"].isin(reference_ids)
        logger.debug(f"Identified {is_reference.sum()} reference features")
    
    # Step 2: Compute adjusted PEP
    pep_ms2 = data["pep"].fillna(1.0)
    pep_align = data["alignment_pep"].fillna(1.0)
    
    # Clip to avoid numerical issues (small epsilon to prevent 0 or 1 exactly)
    eps = 1e-10
    pep_ms2 = np.clip(pep_ms2, eps, 1.0 - eps)
    pep_align = np.clip(pep_align, eps, 1.0 - eps)
    
    # Compute adjusted PEP
    # For reference features: use pep_ms2 only (no alignment evidence for themselves)
    # For aligned features with alignment_pep < 1: combine MS2 and alignment evidence
    # For features without alignment: use pep_ms2 only
    pep_adj = pep_ms2.copy()
    
    # Only apply combination where we have actual alignment evidence (alignment_pep < 1.0 and not a reference)
    has_alignment_evidence = (data["alignment_pep"].notna()) & (data["alignment_pep"] < 1.0) & (~is_reference)
    
    if has_alignment_evidence.any():
        # Combine MS2 and alignment evidence for aligned features
        pep_adj.loc[has_alignment_evidence] = 1.0 - (1.0 - pep_ms2.loc[has_alignment_evidence]) * (1.0 - pep_align.loc[has_alignment_evidence])
        logger.info(f"Combined MS2 and alignment evidence for {has_alignment_evidence.sum()} aligned features")
    
    pep_adj = np.clip(pep_adj, eps, 1.0 - eps)
    data["ms2_aligned_adj_pep"] = pep_adj
    
    logger.info("Computed adjusted PEP from MS2 and alignment evidence")
    
    # Step 3: Re-rank within each (run_id, transition_group_id) by pep_adj
    # Lower pep_adj is better, so rank ascending
    # Use min method to handle ties (same rank for equal pep_adj)
    data["peak_group_rank"] = (
        data.groupby(["run_id", "transition_group_id"])["ms2_aligned_adj_pep"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    
    logger.info("Re-ranked peak groups based on adjusted PEP")
    
    # Step 4: Compute model-based FDR (qvalues) from top-1 pep_adj per group
    # Get top-1 features (rank == 1) per (run_id, transition_group_id)
    top1_mask = data["peak_group_rank"] == 1
    top1_data = data[top1_mask].copy()
    
    if len(top1_data) > 0:
        # Compute qvalues using compute_model_fdr on top-1 pep_adj
        top1_qvalues = compute_model_fdr(top1_data["ms2_aligned_adj_pep"].values)
        
        # Create a mapping from (run_id, transition_group_id) to qvalue
        top1_data["m_score"] = top1_qvalues
        qvalue_map = top1_data.set_index(["run_id", "transition_group_id"])["m_score"].to_dict()
        
        # Assign qvalues to all rows based on their group
        data["m_score"] = data.apply(
            lambda row: qvalue_map.get((row["run_id"], row["transition_group_id"]), np.nan),
            axis=1
        )
        
        logger.info(f"Computed model-based FDR (qvalues) for {len(top1_data)} top-ranked features")
    else:
        # No top-1 features, set m_score to NaN
        data["m_score"] = np.nan
        logger.warning("No top-1 features found, m_score set to NaN")
    
    return data
