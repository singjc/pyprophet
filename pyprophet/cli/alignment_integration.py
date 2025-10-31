"""
CLI command for alignment integration.

This module implements the alignment-integration subcommand that computes
adjusted PEPs and q-values by combining MS2 scores with alignment scores.
"""

import os
import sqlite3
import click
import pandas as pd
import numpy as np
from loguru import logger

from .._config import ExportIOConfig
from ..ipf import compute_model_fdr
from ..io.util import check_sqlite_table
from .util import AdvancedHelpCommand, measure_memory_usage_and_time


@click.command(name="alignment-integration", cls=AdvancedHelpCommand)
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file (OSW, Parquet, or Split Parquet).",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="Output file (optional, defaults to modifying input file).",
)
@click.option(
    "--max_alignment_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Maximum PEP to consider for good alignments.",
)
@measure_memory_usage_and_time
def alignment_integration(infile, outfile, max_alignment_pep):
    """
    Integrate alignment results to compute adjusted PEPs and q-values.
    
    This command combines MS2 scores with alignment scores to:
    1. Compute adjusted PEPs: pep_adj = 1 - (1 - pep_ms2) * (1 - pep_align)
    2. Re-rank features within (run_id, precursor) groups
    3. Compute new q-values using model-based FDR
    
    Results are saved back to the input file (or output file if specified):
    - OSW: Creates ALIGNMENT_MS2_SCORE table
    - Parquet/Split Parquet: Adds columns to precursors_features.parquet
    """
    if outfile is None:
        outfile = infile
    
    # Determine file type and process accordingly
    if infile.endswith(".osw"):
        _process_osw(infile, outfile, max_alignment_pep)
    elif infile.endswith(".parquet"):
        _process_parquet(infile, outfile, max_alignment_pep)
    elif os.path.isdir(infile):
        # Check if it's split parquet format
        oswpq_dirs = [f for f in os.listdir(infile) if f.endswith(".oswpq")]
        if oswpq_dirs:
            _process_split_parquet(infile, outfile, max_alignment_pep)
        else:
            raise click.ClickException(
                f"Directory {infile} does not appear to be a split parquet format"
            )
    else:
        raise click.ClickException(
            f"Unsupported file format: {infile}. Must be .osw, .parquet, or split parquet directory."
        )


def _process_osw(infile, outfile, max_alignment_pep):
    """Process OSW SQLite file."""
    logger.info(f"Processing OSW file: {infile}")
    
    # Copy file if output is different
    if infile != outfile:
        import shutil
        logger.info(f"Copying {infile} to {outfile}")
        shutil.copy2(infile, outfile)
    
    con = sqlite3.connect(outfile)
    
    # Check if alignment tables exist
    if not check_sqlite_table(con, "FEATURE_MS2_ALIGNMENT"):
        logger.warning("FEATURE_MS2_ALIGNMENT table not found. Skipping alignment integration.")
        con.close()
        return
    
    if not check_sqlite_table(con, "SCORE_ALIGNMENT"):
        logger.warning("SCORE_ALIGNMENT table not found. Skipping alignment integration.")
        con.close()
        return
    
    if not check_sqlite_table(con, "SCORE_MS2"):
        logger.error("SCORE_MS2 table not found. Cannot perform alignment integration without MS2 scores.")
        con.close()
        raise click.ClickException("SCORE_MS2 table required but not found")
    
    logger.info("Fetching MS2 scores for all features...")
    
    # Fetch all MS2 scores (not filtered by qvalue)
    ms2_query = """
        SELECT 
            FEATURE.ID AS feature_id,
            FEATURE.RUN_ID AS run_id,
            FEATURE.PRECURSOR_ID AS precursor_id,
            PRECURSOR.DECOY AS decoy,
            SCORE_MS2.PEP AS ms2_pep,
            SCORE_MS2.QVALUE AS ms2_qvalue,
            SCORE_MS2.RANK AS ms2_rank
        FROM FEATURE
        INNER JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID
        INNER JOIN PRECURSOR ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
    """
    ms2_scores = pd.read_sql_query(ms2_query, con)
    logger.info(f"Loaded {len(ms2_scores)} MS2 scored features")
    
    # Fetch alignment scores
    logger.info("Fetching alignment scores...")
    alignment_query = f"""
        SELECT 
            FEATURE_MS2_ALIGNMENT.ALIGNED_FEATURE_ID AS feature_id,
            FEATURE_MS2_ALIGNMENT.REFERENCE_FEATURE_ID AS reference_feature_id,
            SCORE_ALIGNMENT.PEP AS alignment_pep
        FROM FEATURE_MS2_ALIGNMENT
        INNER JOIN SCORE_ALIGNMENT ON SCORE_ALIGNMENT.FEATURE_ID = FEATURE_MS2_ALIGNMENT.ALIGNED_FEATURE_ID
        WHERE FEATURE_MS2_ALIGNMENT.LABEL = 1
        AND SCORE_ALIGNMENT.PEP < {max_alignment_pep}
    """
    alignment_scores = pd.read_sql_query(alignment_query, con)
    logger.info(f"Loaded {len(alignment_scores)} alignment scores passing PEP < {max_alignment_pep}")
    
    # Compute adjusted PEPs and q-values
    logger.info("Computing adjusted PEPs and q-values...")
    result = _compute_adjusted_scores(ms2_scores, alignment_scores)
    
    # Save results to database
    logger.info("Saving results to ALIGNMENT_MS2_SCORE table...")
    _save_osw_results(con, result)
    
    con.close()
    logger.success(f"Alignment integration complete. Results saved to {outfile}")


def _process_parquet(infile, outfile, max_alignment_pep):
    """Process single Parquet file."""
    logger.info(f"Processing Parquet file: {infile}")
    
    # Check for alignment file
    base = infile[:-8] if infile.endswith(".parquet") else infile
    alignment_file = f"{base}_feature_alignment.parquet"
    
    if not os.path.exists(alignment_file):
        logger.warning(f"Alignment file not found: {alignment_file}. Skipping alignment integration.")
        return
    
    # Read main parquet file
    logger.info(f"Reading {infile}...")
    df = pd.read_parquet(infile)
    
    # Check for required columns
    if "SCORE_MS2_PEP" not in df.columns or "SCORE_MS2_Q_VALUE" not in df.columns:
        logger.error("Required MS2 score columns not found in parquet file.")
        raise click.ClickException("SCORE_MS2_PEP and SCORE_MS2_Q_VALUE columns required")
    
    # Read alignment file
    logger.info(f"Reading alignment file: {alignment_file}")
    alignment_df = pd.read_parquet(alignment_file)
    
    # Prepare data for computation
    ms2_scores = df[["FEATURE_ID", "RUN_ID", "PRECURSOR_ID", "DECOY", "SCORE_MS2_PEP", "SCORE_MS2_Q_VALUE", "SCORE_MS2_PEAK_GROUP_RANK"]].copy()
    ms2_scores.columns = ["feature_id", "run_id", "precursor_id", "decoy", "ms2_pep", "ms2_qvalue", "ms2_rank"]
    
    # Filter alignment scores
    alignment_scores = alignment_df[
        (alignment_df["DECOY"] == 1) & 
        (alignment_df["PEP"] < max_alignment_pep)
    ][["FEATURE_ID", "REFERENCE_FEATURE_ID", "PEP"]].copy()
    alignment_scores.columns = ["feature_id", "reference_feature_id", "alignment_pep"]
    
    logger.info(f"Loaded {len(alignment_scores)} alignment scores passing PEP < {max_alignment_pep}")
    
    # Compute adjusted scores
    logger.info("Computing adjusted PEPs and q-values...")
    result = _compute_adjusted_scores(ms2_scores, alignment_scores)
    
    # Merge results back to dataframe
    logger.info("Merging results back to parquet file...")
    result_cols = result[["feature_id", "alignment_ms2_pep", "alignment_ms2_q_value", "alignment_ms2_peak_group_rank"]]
    df = df.merge(result_cols, left_on="FEATURE_ID", right_on="feature_id", how="left")
    df = df.drop(columns=["feature_id"])
    
    # Rename columns to match expected format
    df = df.rename(columns={
        "alignment_ms2_pep": "ALIGNMENT_MS2_PEP",
        "alignment_ms2_q_value": "ALIGNMENT_MS2_Q_VALUE",
        "alignment_ms2_peak_group_rank": "ALIGNMENT_MS2_PEAK_GROUP_RANK"
    })
    
    # Save to output file
    logger.info(f"Saving results to {outfile}...")
    df.to_parquet(outfile, index=False)
    
    logger.success(f"Alignment integration complete. Results saved to {outfile}")


def _process_split_parquet(infile, outfile, max_alignment_pep):
    """Process split Parquet directory."""
    logger.info(f"Processing split Parquet directory: {infile}")
    
    # Check for alignment file
    alignment_file = os.path.join(infile, "feature_alignment.parquet")
    if not os.path.exists(alignment_file):
        logger.warning(f"Alignment file not found: {alignment_file}. Skipping alignment integration.")
        return
    
    # Get list of precursor files
    import glob
    precursor_files = glob.glob(os.path.join(infile, "*.oswpq", "precursors_features.parquet"))
    
    if not precursor_files:
        logger.error("No precursors_features.parquet files found in split parquet directory")
        raise click.ClickException("No precursor feature files found")
    
    logger.info(f"Found {len(precursor_files)} precursor feature files")
    
    # Read alignment file
    logger.info(f"Reading alignment file: {alignment_file}")
    alignment_df = pd.read_parquet(alignment_file)
    
    # Filter alignment scores
    alignment_scores = alignment_df[
        (alignment_df["DECOY"] == 1) & 
        (alignment_df["PEP"] < max_alignment_pep)
    ][["FEATURE_ID", "REFERENCE_FEATURE_ID", "PEP"]].copy()
    alignment_scores.columns = ["feature_id", "reference_feature_id", "alignment_pep"]
    
    logger.info(f"Loaded {len(alignment_scores)} alignment scores passing PEP < {max_alignment_pep}")
    
    # Process each precursor file
    for precursor_file in precursor_files:
        logger.info(f"Processing {precursor_file}...")
        df = pd.read_parquet(precursor_file)
        
        # Check for required columns
        if "SCORE_MS2_PEP" not in df.columns or "SCORE_MS2_Q_VALUE" not in df.columns:
            logger.warning(f"Required MS2 score columns not found in {precursor_file}, skipping...")
            continue
        
        # Prepare data for computation
        ms2_scores = df[["FEATURE_ID", "RUN_ID", "PRECURSOR_ID", "DECOY", "SCORE_MS2_PEP", "SCORE_MS2_Q_VALUE", "SCORE_MS2_PEAK_GROUP_RANK"]].copy()
        ms2_scores.columns = ["feature_id", "run_id", "precursor_id", "decoy", "ms2_pep", "ms2_qvalue", "ms2_rank"]
        
        # Filter alignment scores for this run
        run_ids = ms2_scores["run_id"].unique()
        run_alignment_scores = alignment_scores[alignment_scores["feature_id"].isin(ms2_scores["feature_id"])]
        
        if len(run_alignment_scores) == 0:
            logger.info(f"No alignment scores for features in {precursor_file}, skipping...")
            continue
        
        # Compute adjusted scores
        logger.info(f"Computing adjusted scores for {len(ms2_scores)} features...")
        result = _compute_adjusted_scores(ms2_scores, run_alignment_scores)
        
        # Merge results back to dataframe
        result_cols = result[["feature_id", "alignment_ms2_pep", "alignment_ms2_q_value", "alignment_ms2_peak_group_rank"]]
        df = df.merge(result_cols, left_on="FEATURE_ID", right_on="feature_id", how="left")
        df = df.drop(columns=["feature_id"])
        
        # Rename columns to match expected format
        df = df.rename(columns={
            "alignment_ms2_pep": "ALIGNMENT_MS2_PEP",
            "alignment_ms2_q_value": "ALIGNMENT_MS2_Q_VALUE",
            "alignment_ms2_peak_group_rank": "ALIGNMENT_MS2_PEAK_GROUP_RANK"
        })
        
        # Determine output file path
        if outfile != infile:
            # Create output directory structure
            rel_path = os.path.relpath(precursor_file, infile)
            out_precursor_file = os.path.join(outfile, rel_path)
            os.makedirs(os.path.dirname(out_precursor_file), exist_ok=True)
        else:
            out_precursor_file = precursor_file
        
        # Save to output file
        logger.info(f"Saving results to {out_precursor_file}...")
        df.to_parquet(out_precursor_file, index=False)
    
    logger.success(f"Alignment integration complete for all runs. Results saved to {outfile}")


def _compute_adjusted_scores(ms2_scores, alignment_scores):
    """
    Compute adjusted PEPs and q-values by combining MS2 and alignment scores.
    
    Args:
        ms2_scores: DataFrame with columns [feature_id, run_id, precursor_id, decoy, ms2_pep, ms2_qvalue, ms2_rank]
        alignment_scores: DataFrame with columns [feature_id, reference_feature_id, alignment_pep]
    
    Returns:
        DataFrame with adjusted scores
    """
    # Merge alignment scores into MS2 scores
    df = ms2_scores.merge(alignment_scores, on="feature_id", how="left")
    
    # For reference features (those pointed to by aligned features), set alignment_pep = 1.0
    reference_feature_ids = alignment_scores["reference_feature_id"].unique()
    df.loc[df["feature_id"].isin(reference_feature_ids) & df["alignment_pep"].isna(), "alignment_pep"] = 1.0
    
    # For features without alignment info, set alignment_pep = 1.0 (neutral)
    df["alignment_pep"] = df["alignment_pep"].fillna(1.0)
    
    # Clip PEPs to avoid numerical issues
    epsilon = 1e-10
    df["ms2_pep"] = df["ms2_pep"].clip(epsilon, 1 - epsilon)
    df["alignment_pep"] = df["alignment_pep"].clip(epsilon, 1 - epsilon)
    
    # Compute adjusted PEP: pep_adj = 1 - (1 - pep_ms2) * (1 - pep_align)
    df["alignment_ms2_pep"] = 1 - (1 - df["ms2_pep"]) * (1 - df["alignment_pep"])
    
    # Filter to target features only
    target_df = df[df["decoy"] == 0].copy()
    
    # Within each (run_id, precursor_id) group, keep top-1 by adjusted PEP
    logger.info("Ranking features within (run_id, precursor_id) groups...")
    target_df["alignment_ms2_peak_group_rank"] = target_df.groupby(["run_id", "precursor_id"])["alignment_ms2_pep"].rank(method="first")
    
    # Get top-1 features for q-value computation
    top1_features = target_df[target_df["alignment_ms2_peak_group_rank"] == 1].copy()
    
    # Compute q-values using model-based FDR
    logger.info(f"Computing q-values for {len(top1_features)} top-1 features...")
    top1_features["alignment_ms2_q_value"] = compute_model_fdr(top1_features["alignment_ms2_pep"].values)
    
    # Create a mapping of feature_id to q-value for top-1 features
    qvalue_map = dict(zip(top1_features["feature_id"], top1_features["alignment_ms2_q_value"]))
    
    # For non-top-1 features, assign q-value based on their group's top-1 feature
    # This ensures all features in a group share the same q-value (optional - can be modified)
    # For now, we'll compute q-values for all features based on their adjusted PEP
    target_df["alignment_ms2_q_value"] = compute_model_fdr(target_df["alignment_ms2_pep"].values)
    
    # Merge back to include decoys
    result = df[["feature_id", "run_id", "precursor_id", "decoy"]].merge(
        target_df[["feature_id", "alignment_ms2_pep", "alignment_ms2_q_value", "alignment_ms2_peak_group_rank"]],
        on="feature_id",
        how="left"
    )
    
    return result


def _save_osw_results(con, result):
    """Save results to OSW database."""
    # Drop existing table if it exists
    con.execute("DROP TABLE IF EXISTS ALIGNMENT_MS2_SCORE")
    
    # Create new table
    create_table_sql = """
        CREATE TABLE ALIGNMENT_MS2_SCORE (
            FEATURE_ID INTEGER NOT NULL,
            RANK INTEGER,
            PEP REAL,
            QVALUE REAL,
            PRIMARY KEY (FEATURE_ID)
        )
    """
    con.execute(create_table_sql)
    
    # Prepare data for insertion (only target features with valid scores)
    save_df = result[result["decoy"] == 0].copy()
    save_df = save_df[save_df["alignment_ms2_pep"].notna()]
    save_df = save_df[["feature_id", "alignment_ms2_peak_group_rank", "alignment_ms2_pep", "alignment_ms2_q_value"]]
    save_df.columns = ["FEATURE_ID", "RANK", "PEP", "QVALUE"]
    
    # Insert data
    logger.info(f"Inserting {len(save_df)} records into ALIGNMENT_MS2_SCORE table...")
    save_df.to_sql("ALIGNMENT_MS2_SCORE", con, if_exists="append", index=False)
    
    # Create index
    con.execute("CREATE INDEX IF NOT EXISTS idx_alignment_ms2_score_feature_id ON ALIGNMENT_MS2_SCORE (FEATURE_ID)")
    
    con.commit()
    logger.success(f"Successfully created ALIGNMENT_MS2_SCORE table with {len(save_df)} records")
