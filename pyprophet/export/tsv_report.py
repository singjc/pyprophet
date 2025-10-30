"""
Module for generating reports from TSV/CSV export files.

This module provides functionality to create simplified PDF reports from
pyprophet export tsv output files, including:
- ID counts (precursor, peptide, protein levels)
- Quantification plots (violin/box plots)
- CV distribution plots
- Intensity correlation heatmaps
- Jaccard similarity heatmaps
"""

import os
import pandas as pd
from pathlib import Path
from loguru import logger
from matplotlib.backends.backend_pdf import PdfPages

from ..report import PlotGenerator
from ..io._base import BaseOSWWriter


def export_tsv_report(
    infile: str,
    outfile: str = None,
    top_n: int = 3,
    consistent_top: bool = True,
    color_palette: str = "normal",
):
    """
    Generate a report from a TSV/CSV export file.

    Parameters
    ----------
    infile : str
        Path to input TSV or CSV file (output from pyprophet export tsv).
    outfile : str, optional
        Path to output PDF file. If None, derives from infile.
    top_n : int, default=3
        Number of top features to use for peptide/protein summarization.
    consistent_top : bool, default=True
        Whether to use same top features across all runs.
    color_palette : str, default="normal"
        Color palette for plots: "normal", "protan", "deutran", or "tritan".
    """
    # Detect if input is CSV or TSV
    if infile.endswith(".csv"):
        sep = ","
        file_ext = ".csv"
    else:
        sep = "\t"
        file_ext = ".tsv"

    # Read the input file
    logger.info(f"Reading input file: {infile}")
    try:
        df = pd.read_csv(infile, sep=sep)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        raise

    # Validate input data
    required_columns = ["filename"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input file is missing required columns: {missing_cols}. "
            "This does not appear to be a valid pyprophet TSV export file."
        )

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Generate output filename if not provided
    if outfile is None:
        outfile = infile.replace(file_ext, "_report.pdf")

    logger.info(f"Generating report: {outfile}")

    # Create the report
    _generate_tsv_report(
        df=df,
        outfile=outfile,
        top_n=top_n,
        consistent_top=consistent_top,
        color_palette=color_palette,
    )

    logger.success(f"Report generated successfully: {outfile}")


def _generate_tsv_report(
    df: pd.DataFrame,
    outfile: str,
    top_n: int,
    consistent_top: bool,
    color_palette: str,
):
    """
    Generate the actual report PDF.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe from TSV export.
    outfile : str
        Path to output PDF file.
    top_n : int
        Number of top features for summarization.
    consistent_top : bool
        Whether to use consistent top features.
    color_palette : str
        Color palette for plots.
    """
    # Standardize column names to lowercase for consistency
    df.columns = df.columns.str.lower()

    # Initialize plotter
    plotter = PlotGenerator(color_palette)

    with PdfPages(outfile) as pdf:
        # Create summary tables for each level
        _create_tsv_summary_table(pdf, df)

        # Generate plots for each level that has data
        # Precursor level
        if _has_precursor_data(df):
            logger.info("Generating precursor-level plots")
            _plot_level(pdf, plotter, df, "precursor", top_n, consistent_top)

        # Peptide level
        if _has_peptide_data(df):
            logger.info("Generating peptide-level plots")
            _plot_level(pdf, plotter, df, "peptide", top_n, consistent_top)

        # Protein level
        if _has_protein_data(df):
            logger.info("Generating protein-level plots")
            _plot_level(pdf, plotter, df, "protein", top_n, consistent_top)


def _has_precursor_data(df: pd.DataFrame) -> bool:
    """Check if dataframe has precursor-level data."""
    # Need columns for precursor identification
    required = ["filename"]
    return all(col in df.columns for col in required)


def _has_peptide_data(df: pd.DataFrame) -> bool:
    """Check if dataframe has peptide-level data."""
    # Need sequence or peptide identifiers
    return any(
        col in df.columns for col in ["sequence", "fullpeptidename", "modifiedpeptide"]
    )


def _has_protein_data(df: pd.DataFrame) -> bool:
    """Check if dataframe has protein-level data."""
    # Need protein identifiers
    return "proteinname" in df.columns or "protein" in df.columns


def _create_tsv_summary_table(pdf, df: pd.DataFrame):
    """
    Create a summary table of identification counts.

    Parameters
    ----------
    pdf : PdfPages
        PDF file to write to.
    df : pd.DataFrame
        Input dataframe.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Count unique identifications at each level per run
    summary_data = []

    runs = df["filename"].unique() if "filename" in df.columns else []

    for run in runs:
        run_df = df[df["filename"] == run]
        row = {"Run": os.path.basename(run)}

        # Precursor count
        precursor_cols = [
            "transition_group_id",
            "transitiongroupid",
            "precursor_id",
            "precursorid",
        ]
        precursor_col = next(
            (col for col in precursor_cols if col in run_df.columns), None
        )
        if precursor_col:
            row["Precursors"] = run_df[precursor_col].nunique()

        # Peptide count
        peptide_cols = ["sequence", "modifiedpeptide", "fullpeptidename"]
        peptide_col = next(
            (col for col in peptide_cols if col in run_df.columns), None
        )
        if peptide_col:
            row["Peptides"] = run_df[peptide_col].nunique()

        # Protein count
        protein_cols = ["proteinname", "protein"]
        protein_col = next(
            (col for col in protein_cols if col in run_df.columns), None
        )
        if protein_col:
            # Handle protein groups (separated by semicolons or slashes)
            proteins = set()
            for prot_str in run_df[protein_col].dropna():
                if isinstance(prot_str, str):
                    # Split by common separators
                    for sep in [";", "/", ","]:
                        if sep in prot_str:
                            proteins.update(prot_str.split(sep))
                            break
                    else:
                        proteins.add(prot_str)
            row["Proteins"] = len(proteins)

        summary_data.append(row)

    # Create summary table figure
    fig, ax = plt.subplots(figsize=(12, max(4, len(summary_data) * 0.4 + 2)))
    ax.axis("tight")
    ax.axis("off")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Add total row
        total_row = {"Run": "TOTAL"}
        for col in summary_df.columns:
            if col != "Run":
                total_row[col] = summary_df[col].sum()
        summary_df = pd.concat(
            [summary_df, pd.DataFrame([total_row])], ignore_index=True
        )

        # Create table
        table = ax.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Style total row
        total_idx = len(summary_df)
        for i in range(len(summary_df.columns)):
            table[(total_idx, i)].set_facecolor("#E7E6E6")
            table[(total_idx, i)].set_text_props(weight="bold")

    ax.set_title("Identification Summary", fontsize=14, weight="bold", pad=20)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_level(
    pdf, plotter: PlotGenerator, df: pd.DataFrame, level: str, top_n: int, consistent_top: bool
):
    """
    Generate plots for a specific level.

    Parameters
    ----------
    pdf : PdfPages
        PDF file to write to.
    plotter : PlotGenerator
        Plotter instance for creating plots.
    df : pd.DataFrame
        Input dataframe.
    level : str
        Level to plot: "precursor", "peptide", or "protein".
    top_n : int
        Number of top features for summarization.
    consistent_top : bool
        Whether to use consistent top features.
    """
    import matplotlib.pyplot as plt

    # Prepare data at the specified level
    plot_df = _prepare_level_data(df, level, top_n, consistent_top)

    if plot_df is None or len(plot_df) == 0:
        logger.warning(f"No data available for {level} level")
        return

    # Determine ID key for this level
    if level == "precursor":
        id_key = "precursor_id"
    elif level == "peptide":
        id_key = "peptide_id"
    else:  # protein
        id_key = "protein_id"

    # Ensure ID key exists in dataframe
    if id_key not in plot_df.columns:
        logger.warning(f"Missing {id_key} column for {level} level plots")
        return

    title = f"{level.capitalize()} Level Quantification"

    # First set of plots: ID barplot, ID consistency, violin plot, CV distribution
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=14)

        plotter.add_id_barplot(axes[0, 0], plot_df, id_key)
        plotter.plot_identification_consistency(axes[0, 1], plot_df, id_key)
        plotter.add_violinplot(axes[1, 0], plot_df, id_key)
        plotter.plot_cv_distribution(axes[1, 1], plot_df, id_key)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to generate first set of {level} plots: {str(e)}")

    # Second set of plots: Jaccard similarity and intensity correlation
    try:
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))

        plotter.plot_jaccard_similarity(axes[0], plot_df, id_key)
        plotter.plot_intensity_correlation(axes[1], plot_df, id_key)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to generate second set of {level} plots: {str(e)}")


def _prepare_level_data(
    df: pd.DataFrame, level: str, top_n: int, consistent_top: bool
) -> pd.DataFrame:
    """
    Prepare data for a specific level, including summarization if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    level : str
        Level to prepare: "precursor", "peptide", or "protein".
    top_n : int
        Number of top features for summarization.
    consistent_top : bool
        Whether to use consistent top features.

    Returns
    -------
    pd.DataFrame
        Prepared dataframe for plotting with standardized column names.
    """
    # Make a copy to avoid modifying original
    work_df = df.copy()

    # Standardize column names
    col_mapping = {
        "transition_group_id": "precursor_id",
        "transitiongroupid": "precursor_id",
        "fullpeptidename": "peptide_id",
        "sequence": "sequence",
        "modifiedpeptide": "peptide_id",
        "proteinname": "protein_id",
        "protein": "protein_id",
        "intensity": "area_intensity",
    }

    for old_name, new_name in col_mapping.items():
        if old_name in work_df.columns and new_name not in work_df.columns:
            work_df[new_name] = work_df[old_name]

    # Map filename to run_id (required by PlotGenerator)
    if "filename" in work_df.columns and "run_id" not in work_df.columns:
        # Create numeric run IDs while preserving order
        unique_files = work_df["filename"].unique()
        file_to_id = {f: i for i, f in enumerate(unique_files)}
        work_df["run_id"] = work_df["filename"].map(file_to_id)

    # Ensure we have area_intensity column (required by PlotGenerator)
    if "area_intensity" not in work_df.columns:
        if "intensity" in work_df.columns:
            work_df["area_intensity"] = work_df["intensity"]
        else:
            logger.warning("No intensity column found in data")
            return None

    # Filter out zero/NA intensities
    work_df = work_df[work_df["area_intensity"] > 0].copy()

    if level == "precursor":
        # For precursor level, just ensure we have the right ID
        if "precursor_id" not in work_df.columns:
            logger.warning("No precursor ID column found")
            return None
        # Keep only best peak group per precursor per run if m_score exists
        if "m_score" in work_df.columns:
            work_df = work_df.loc[
                work_df.groupby(["filename", "precursor_id"])["m_score"].idxmin()
            ]
        return work_df

    elif level == "peptide":
        # Summarize from precursor to peptide level
        if "peptide_id" not in work_df.columns:
            # Try to create from sequence
            if "sequence" in work_df.columns:
                work_df["peptide_id"] = work_df["sequence"]
            else:
                logger.warning("No peptide ID or sequence column found")
                return None

        # Keep only best peak group per precursor per run if m_score exists
        if "precursor_id" in work_df.columns and "m_score" in work_df.columns:
            work_df = work_df.loc[
                work_df.groupby(["filename", "precursor_id"])["m_score"].idxmin()
            ]

        # Select top precursors for each peptide
        if consistent_top and "precursor_id" in work_df.columns:
            # Use precursors with highest median intensity across all runs
            median_intensity = (
                work_df.groupby(["precursor_id", "peptide_id"])["area_intensity"]
                .median()
                .reset_index()
            )
            top_precursors = (
                median_intensity.groupby("peptide_id")
                .apply(lambda x: x.nlargest(top_n, "area_intensity")["precursor_id"])
                .reset_index()["precursor_id"]
            )
            work_df = work_df[work_df["precursor_id"].isin(top_precursors)]

        # Aggregate to peptide level (mean of top precursors)
        agg_df = (
            work_df.groupby(["peptide_id", "filename", "run_id"])["area_intensity"]
            .mean()
            .reset_index()
        )
        return agg_df

    elif level == "protein":
        # Summarize from peptide to protein level
        # First, go to peptide level
        peptide_df = _prepare_level_data(df, "peptide", top_n, consistent_top)
        if peptide_df is None:
            return None

        # Get protein mapping
        if "protein_id" not in work_df.columns:
            logger.warning("No protein ID column found")
            return None

        # Create peptide to protein mapping
        prot_map = (
            work_df[["peptide_id", "protein_id"]]
            .drop_duplicates()
            .set_index("peptide_id")["protein_id"]
        )

        # Merge with peptide data
        peptide_df = peptide_df.merge(
            prot_map.reset_index(), on="peptide_id", how="left"
        )

        # Handle protein groups - explode if needed
        # (For simplicity, we'll just use the first protein in groups)
        if peptide_df["protein_id"].dtype == object:
            # Split protein groups
            peptide_df["protein_id"] = peptide_df["protein_id"].str.split(";").str[0]

        if consistent_top:
            # Calculate median intensity for each peptide
            peptide_df["median_intensity"] = peptide_df.groupby("peptide_id")[
                "area_intensity"
            ].transform("median")

            # Get top N peptides per protein
            top_peptides = (
                peptide_df.groupby("protein_id")
                .apply(lambda x: x.nlargest(top_n, "median_intensity")["peptide_id"])
                .reset_index()["peptide_id"]
            )
            peptide_df = peptide_df[peptide_df["peptide_id"].isin(top_peptides)]
            peptide_df = peptide_df.drop(columns=["median_intensity"])

        # Aggregate to protein level (mean of top peptides)
        agg_df = (
            peptide_df.groupby(["protein_id", "filename", "run_id"])["area_intensity"]
            .mean()
            .reset_index()
        )
        return agg_df

    return None
