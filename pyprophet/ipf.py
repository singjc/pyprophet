"""
This module implements the Inference of PeptidoForms (IPF) workflow.

IPF is a statistical framework for inferring peptidoforms (modified peptides)
and their probabilities from mass spectrometry data. The module includes
functions for precursor-level and peptidoform-level inference, Bayesian modeling,
and signal propagation across aligned runs.

Key Features:
    - Precursor-level inference using MS1 and MS2 data.
    - Peptidoform-level inference using transition-level data.
    - Bayesian modeling for posterior probability computation.
    - Signal propagation across aligned runs.
    - Model-based FDR estimation.

Functions:
    - compute_model_fdr: Computes model-based FDR estimates from posterior error probabilities.
    - prepare_precursor_bm: Prepares Bayesian model data for precursor-level inference.
    - transfer_confident_evidence_across_runs: Propagates confident evidence across aligned runs.
    - prepare_transition_bm: Prepares Bayesian model data for transition-level inference.
    - apply_bm: Applies the Bayesian model to compute posterior probabilities.
    - precursor_inference: Conducts precursor-level inference.
    - peptidoform_inference: Conducts peptidoform-level inference.
    - infer_peptidoforms: Orchestrates the IPF workflow.

Classes:
    None
"""

import os
import glob
from itertools import islice
import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from scipy.stats import rankdata
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.backends.backend_pdf import PdfPages

from ._config import IPFIOConfig
from .io.dispatcher import ReaderDispatcher, WriterDispatcher


def compute_model_fdr(data_in):
    """
    Computes model-based FDR estimates from posterior error probabilities.

    Args:
        data_in (array-like): Input posterior error probabilities.

    Returns:
        np.ndarray: FDR estimates for the input data.
    """
    data = np.asarray(data_in)

    # compute model based FDR estimates from posterior error probabilities
    order = np.argsort(data)

    ranks = np.zeros(data.shape[0], dtype=int)
    fdr = np.zeros(data.shape[0])

    # rank data with with maximum ranks for ties
    ranks[order] = rankdata(data[order], method="max")

    # compute FDR/q-value by using cumulative sum of maximum rank for ties
    fdr[order] = data[order].cumsum()[ranks[order] - 1] / ranks[order]

    return fdr


def prepare_precursor_bm(data):
    """
    Prepares Bayesian model data for precursor-level inference.

    Args:
        data (pd.DataFrame): Input data containing MS1 and MS2 precursor probabilities.

    Returns:
        pd.DataFrame: Bayesian model data for precursor-level inference.
    """
    # MS1-level precursors
    ms1_precursor_data = data[
        ["feature_id", "ms2_peakgroup_pep", "ms1_precursor_pep"]
    ].dropna(axis=0, how="any")
    ms1_bm_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "feature_id": ms1_precursor_data["feature_id"],
                    "prior": 1 - ms1_precursor_data["ms2_peakgroup_pep"],
                    "evidence": 1 - ms1_precursor_data["ms1_precursor_pep"],
                    "hypothesis": True,
                }
            ),
            pd.DataFrame(
                {
                    "feature_id": ms1_precursor_data["feature_id"],
                    "prior": ms1_precursor_data["ms2_peakgroup_pep"],
                    "evidence": ms1_precursor_data["ms1_precursor_pep"],
                    "hypothesis": False,
                }
            ),
        ]
    )

    # MS2-level precursors
    ms2_precursor_data = data[
        ["feature_id", "ms2_peakgroup_pep", "ms2_precursor_pep"]
    ].dropna(axis=0, how="any")
    ms2_bm_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "feature_id": ms2_precursor_data["feature_id"],
                    "prior": 1 - ms2_precursor_data["ms2_peakgroup_pep"],
                    "evidence": 1 - ms2_precursor_data["ms2_precursor_pep"],
                    "hypothesis": True,
                }
            ),
            pd.DataFrame(
                {
                    "feature_id": ms2_precursor_data["feature_id"],
                    "prior": ms2_precursor_data["ms2_peakgroup_pep"],
                    "evidence": ms2_precursor_data["ms2_precursor_pep"],
                    "hypothesis": False,
                }
            ),
        ]
    )

    # missing precursor data
    missing_precursor_data = (
        data[["feature_id", "ms2_peakgroup_pep"]]
        .dropna(axis=0, how="any")
        .drop_duplicates()
    )
    missing_bm_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "feature_id": missing_precursor_data["feature_id"],
                    "prior": 1 - missing_precursor_data["ms2_peakgroup_pep"],
                    "evidence": 0,
                    "hypothesis": True,
                }
            ),
            pd.DataFrame(
                {
                    "feature_id": missing_precursor_data["feature_id"],
                    "prior": missing_precursor_data["ms2_peakgroup_pep"],
                    "evidence": 1,
                    "hypothesis": False,
                }
            ),
        ]
    )

    # combine precursor data
    precursor_bm_data = pd.concat([ms1_bm_data, ms2_bm_data])
    # append missing precursors if no MS1/MS2 evidence is available
    precursor_bm_data = pd.concat(
        [
            precursor_bm_data,
            missing_bm_data.loc[
                ~missing_bm_data["feature_id"].isin(precursor_bm_data["feature_id"])
            ],
        ]
    )

    return precursor_bm_data


def transfer_confident_evidence_across_runs(
    df1,
    across_run_confidence_threshold,
    group_cols=[
        "feature_id",
        "transition_id",
        "peptide_id",
        "bmask",
        "num_peptidoforms",
        "alignment_group_id",
    ],
    value_cols=["pep", "precursor_peakgroup_pep"],
):
    """
    Propagates confident evidence across aligned runs.

    Args:
        df1 (pd.DataFrame): Input data containing feature-level information.
        across_run_confidence_threshold (float): Confidence threshold for propagation.
        group_cols (list): Columns to group by during propagation.
        value_cols (list): Columns to apply the minimum reduction.

    Returns:
        pd.DataFrame: Data with propagated evidence across runs.
    """
    feature_ids = np.unique(df1["feature_id"])
    df_list = []
    for feature_id in feature_ids:
        tmp_df = df1[
            (df1["feature_id"] == feature_id)
            | (
                (df1["feature_id"] != feature_id)
                & (df1["pep"] <= across_run_confidence_threshold)
            )
        ]
        tmp_df["feature_id"] = feature_id
        df_list.append(tmp_df)
    df_filtered = pd.concat(df_list)

    # Group by relevant columns and apply min reduction
    df_result = df_filtered.groupby(group_cols, as_index=False)[value_cols].min()

    return df_result


def prepare_transition_bm(
    data, propagate_signal_across_runs, across_run_confidence_threshold
):
    """
    Prepares Bayesian model data for transition-level inference.

    Args:
        data (pd.DataFrame): Input data containing transition-level information.
        propagate_signal_across_runs (bool): Whether to propagate signal across runs.
        across_run_confidence_threshold (float): Confidence threshold for propagation.

    Returns:
        pd.DataFrame: Bayesian model data for transition-level inference.
    """
    # Propagate peps <= threshold for aligned feature groups across runs
    if propagate_signal_across_runs:
        ## Separate out features that need propagation and those that don't to avoid calling apply on the features that don't need propagated peps
        non_prop_data = data.loc[data["feature_id"] == data["alignment_group_id"]]
        prop_data = data.loc[data["feature_id"] != data["alignment_group_id"]]

        # Group by alignment_group_id and apply function in parallel
        data_with_confidence = (
            prop_data.groupby("alignment_group_id", group_keys=False)
            .apply(
                lambda df: transfer_confident_evidence_across_runs(
                    df, across_run_confidence_threshold
                )
            )
            .reset_index(drop=True)
        )

        logger.info(
            f"Propagating signal for {len(prop_data['feature_id'].unique())} aligned features of total {len(data['feature_id'].unique())} features across runs ..."
        )

        ## Concat non prop data with prop data
        data = pd.concat([non_prop_data, data_with_confidence], ignore_index=True)

    # peptide_id = -1 indicates h0, i.e. the peak group is wrong!
    # initialize priors
    data.loc[data.peptide_id != -1, "prior"] = (
        1 - data.loc[data.peptide_id != -1, "precursor_peakgroup_pep"]
    ) / data.loc[data.peptide_id != -1, "num_peptidoforms"]  # potential peptidoforms
    data.loc[data.peptide_id == -1, "prior"] = data.loc[
        data.peptide_id == -1, "precursor_peakgroup_pep"
    ]  # h0

    # set evidence
    data.loc[data.bmask == 1, "evidence"] = (
        1 - data.loc[data.bmask == 1, "pep"]
    )  # we have evidence FOR this peptidoform or h0
    data.loc[data.bmask == 0, "evidence"] = data.loc[
        data.bmask == 0, "pep"
    ]  # we have evidence AGAINST this peptidoform or h0

    if propagate_signal_across_runs:
        cols = [
            "feature_id",
            "alignment_group_id",
            "num_peptidoforms",
            "prior",
            "evidence",
            "peptide_id",
        ]
    else:
        cols = ["feature_id", "num_peptidoforms", "prior", "evidence", "peptide_id"]
    data = data[cols]
    data = data.rename(columns=lambda x: x.replace("peptide_id", "hypothesis"))

    return data


def apply_bm(data):
    """
    Applies the Bayesian model to compute posterior probabilities.

    Args:
        data (pd.DataFrame): Input Bayesian model data.

    Returns:
        pd.DataFrame: Data with posterior probabilities for each hypothesis.
    """
    # compute likelihood * prior per feature & hypothesis
    # all priors are identical but pandas DF multiplication requires aggregation, so we use min()
    pp_data = (
        data.groupby(["feature_id", "hypothesis"])["evidence"].prod()
        * data.groupby(["feature_id", "hypothesis"])["prior"].min()
    ).reset_index()
    pp_data.columns = ["feature_id", "hypothesis", "likelihood_prior"]

    # compute likelihood sum per feature
    pp_data["likelihood_sum"] = pp_data.groupby("feature_id")[
        "likelihood_prior"
    ].transform("sum")

    # compute posterior hypothesis probability
    pp_data["posterior"] = pp_data["likelihood_prior"] / pp_data["likelihood_sum"]

    return pp_data.fillna(value=0)


def precursor_inference(
    data,
    ipf_ms1_scoring,
    ipf_ms2_scoring,
    ipf_max_precursor_pep,
    ipf_max_precursor_peakgroup_pep,
):
    """
    Conducts precursor-level inference.

    Args:
        data (pd.DataFrame): Input data containing precursor-level information.
        ipf_ms1_scoring (bool): Whether to use MS1-level scoring.
        ipf_ms2_scoring (bool): Whether to use MS2-level scoring.
        ipf_max_precursor_pep (float): Maximum PEP threshold for precursors.
        ipf_max_precursor_peakgroup_pep (float): Maximum PEP threshold for peak groups.

    Returns:
        pd.DataFrame: Inferred precursor probabilities.
    """
    # prepare MS1-level precursor data
    if ipf_ms1_scoring:
        ms1_precursor_data = data[data["ms1_precursor_pep"] < ipf_max_precursor_pep][
            ["feature_id", "ms1_precursor_pep"]
        ].drop_duplicates()
    else:
        ms1_precursor_data = data[["feature_id"]].drop_duplicates()
        ms1_precursor_data["ms1_precursor_pep"] = np.nan

    # prepare MS2-level precursor data
    if ipf_ms2_scoring:
        ms2_precursor_data = data[data["ms2_precursor_pep"] < ipf_max_precursor_pep][
            ["feature_id", "ms2_precursor_pep"]
        ].drop_duplicates()
    else:
        ms2_precursor_data = data[["feature_id"]].drop_duplicates()
        ms2_precursor_data["ms2_precursor_pep"] = np.nan

    # prepare MS2-level peak group data
    ms2_pg_data = data[["feature_id", "ms2_peakgroup_pep"]].drop_duplicates()

    if ipf_ms1_scoring or ipf_ms2_scoring:
        # merge MS1- & MS2-level precursor and peak group data
        precursor_data = ms2_precursor_data.merge(
            ms1_precursor_data, on=["feature_id"], how="outer"
        ).merge(ms2_pg_data, on=["feature_id"], how="outer")

        # prepare precursor-level Bayesian model
        logger.info("Preparing precursor-level data ... ")
        precursor_data_bm = prepare_precursor_bm(precursor_data)

        # compute posterior precursor probability
        logger.info("Conducting precursor-level inference ... ")
        prec_pp_data = apply_bm(precursor_data_bm)
        prec_pp_data["precursor_peakgroup_pep"] = 1 - prec_pp_data["posterior"]

        inferred_precursors = prec_pp_data[prec_pp_data["hypothesis"]][
            ["feature_id", "precursor_peakgroup_pep"]
        ]

    else:
        # no precursor-level data on MS1 and/or MS2 should be used; use peak group-level data
        logger.info("Skipping precursor-level inference.")
        inferred_precursors = ms2_pg_data.rename(
            columns=lambda x: x.replace("ms2_peakgroup_pep", "precursor_peakgroup_pep")
        )

    inferred_precursors = inferred_precursors[
        (
            inferred_precursors["precursor_peakgroup_pep"]
            < ipf_max_precursor_peakgroup_pep
        )
    ]

    return inferred_precursors


def peptidoform_inference(
    transition_table,
    precursor_data,
    ipf_grouped_fdr,
    propagate_signal_across_runs,
    across_run_confidence_threshold,
):
    """
    Conducts peptidoform-level inference.

    Args:
        transition_table (pd.DataFrame): Input data containing transition-level information.
        precursor_data (pd.DataFrame): Precursor-level probabilities.
        ipf_grouped_fdr (bool): Whether to use grouped FDR estimation.
        propagate_signal_across_runs (bool): Whether to propagate signal across runs.
        across_run_confidence_threshold (float): Confidence threshold for propagation.

    Returns:
        pd.DataFrame: Inferred peptidoform probabilities and FDR estimates.
    """
    transition_table = pd.merge(transition_table, precursor_data, on="feature_id")

    # compute transition posterior probabilities
    logger.info("Preparing peptidoform-level data ... ")
    transition_data_bm = prepare_transition_bm(
        transition_table, propagate_signal_across_runs, across_run_confidence_threshold
    )

    # compute posterior peptidoform probability
    logger.info("Conducting peptidoform-level inference ... ")

    pf_pp_data = apply_bm(transition_data_bm)
    pf_pp_data["pep"] = 1 - pf_pp_data["posterior"]

    # compute model-based FDR
    if ipf_grouped_fdr:
        pf_pp_data["qvalue"] = (
            pd.merge(
                pf_pp_data,
                transition_data_bm[
                    ["feature_id", "num_peptidoforms"]
                ].drop_duplicates(),
                on=["feature_id"],
                how="inner",
            )
            .groupby("num_peptidoforms")["pep"]
            .transform(compute_model_fdr)
        )
    else:
        pf_pp_data["qvalue"] = compute_model_fdr(pf_pp_data["pep"])

    # merge precursor-level data with UIS data
    result = pf_pp_data.merge(
        precursor_data[["feature_id", "precursor_peakgroup_pep"]].drop_duplicates(),
        on=["feature_id"],
        how="inner",
    )

    return result


def plot_precursor_inference(pdf_handle, precursor_table, precursor_data):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Add supertitle for the entire figure
    fig.suptitle("Precursor-Level Inference Plots", fontsize=16)

    # Top-left: MS2 peak-group PEP histogram
    ax = axs[0, 0]
    ax.hist(precursor_table["ms2_peakgroup_pep"].dropna(), bins=50)
    ax.set_title("MS2 Peak-Group PEP")
    ax.set_xlabel("ms2_peakgroup_pep")
    ax.set_ylabel("Count")

    # Top-right: MS1 precursor PEP histogram or “No data”
    ax = axs[0, 1]
    vals = precursor_table["ms1_precursor_pep"].dropna()
    if len(vals):
        ax.hist(vals, bins=50)
        ax.set_title("MS1 Precursor PEP")
        ax.set_xlabel("ms1_precursor_pep")
        ax.set_ylabel("Count")
    else:
        ax.text(
            0.5, 0.5, "No ms1_precursor_pep data", ha="center", va="center", fontsize=12
        )
        ax.set_axis_off()

    # Bottom-left: MS2 precursor PEP histogram or “No data”
    ax = axs[1, 0]
    vals2 = precursor_table["ms2_precursor_pep"].dropna()
    if len(vals2):
        ax.hist(vals2, bins=50)
        ax.set_title("MS2 Precursor PEP")
        ax.set_xlabel("ms2_precursor_pep")
        ax.set_ylabel("Count")
    else:
        ax.text(
            0.5, 0.5, "No ms2_precursor_pep data", ha="center", va="center", fontsize=12
        )
        ax.set_axis_off()

    # merge precursor data with precursor_table
    viz_df = precursor_table.merge(
        precursor_data[["feature_id", "precursor_peakgroup_pep"]],
        on=["feature_id"],
        how="inner",
    )

    # Check if ms1_precursor_pep and ms2_peakgroup_pep are not all NaN
    if (
        viz_df["ms1_precursor_pep"].isna().all()
        or viz_df["ms2_peakgroup_pep"].isna().all()
    ):
        ax = axs[1, 1]
        ax.text(
            0.5,
            0.5,
            "No data for ms1_precursor_pep or ms2_peakgroup_pep",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
    else:
        # Bottom-right: 2D scatter colored by posterior
        ax = axs[1, 1]
        sc = ax.scatter(
            viz_df.ms2_peakgroup_pep,
            viz_df.ms1_precursor_pep,
            s=20,
            c=1 - viz_df.precursor_peakgroup_pep,
            cmap="viridis",
            alpha=0.7,
        )
        ax.set_title("Posterior vs. MS2 & MS1 PEP")
        ax.set_xlabel("ms2_peakgroup_pep")
        ax.set_ylabel("ms1_precursor_pep")
        fig.colorbar(sc, ax=ax, label="posterior")

    fig.tight_layout()
    pdf_handle.savefig(fig)
    plt.close(fig)


def plot_peptidoform_inference(
    pdf_handle, peptidoform_table, peptidoform_data, precursor_data=None
):
    """
    Generate an 4x2 subplot figure for peptidoform-level inference and save to pdf_handle.

    Parameters:
    - pdf_handle: PdfPages object open for saving figures.
    - peptidoform_table: DataFrame before inference (cols: feature_id, transition_id, pep, peptide_id, bmask, num_peptidoforms).
    - peptidoform_data: DataFrame after inference (cols: feature_id, hypothesis, likelihood_prior, likelihood_sum, posterior, pep, qvalue, precursor_peakgroup_pep).
    - precursor_data: (optional) DataFrame with cols feature_id, precursor_peakgroup_pep.
    """
    # Create a 4x2 grid: total 8 plots
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle("Peptidoform-Level Inference Summary", fontsize=18)

    # 1) Raw transition-level PEP histogram
    logger.debug("Plotting raw transition-level PEP histogram ...")
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.hist(peptidoform_table["pep"].dropna(), bins=50)
    ax1.set_title("Raw transition-level PEPs")
    ax1.set_xlabel("pep")
    ax1.set_ylabel("Count")

    # 2) Posterior probability histogram
    logger.debug("Plotting posterior probability distribution ...")
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.hist(peptidoform_data["posterior"].dropna(), bins=50)
    ax2.set_title("Posterior probability distribution")
    ax2.set_xlabel("posterior")
    ax2.set_ylabel("Count")

    # 3) 2D histogram of posterior vs. group size
    logger.debug("Plotting 2D histogram of posterior vs. group size …")
    # Merge to get num_peptidoforms
    df_merge = peptidoform_data.merge(
        peptidoform_table[["feature_id", "num_peptidoforms"]].drop_duplicates(),
        on="feature_id",
        how="inner",
    )
    ax3 = fig.add_subplot(4, 2, 3)

    # Use 40 bins in X, 3 bins in Y for the discrete posterior levels, linear color scale
    h = ax3.hist2d(
        df_merge["num_peptidoforms"],
        df_merge["posterior"],
        bins=[40, 3],
        cmap="plasma",  # high‐contrast perceptual map
    )

    # no log‐scaling on the X axis
    ax3.set_title("2D histogram: posterior vs. group size")
    ax3.set_xlabel("num_peptidoforms")
    ax3.set_ylabel("posterior")

    # colorbar for the histogram (linear counts)
    cbar = fig.colorbar(h[3], ax=ax3, label="counts")

    # 4) 3D surface: prior vs evidence vs posterior
    logger.debug("Plotting 3D surface of prior vs evidence vs posterior ...")
    # Recompute prior & evidence
    df_pep = peptidoform_table.copy()
    df_pep["prior"] = np.where(
        df_pep["peptide_id"] != -1, 1.0 / df_pep["num_peptidoforms"], df_pep["pep"]
    )
    df_pep["evidence"] = np.where(
        df_pep["bmask"] == 1, 1 - df_pep["pep"], df_pep["pep"]
    )
    merged_surface = pd.merge(
        df_pep,
        peptidoform_data[["feature_id", "hypothesis", "posterior"]],
        left_on=["feature_id", "peptide_id"],
        right_on=["feature_id", "hypothesis"],
    )
    merged_surface = merged_surface.dropna(subset=["prior", "evidence", "posterior"])
    priors = merged_surface["prior"]
    evidences = merged_surface["evidence"]
    posteriors = merged_surface["posterior"]
    xi = np.linspace(priors.min(), priors.max(), 30)
    yi = np.linspace(evidences.min(), evidences.max(), 30)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((priors, evidences), posteriors, (xi, yi), method="linear")
    ax4 = fig.add_subplot(4, 2, 4, projection="3d")
    ax4.plot_surface(xi, yi, zi, rcount=20, ccount=20, alpha=0.8)
    ax4.set_title("Posterior surface")
    ax4.set_xlabel("prior")
    ax4.set_ylabel("evidence")
    ax4.set_zlabel("posterior")

    # 5) Scatter: transition PEP vs precursor confidence
    logger.debug("Plotting 2D contour of posterior surface ...")
    ax5 = fig.add_subplot(4, 2, 5)
    cf = ax5.contourf(
        xi, yi, zi, levels=50, cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)
    )
    ax5.set_title("Posterior surface (2D contour)")
    ax5.set_xlabel("prior")
    ax5.set_ylabel("evidence")
    cbar = fig.colorbar(cf, ax=ax5, label="posterior")

    # 6) 2D histogram: pep vs precursor confidence
    logger.debug("Plotting 2D histogram of transition pep vs precursor confidence ...")
    ax6 = fig.add_subplot(4, 2, 6)
    if precursor_data is not None:
        trans = peptidoform_table[["feature_id", "pep"]].drop_duplicates()
        merged2 = pd.merge(
            trans,
            precursor_data[["feature_id", "precursor_peakgroup_pep"]],
            on="feature_id",
            how="inner",
        )
        merged2 = pd.merge(
            merged2,
            peptidoform_data[["feature_id", "posterior"]],
            on="feature_id",
            how="inner",
        )

        ax6.hist2d(merged2["pep"], merged2["precursor_peakgroup_pep"], bins=50)
        ax6.set_title("2D hist: pep vs. precursor confidence")
        ax6.set_xlabel("transition pep")
        ax6.set_ylabel("precursor_peakgroup_pep")
    else:
        ax6.text(0.5, 0.5, "No precursor_data for hist2d", ha="center", va="center")
        ax6.set_axis_off()

    # 7) Cumulative q-value curves
    logger.debug("Plotting cumulative q-value curves (-log10) by group size …")
    ax7 = fig.add_subplot(4, 2, 7)

    # prepare colormap
    group_sizes = sorted(df_merge["num_peptidoforms"].unique())
    norm = mpl.colors.Normalize(vmin=min(group_sizes), vmax=max(group_sizes))
    cmap = mpl.cm.get_cmap("viridis")

    # threshold in –log10 units
    thresh_q = 0.01
    thresh_y = -np.log10(thresh_q)

    # compute a floor for zero q-values
    all_q = df_merge["qvalue"]
    # use the smallest non-zero qvalue, or 1e-16 if none
    min_nonzero = all_q[all_q > 0].min() if (all_q > 0).any() else 1e-16
    eps = min_nonzero / 10.0

    for k, grp in df_merge.groupby("num_peptidoforms"):
        q_sorted = np.sort(grp["qvalue"])
        # clip zeros up to eps to avoid -log10(0)
        q_clipped = np.clip(q_sorted, a_min=eps, a_max=None)
        y_vals = -np.log10(q_clipped)
        x_vals = np.arange(1, len(y_vals) + 1)
        ax7.plot(
            x_vals,
            y_vals,
            color=cmap(norm(k)),
            linewidth=1,
            alpha=0.8,
        )

    # dashed line at 1% Q (–log10 = 2)
    ax7.axhline(thresh_y, color="red", linestyle="--", linewidth=1)
    ax7.text(
        0.98,
        thresh_y + 0.05,
        "1% Q (\u2013log10=2)",
        color="red",
        ha="right",
        va="bottom",
        transform=ax7.get_yaxis_transform(),
    )

    # annotate total count below threshold (across *all* groups)
    total_below = (df_merge["qvalue"] <= thresh_q).sum()
    ax7.text(
        0.98,
        0.95,
        f"Total ≤1% Q: {total_below}",
        transform=ax7.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
    )

    ax7.set_title("Cumulative –log10(qvalue) by group size")
    ax7.set_xlabel("Rank")
    ax7.set_ylabel("-log10(qvalue)")

    # colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax7, pad=0.01)
    cbar.set_label("num_peptidoforms")

    # 8) Heatmap: feature × hypothesis posterior, with percentile‐based scaling
    logger.debug("Plotting feature × hypothesis posterior heatmap …")
    ax8 = fig.add_subplot(4, 2, 8)

    # pivot into matrix
    pivot = (
        df_merge[df_merge["hypothesis"] != -1]
        .pivot(index="feature_id", columns="hypothesis", values="posterior")
        .fillna(0)
    )

    # find a high percentile of the nonzero values to saturate at (e.g. 95th percentile)
    nonzero = pivot.values[pivot.values > 0]
    if len(nonzero):
        vmax = np.percentile(nonzero, 95)
    else:
        vmax = 1.0

    # plot with a high‐contrast colormap and clipped vmax
    cax = ax8.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )

    ax8.set_title("Feature × Peptidoform posterior heatmap")
    ax8.set_xlabel("hypothesis")
    ax8.set_ylabel("feature index")

    # add colorbar, note it shows 0 → vmax
    cbar = fig.colorbar(cax, ax=ax8, label="posterior (0–95th pct)")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_handle.savefig(fig)
    plt.close(fig)


def infer_peptidoforms(config: IPFIOConfig):
    """
    Orchestrates the Inference of PeptidoForms (IPF) workflow.

    Args:
        config (IPFIOConfig): Configuration object for the IPF workflow.

    Returns:
        None
    """
    logger.info("Starting IPF (Inference of PeptidoForms).")

    batch_size = config.batch_size

    reader = ReaderDispatcher.get_reader(config)
    pdf_handle = PdfPages(f"{config.prefix}_ipf_report.pdf")

    # precursor level
    precursor_table = reader.read(level="peakgroup_precursor", peptide_ids=None)
    precursor_data = precursor_inference(
        precursor_table,
        config.ipf_ms1_scoring,
        config.ipf_ms2_scoring,
        config.ipf_max_precursor_pep,
        config.ipf_max_precursor_peakgroup_pep,
    )
    plot_precursor_inference(pdf_handle, precursor_table, precursor_data)

    # get peptide ids
    peptidoform_group_mapping = reader.read(level="peptide_ids")

    logger.info(
        f"Processing {len(peptidoform_group_mapping)} peptidoform ids, and {len(peptidoform_group_mapping['isoform_id'].unique())} peptidoform groups."
    )

    peptidoform_group_ids = peptidoform_group_mapping["isoform_id"].unique()

    # Iterate over peptide_ids in batches to avoid memory issues
    if batch_size > 0:
        peptidoform_group_id_batches = [
            peptidoform_group_ids[i : i + batch_size]
            for i in range(0, len(peptidoform_group_ids), batch_size)
        ]
    else:
        peptidoform_group_id_batches = [peptidoform_group_ids]

    # Initialize an empty list to collect results
    all_peptidoform_data = []
    for group_idx, peptidoform_group_id_batches_batch in enumerate(
        peptidoform_group_id_batches
    ):
        peptidoform_group_batch_mapping = peptidoform_group_mapping[
            peptidoform_group_mapping["isoform_id"].isin(
                peptidoform_group_id_batches_batch
            )
        ]

        logger.info(
            f"Processing batch {group_idx + 1} of {len(peptidoform_group_id_batches)}: {len(peptidoform_group_batch_mapping)} peptidoforms with {len(peptidoform_group_batch_mapping['isoform_id'].unique())} unique peptidoform groups."
        )

        peptide_ids_batch = peptidoform_group_batch_mapping["peptide_id"].unique()

        # peptidoform level
        peptidoform_table = reader.read(
            level="transition", peptide_ids=peptide_ids_batch
        )

        ## prepare for propagating signal across runs for aligned features
        if config.propagate_signal_across_runs:
            across_run_feature_map = reader.read(level="alignment")

            peptidoform_table = peptidoform_table.merge(
                across_run_feature_map, how="left", on="feature_id"
            )
            ## Fill missing alignment_group_id with feature_id for those that are not aligned
            peptidoform_table["alignment_group_id"] = peptidoform_table[
                "alignment_group_id"
            ].astype(object)
            mask = peptidoform_table["alignment_group_id"].isna()
            peptidoform_table.loc[mask, "alignment_group_id"] = peptidoform_table.loc[
                mask, "feature_id"
            ].astype(str)

            peptidoform_table = peptidoform_table.astype(
                {"alignment_group_id": "int64"}
            )

        peptidoform_data = peptidoform_inference(
            peptidoform_table,
            precursor_data,
            config.ipf_grouped_fdr,
            config.propagate_signal_across_runs,
            config.across_run_confidence_threshold,
        )

        plot_peptidoform_inference(
            pdf_handle,
            peptidoform_table,
            peptidoform_data,
            precursor_data=precursor_data,
        )

        # finalize results and write to table
        peptidoform_data = peptidoform_data[peptidoform_data["hypothesis"] != -1][
            ["feature_id", "hypothesis", "precursor_peakgroup_pep", "qvalue", "pep"]
        ]
        peptidoform_data.columns = [
            "FEATURE_ID",
            "PEPTIDE_ID",
            "PRECURSOR_PEAKGROUP_PEP",
            "QVALUE",
            "PEP",
        ]

        # Convert feature_id to int64
        peptidoform_data = peptidoform_data.astype({"FEATURE_ID": "int64"})

        all_peptidoform_data.append(peptidoform_data)

    # Concatenate all batches of peptidoform data
    peptidoform_data = pd.concat(all_peptidoform_data, ignore_index=True)

    # Save results
    logger.info("Storing results.")
    num_unique_peptides = peptidoform_data[peptidoform_data["QVALUE"] <= 0.01][
        ["FEATURE_ID", "PEPTIDE_ID"]
    ].drop_duplicates()
    logger.info(
        f"Number of unique feature-peptidoform pairs at <= 1% QVALUE: {num_unique_peptides['FEATURE_ID'].nunique()} features, {num_unique_peptides['PEPTIDE_ID'].nunique()} peptides."
    )
    pdf_handle.close()
    writer = WriterDispatcher.get_writer(config)
    writer.save_results(result=peptidoform_data)


def pre_propagate_evidence(config: IPFIOConfig):
    """
    Pre-propagates evidence across aligned runs. Creates a

    Args:
        config (IPFIOConfig): Configuration object for the IPF workflow.

    Returns:
        None
    """
    logger.info("Pre-propagating evidence across aligned runs.")

    re_create_tables = config.re_create_tables
    chunk_size = config.batch_size

    # --- connect to on-disk DuckDB -------------------------------
    con = duckdb.connect("filtered.duckdb")
    con.execute(f"PRAGMA threads={os.cpu_count() - 1}")

    # --- list your parquet files & threshold ---------------------
    pep_threshold = config.ipf_max_transition_pep
    across_run_confidence_threshold = config.across_run_confidence_threshold

    # 1) read & register your alignment map
    if re_create_tables:
        reader = ReaderDispatcher.get_reader(config)
        across_run_feature_map = reader.read(level="alignment")
        con.register("across_run_feature_map_tbl", across_run_feature_map)
        con.execute("""
        CREATE TABLE IF NOT EXISTS across_run_feature_map AS
        SELECT * FROM across_run_feature_map_tbl
        """)
        del across_run_feature_map
        con.unregister("across_run_feature_map_tbl")

    # 2) build feature → run lookup
    if re_create_tables:
        precursor_files = glob.glob("oswpq/*.oswpq/precursors_features.parquet")
        transition_files = glob.glob("oswpq/*.oswpq/transition_features.parquet")
        con.execute(f"""
        CREATE TABLE IF NOT EXISTS feature_run_map AS
        SELECT DISTINCT
            FEATURE_ID AS feature_id,
            RUN_ID     AS run_id
        FROM read_parquet({precursor_files}, hive_partitioning=1)
        ;
        """)

    # 3) filter & persist transition‐level PEPs
    if re_create_tables:
        con.execute(f"""
        CREATE TABLE IF NOT EXISTS filtered_transitions AS
        SELECT
            FEATURE_ID          AS feature_id,
            TRANSITION_ID       AS transition_id,
            IPF_PEPTIDE_ID      AS peptide_id,
            SCORE_TRANSITION_PEP AS pep
        FROM read_parquet({transition_files}, hive_partitioning=1)
        WHERE
            TRANSITION_TYPE         <> ''
        AND TRANSITION_DECOY       =  0
        AND SCORE_TRANSITION_SCORE IS NOT NULL
        AND SCORE_TRANSITION_PEP   <  {pep_threshold}
        ;
        """)

    # 4) join in run_id & alignment_group_id
    if re_create_tables:
        con.execute("""
        CREATE TABLE IF NOT EXISTS merged_transitions AS
        SELECT
            f.run_id,
            t.*,
            COALESCE(m.alignment_group_id, t.feature_id) AS alignment_group_id
        FROM filtered_transitions AS t
        LEFT JOIN feature_run_map         AS f USING(feature_id)
        LEFT JOIN across_run_feature_map AS m USING(feature_id)
        ;
        """)

    # ── helper to chunk an iterable ────────────────────────────────
    def chunked(iterable, size):
        it = iter(iterable)
        while True:
            batch = list(islice(it, size))
            if not batch:
                return
            yield batch

    # 5.1) make sure we have a table to remember which groups ran
    con.execute("""
    CREATE TABLE IF NOT EXISTS processed_groups (
    id BIGINT PRIMARY KEY
    )
    """)

    # 5) chunk *by* alignment_group_id so you never split a group
    # -------------------------------------------------------------

    while True:
        # 5a) grab all IDs needing propagation
        # fetch only *unprocessed* alignment_group_ids
        group_ids = [
            r[0]
            for r in con.execute("""
            SELECT DISTINCT alignment_group_id
            FROM merged_transitions
            WHERE feature_id != alignment_group_id
            AND alignment_group_id NOT IN (SELECT id FROM processed_groups)
        """).fetchall()
        ]
        if not group_ids:
            print("✅ all done!")
            break
        print(
            f"Found {len(group_ids)} alignment groups needing propagation.", flush=True
        )
        # 5b) process in batches of, say, 500 groups at a time
        for i, batch in enumerate(chunked(group_ids, chunk_size), start=1):
            print(
                f"[{i} of {len(group_ids) // chunk_size + 1}] propagating {len(batch)} groups…",
                flush=True,
            )
            ids = ",".join(map(str, batch))
            # pull *all* rows for these groups in one go
            df_chunk = con.execute(f"""
                SELECT *
                FROM merged_transitions
                WHERE alignment_group_id IN ({ids})
                AND feature_id != alignment_group_id
            """).df()
            # apply your Python propagation
            df_prop = transfer_confident_evidence_across_runs(
                df_chunk,
                across_run_confidence_threshold,
                group_cols=[
                    "run_id",
                    "feature_id",
                    "transition_id",
                    "peptide_id",
                    "alignment_group_id",
                ],
                value_cols=["pep"],
            )
            # write them back, then free memory immediately
            con.register("tmp", df_prop)
            # start a manual transaction
            con.execute("BEGIN TRANSACTION;")
            try:
                # your delete + insert here
                con.execute("""
                DELETE FROM merged_transitions
                USING tmp
                WHERE merged_transitions.run_id             = tmp.run_id
                    AND merged_transitions.feature_id         = tmp.feature_id
                    AND merged_transitions.transition_id      = tmp.transition_id
                    AND merged_transitions.alignment_group_id = tmp.alignment_group_id
                """)
                con.execute("""
                INSERT INTO merged_transitions
                SELECT * FROM tmp
                """)
                # now mark these groups as done
                # we can bulk-insert them
                values = ",".join(f"({g})" for g in batch)
                con.execute(f"""
                INSERT INTO processed_groups(id)
                VALUES {values}
                ON CONFLICT DO NOTHING   -- ignore duplicates
                """)
            except Exception:
                # undo everything in this transaction
                con.execute("ROLLBACK;")
                raise
            else:
                # make it permanent
                con.execute("COMMIT;")
            con.unregister("tmp")
            del df_chunk, df_prop
