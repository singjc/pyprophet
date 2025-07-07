import click
from loguru import logger

from .util import transform_betas, write_logfile, measure_memory_usage_and_time
from .._config import IPFIOConfig
from ..ipf import infer_peptidoforms, pre_propagate_evidence

# from ..ipf_bayenet import infer_peptidoforms as infer_peptidoforms_bayenet
from ..glyco.glycoform import infer_glycoforms

infer_peptidoforms_bayenet = None


# IPF
@click.command()
# File handling
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file. Valid formats are .osw, .parquet (produced by export parquet)",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="PyProphet output file. Valid formats are .osw, .parquet. Must be the same format as input file.",
)
@click.option(
    "--pth",
    "pretrained_model",
    type=click.Path(exists=True),
    help="Pretrained model file. Must be a .pth file.",
)
# IPF parameters
@click.option(
    "--ipf_ms1_scoring/--no-ipf_ms1_scoring",
    default=True,
    show_default=True,
    help="Use MS1 precursor data for IPF.",
)
@click.option(
    "--ipf_ms2_scoring/--no-ipf_ms2_scoring",
    default=True,
    show_default=True,
    help="Use MS2 precursor data for IPF.",
)
@click.option(
    "--ipf_h0/--no-ipf_h0",
    default=True,
    show_default=True,
    help="Include possibility that peak groups are not covered by peptidoform space.",
)
@click.option(
    "--ipf_grouped_fdr/--no-ipf_grouped_fdr",
    default=False,
    show_default=True,
    help="[Experimental] Compute grouped FDR instead of pooled FDR to better support data where peak groups are evaluated to originate from very heterogeneous numbers of peptidoforms.",
)
@click.option(
    "--ipf_max_precursor_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored precursors in IPF.",
)
@click.option(
    "--ipf_max_peakgroup_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored peak groups in IPF.",
)
@click.option(
    "--ipf_max_precursor_peakgroup_pep",
    default=0.4,
    show_default=True,
    type=float,
    help="Maximum BHM layer 1 integrated precursor peakgroup PEP to consider in IPF.",
)
@click.option(
    "--ipf_max_transition_pep",
    default=0.6,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored transitions in IPF.",
)
@click.option(
    "--propagate_signal_across_runs/--no-propagate_signal_across_runs",
    default=False,
    show_default=True,
    help="Propagate signal across runs (requires running alignment).",
)
@click.option(
    "--ipf_max_alignment_pep",
    default=1.0,
    show_default=True,
    type=float,
    help="Maximum PEP to consider for good alignments.",
)
@click.option(
    "--across_run_confidence_threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="Maximum PEP to consider for propagating signal across runs for aligned features.",
)
@click.option(
    "--batch_size",
    default=100_000,
    show_default=True,
    type=int,
    help="Batch size for processing the input file. Adjust based on available memory.",
)
@click.option(
    "--create_propagation_db/--no-create_propagation_db",
    default=False,
    show_default=True,
    help="[Experimental] Create propagation database.",
)
@click.option(
    "--re_create_tables/--no-re_create_tables",
    default=False,
    show_default=True,
    help="[Experimental] Re-create tables for duckdb pre-propagated evidence.",
)
# Bayesian Network Model Parameters
@click.option(
    "--use_bayenet/--no-use_bayenet",
    default=False,
    show_default=True,
    help="Use generative Bayesian network model for IPF. If disabled, uses a flat Bayesian hierarchical model.",
)
@click.option(
    "--use_beta/--no-use_beta",
    default=True,
    show_default=True,
    help="Use beta log-likelihoods for the Bayesian network model. If disabled, uses Gaussian log-likelihoods.",
)
@click.option(
    "--num_steps",
    default=350,
    show_default=True,
    type=int,
    help="Number of steps for the Bayesian network model optimization for stochastic variational inference. Adjust based on convergence needs.",
)
@click.option(
    "--eta",
    "learning_rate",
    default=1e-3,
    show_default=True,
    type=float,
    help="Maximum PEP to consider for good alignments.",
)
@click.option(
    "--betas",
    default=(0.9, 0.999),
    show_default=True,
    type=(float, float),
    help="Beta parameters for the optimizer used in the Bayesian network model.",
    callback=transform_betas,
)
@click.pass_context
@measure_memory_usage_and_time
@logger.catch(reraise=True)
def ipf(
    ctx,
    infile,
    outfile,
    pretrained_model,
    ipf_ms1_scoring,
    ipf_ms2_scoring,
    ipf_h0,
    ipf_grouped_fdr,
    ipf_max_precursor_pep,
    ipf_max_peakgroup_pep,
    ipf_max_precursor_peakgroup_pep,
    ipf_max_transition_pep,
    propagate_signal_across_runs,
    ipf_max_alignment_pep,
    across_run_confidence_threshold,
    batch_size,
    create_propagation_db,
    re_create_tables,
    use_bayenet,
    use_beta,
    num_steps,
    learning_rate,
    betas,
):
    """
    Infer peptidoforms after scoring of MS1, MS2 and transition-level data.
    """

    # Check to see if we're calling from the previous deprecated command call
    if ctx.command.deprecated:
        logger.warning(
            "The 'pyprophet ipf' command is deprecated and will be removed in future versions. Use 'pyprophet infer peptidoform' instead."
        )

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    config = IPFIOConfig.from_cli_args(
        infile,
        outfile,
        pretrained_model,
        1,  # Subsample ratio is not applicable for IPF
        "ipf",  # Level is not applicable for IPF
        "ipf",
        ipf_ms1_scoring,
        ipf_ms2_scoring,
        ipf_h0,
        ipf_grouped_fdr,
        ipf_max_precursor_pep,
        ipf_max_peakgroup_pep,
        ipf_max_precursor_peakgroup_pep,
        ipf_max_transition_pep,
        propagate_signal_across_runs,
        ipf_max_alignment_pep,
        across_run_confidence_threshold,
        batch_size=batch_size,
        re_create_tables=re_create_tables,
        use_bayenet=use_bayenet,
        use_beta=use_beta,
        num_steps=num_steps,
        learning_rate=learning_rate,
        betas=betas,
    )
    write_logfile(
        ctx.obj["LOG_LEVEL"], f"{config.prefix}_pyp_ipf.log", ctx.obj["LOG_HEADER"]
    )
    if use_bayenet:  # Generative BHM
        infer_peptidoforms_bayenet(config)
    elif create_propagation_db:
        pre_propagate_evidence(config)
    else:  # Base flat BHM
        infer_peptidoforms(config)


# Infer glycoforms
@click.command()
@click.option(
    "--in", "infile", required=True, type=click.Path(exists=True), help="Input file."
)
@click.option("--out", "outfile", type=click.Path(exists=False), help="Output file.")
@click.option(
    "--ms1_precursor_scoring/--no-ms1_precursor_scoring",
    default=True,
    show_default=True,
    help="Use MS1 precursor data for glycoform inference.",
)
@click.option(
    "--ms2_precursor_scoring/--no-ms2_precursor_scoring",
    default=True,
    show_default=True,
    help="Use MS2 precursor data for glycoform inference.",
)
@click.option(
    "--grouped_fdr/--no-grouped_fdr",
    default=False,
    show_default=True,
    help="[Experimental] Compute grouped FDR instead of pooled FDR to better support data where peak groups are evaluated to originate from very heterogeneous numbers of glycoforms.",
)
@click.option(
    "--max_precursor_pep",
    default=1,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored precursors.",
)
@click.option(
    "--max_peakgroup_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored peak groups.",
)
@click.option(
    "--max_precursor_peakgroup_pep",
    default=1,
    show_default=True,
    type=float,
    help="Maximum BHM layer 1 integrated precursor peakgroup PEP to consider.",
)
@click.option(
    "--max_transition_pep",
    default=0.6,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored transitions.",
)
@click.option(
    "--use_glycan_composition/--use_glycan_struct",
    "use_glycan_composition",
    default=True,
    show_default=True,
    help="Compute glycoform-level FDR based on glycan composition or struct.",
)
@click.option(
    "--ms1_mz_window",
    default=10,
    show_default=True,
    type=float,
    help="MS1 m/z window in Thomson or ppm.",
)
@click.option(
    "--ms1_mz_window_unit",
    default="ppm",
    show_default=True,
    type=click.Choice(["ppm", "Da", "Th"]),
    help="MS1 m/z window unit.",
)
@click.option(
    "--propagate_signal_across_runs/--no-propagate_signal_across_runs",
    default=False,
    show_default=True,
    help="Propagate signal across runs (requires running alignment).",
)
@click.option(
    "--max_alignment_pep",
    default=1.0,
    show_default=True,
    type=float,
    help="Maximum PEP to consider for good alignments.",
)
@click.option(
    "--across_run_confidence_threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="Maximum PEP to consider for propagating signal across runs for aligned features.",
)
def glycoform(
    infile,
    outfile,
    ms1_precursor_scoring,
    ms2_precursor_scoring,
    grouped_fdr,
    max_precursor_pep,
    max_peakgroup_pep,
    max_precursor_peakgroup_pep,
    max_transition_pep,
    use_glycan_composition,
    ms1_mz_window,
    ms1_mz_window_unit,
    propagate_signal_across_runs,
    max_alignment_pep,
    across_run_confidence_threshold,
):
    """
    Infer glycoforms after scoring of MS1, MS2 and transition-level data.
    """

    if outfile is None:
        outfile = infile

    infer_glycoforms(
        infile=infile,
        outfile=outfile,
        ms1_precursor_scoring=ms1_precursor_scoring,
        ms2_precursor_scoring=ms2_precursor_scoring,
        grouped_fdr=grouped_fdr,
        max_precursor_pep=max_precursor_pep,
        max_peakgroup_pep=max_peakgroup_pep,
        max_precursor_peakgroup_pep=max_precursor_peakgroup_pep,
        max_transition_pep=max_transition_pep,
        use_glycan_composition=use_glycan_composition,
        ms1_mz_window=ms1_mz_window,
        ms1_mz_window_unit=ms1_mz_window_unit,
        propagate_signal_across_runs=propagate_signal_across_runs,
        max_alignment_pep=max_alignment_pep,
        across_run_confidence_threshold=across_run_confidence_threshold,
    )
