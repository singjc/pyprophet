import sys
import os
import click
import pandas as pd
import numpy as np
import polars as pl
import sqlite3

from .stats import error_statistics, lookup_values_from_error_table, final_err_table, summary_err_table
from .report import save_report
from shutil import copyfile
from .data_handling import is_sqlite_file, check_sqlite_table, is_parquet_file, get_parquet_column_names
from .glyco.stats import statistics_report as glyco_statistics_report


def statistics_report(data, outfile, context, analyte, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, color_palette):

    error_stat, pi0 = error_statistics(data[data.decoy==0]['score'], data[data.decoy==1]['score'], parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, True, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps)

    stat_table = final_err_table(error_stat)
    summary_table = summary_err_table(error_stat)

    # print summary table
    click.echo("=" * 80)
    click.echo(summary_table)
    click.echo("=" * 80)

    p_values, s_values, peps, q_values = lookup_values_from_error_table(data["score"].values, error_stat)
    data["p_value"] = p_values
    data["s_value"] = s_values
    data["q_value"] = q_values
    data["pep"] = peps

    if context == 'run-specific':
        outfile = outfile + "_" + str(data['run_id'].unique()[0])

    # export PDF report
    save_report(outfile + "_" + context + "_" + analyte + ".pdf", 
                outfile + ": " + context + " " + analyte + "-level error-rate control", 
                data[data.decoy==1]["score"].values, data[data.decoy==0]["score"].values, stat_table["cutoff"].values, 
                stat_table["svalue"].values, stat_table["qvalue"].values, data[data.decoy==0]["p_value"].values, 
                pi0, 
                color_palette)

    return(data)

def infer_genes(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, color_palette):
    
    if is_parquet_file(infile):
        all_column_names = get_parquet_column_names(infile)
        if not any([col.startswith("SCORE_MS2_") for col in all_column_names]):
            raise click.ClickException("Apply scoring to MS2-level data before running gene-level scoring.")
    else:
        con = sqlite3.connect(infile)
        if not check_sqlite_table(con, "SCORE_MS2"):
            raise click.ClickException("Apply scoring to MS2-level data before running gene-level scoring.")

    if context in ['global','experiment-wide','run-specific']:
        if context == 'global':
            run_id = 'NULL'
            group_id = 'GENE.ID'
        else:
            run_id = 'RUN_ID'
            group_id = 'RUN_ID || "_" || GENE.ID'

        if is_parquet_file(infile):
            # Read necessary columns from parquet
            cols = ['RUN_ID', 'GENE_ID', 'PRECURSOR_DECOY', 'SCORE_MS2_SCORE', 'PEPTIDE_ID']
            data = pl.read_parquet(infile, columns=cols)
            
            data = (
                data.with_columns(
                    # Common transformations
                    pl.col('GENE_ID').cast(pl.Utf8),
                    pl.col('PEPTIDE_ID').cast(pl.Utf8),
                    pl.col('PRECURSOR_DECOY').alias('DECOY')
                )
                .with_columns(
                    # Conditional transformations
                    pl.when(pl.lit(context == 'global'))
                    .then(pl.struct([
                        pl.lit(None).cast(pl.Int64).alias('RUN_ID_NEW'),
                        pl.col('GENE_ID').alias('GROUP_ID')
                    ]))
                    .otherwise(pl.struct([
                        pl.col('RUN_ID').cast(pl.Int64).alias('RUN_ID_NEW'),
                        (pl.col('RUN_ID').cast(pl.Utf8) + "_" + pl.col('GENE_ID')).alias('GROUP_ID')
                    ]))
                    .alias('fields')
                )
                .unnest('fields')
                .drop('RUN_ID')  # Drop original RUN_ID
                .rename({'RUN_ID_NEW': 'RUN_ID'})  # Rename to original name
                .group_by('GROUP_ID')
                .agg(
                    pl.col('SCORE_MS2_SCORE').max().alias('SCORE'),
                    pl.first('RUN_ID'),
                    pl.first('GENE_ID'),
                    pl.first('DECOY')
                )
                .with_columns(
                    pl.lit(context).cast(pl.Utf8).alias('CONTEXT')
                )
                .select([
                    'RUN_ID',
                    'GROUP_ID',
                    'GENE_ID',
                    'DECOY',
                    'SCORE',
                    'CONTEXT'
                ])
                .sort('SCORE', descending=True)
                .to_pandas()
            )
        else:
            con.executescript('''
                CREATE INDEX IF NOT EXISTS idx_peptide_gene_mapping_gene_id ON PEPTIDE_GENE_MAPPING (GENE_ID);
                CREATE INDEX IF NOT EXISTS idx_peptide_gene_mapping_peptide_id ON PEPTIDE_GENE_MAPPING (PEPTIDE_ID);
                CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID);
                CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);
                CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);
                CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
                CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
                CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
                CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);
            ''')

            data = pd.read_sql_query('''
                SELECT %s AS RUN_ID,
                       %s AS GROUP_ID,
                       GENE.ID AS GENE_ID,
                       PRECURSOR.DECOY AS DECOY,
                       SCORE,
                       "%s" AS CONTEXT
                FROM GENE
                INNER JOIN
                  (SELECT PEPTIDE_GENE_MAPPING.PEPTIDE_ID AS PEPTIDE_ID,
                          GENE_ID
                   FROM
                     (SELECT PEPTIDE_ID,
                             COUNT(*) AS NUM_GENES
                      FROM PEPTIDE_GENE_MAPPING
                      GROUP BY PEPTIDE_ID) AS GENES_PER_PEPTIDE
                   INNER JOIN PEPTIDE_GENE_MAPPING ON GENES_PER_PEPTIDE.PEPTIDE_ID = PEPTIDE_GENE_MAPPING.PEPTIDE_ID
                   WHERE NUM_GENES == 1) AS PEPTIDE_GENE_MAPPING ON GENE.ID = PEPTIDE_GENE_MAPPING.GENE_ID
                INNER JOIN PEPTIDE ON PEPTIDE_GENE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
                INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
                INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
                INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                GROUP BY GROUP_ID
                HAVING MAX(SCORE)
                ORDER BY SCORE DESC
                ''' % (run_id, group_id, context), con)
    else:
        raise click.ClickException("Unspecified context selected.")

    data.columns = [col.lower() for col in data.columns]
    
    if is_sqlite_file(infile):
        con.close()

    if context == 'run-specific':
        data = data.groupby('run_id').apply(statistics_report, outfile, context, "gene", parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, color_palette)
    elif context in ['global', 'experiment-wide']:
        data = statistics_report(data, outfile, context, "gene", parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, color_palette)

    # Store results
    if infile != outfile:
        copyfile(infile, outfile)

    if is_parquet_file(infile):
        init_df = pl.read_parquet(outfile)
        if context == 'global':
            df = data[['gene_id','score','p_value','q_value','pep']]
            df.columns = ['GENE_ID','SCORE','PVALUE','QVALUE','PEP']
            df = df.rename(columns=lambda x: f'SCORE_GENE_{context.upper().replace("-", "_")}_{x}' if x not in ['GENE_ID'] else x)
            df = init_df.join(pl.from_pandas(df).with_columns(pl.col('GENE_ID').cast(pl.Int64)), on=['GENE_ID'], how='left', coalesce=True)
        else:
            df = data[['run_id', 'gene_id','score','p_value','q_value','pep']]
            df.columns = ['RUN_ID', 'GENE_ID','SCORE','PVALUE','QVALUE','PEP']
            df = df.rename(columns=lambda x: f'SCORE_GENE_{context.upper().replace("-", "_")}_{x}' if x not in ['RUN_ID', 'GENE_ID'] else x)
            df = init_df.join(
                pl.from_pandas(df).with_columns(pl.col('GENE_ID').cast(pl.Int64)), 
                on=['RUN_ID', 'GENE_ID'], 
                how='left', 
                coalesce=True
            )
        df.write_parquet(
            outfile,
            compression="zstd",
            compression_level=11
        )
    else:
        con = sqlite3.connect(outfile)
        c = con.cursor()
        c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_GENE"')
        if c.fetchone()[0] == 1:
            c.execute('DELETE FROM SCORE_GENE WHERE CONTEXT =="%s"' % context)
        c.fetchall()

        df = data[['context','run_id','gene_id','score','p_value','q_value','pep']]
        df.columns = ['CONTEXT','RUN_ID','GENE_ID','SCORE','PVALUE','QVALUE','PEP']
        table = "SCORE_GENE"
        df.to_sql(table, con, index=False, dtype={"RUN_ID": "INTEGER"}, if_exists='append')

        con.close()


def infer_proteins(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, color_palette):
    
    if is_parquet_file(infile):
        all_column_names = get_parquet_column_names(infile)
        if not any([col.startswith("SCORE_MS2_") for col in all_column_names]):
            raise click.ClickException("Apply scoring to MS2-level data before running protein-level scoring.")
    else:
        con = sqlite3.connect(infile)
        if not check_sqlite_table(con, "SCORE_MS2"):
            raise click.ClickException("Apply scoring to MS2-level data before running protein-level scoring.")

    if context in ['global','experiment-wide','run-specific']:
        if context == 'global':
            run_id = 'NULL'
            group_id = 'PROTEIN.ID'
        else:
            run_id = 'RUN_ID'
            group_id = 'RUN_ID || "_" || PROTEIN.ID'

        if is_parquet_file(infile):
            # Read necessary columns from parquet
            cols = ['RUN_ID', 'PROTEIN_ID', 'PRECURSOR_DECOY', 'SCORE_MS2_SCORE', 'PEPTIDE_ID']
            data = pl.read_parquet(infile, columns=cols)
            
            data = (
                data.with_columns(
                    # Common transformations
                    pl.col('PROTEIN_ID').cast(pl.Utf8),
                    pl.col('PEPTIDE_ID').cast(pl.Utf8),
                    pl.col('PRECURSOR_DECOY').alias('DECOY')
                )
                .with_columns(
                    # Conditional transformations
                    pl.when(pl.lit(context == 'global'))
                    .then(pl.struct([
                        pl.lit(None).cast(pl.Int64).alias('RUN_ID_NEW'),
                        pl.col('PROTEIN_ID').alias('GROUP_ID')
                    ]))
                    .otherwise(pl.struct([
                        pl.col('RUN_ID').cast(pl.Int64).alias('RUN_ID_NEW'),
                        (pl.col('RUN_ID').cast(pl.Utf8) + "_" + pl.col('PROTEIN_ID')).alias('GROUP_ID')
                    ]))
                    .alias('fields')
                )
                .unnest('fields')
                .drop('RUN_ID')  # Drop original RUN_ID
                .rename({'RUN_ID_NEW': 'RUN_ID'})  # Rename to original name
                .group_by('GROUP_ID')
                .agg(
                    pl.col('SCORE_MS2_SCORE').max().alias('SCORE'),
                    pl.first('RUN_ID'),
                    pl.first('PROTEIN_ID'),
                    pl.first('DECOY')
                )
                .with_columns(
                    pl.lit(context).cast(pl.Utf8).alias('CONTEXT')
                )
                .select([
                    'RUN_ID',
                    'GROUP_ID',
                    'PROTEIN_ID',
                    'DECOY',
                    'SCORE',
                    'CONTEXT'
                ])
                .sort('SCORE', descending=True)
                .to_pandas()
            )
        else:
            con.executescript('''
                CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_protein_id ON PEPTIDE_PROTEIN_MAPPING (PROTEIN_ID);
                CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_peptide_id ON PEPTIDE_PROTEIN_MAPPING (PEPTIDE_ID);
                CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID);
                CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);
                CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);
                CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
                CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
                CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
                CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);
            ''')

            data = pd.read_sql_query('''
                SELECT %s AS RUN_ID,
                       %s AS GROUP_ID,
                       PROTEIN.ID AS PROTEIN_ID,
                       PRECURSOR.DECOY AS DECOY,
                       SCORE,
                       "%s" AS CONTEXT
                FROM PROTEIN
                INNER JOIN
                  (SELECT PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID AS PEPTIDE_ID,
                          PROTEIN_ID
                   FROM
                     (SELECT PEPTIDE_ID,
                             COUNT(*) AS NUM_PROTEINS
                      FROM PEPTIDE_PROTEIN_MAPPING
                      GROUP BY PEPTIDE_ID) AS PROTEINS_PER_PEPTIDE
                   INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PROTEINS_PER_PEPTIDE.PEPTIDE_ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
                   WHERE NUM_PROTEINS == 1) AS PEPTIDE_PROTEIN_MAPPING ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID
                INNER JOIN PEPTIDE ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID
                INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
                INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
                INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                GROUP BY GROUP_ID
                HAVING MAX(SCORE)
                ORDER BY SCORE DESC
                ''' % (run_id, group_id, context), con)
    else:
        raise click.ClickException("Unspecified context selected.")

    data.columns = [col.lower() for col in data.columns]
    
    if is_sqlite_file(infile):
        con.close()

    if context == 'run-specific':
        data = data.groupby('run_id').apply(statistics_report, outfile, context, "protein", parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, color_palette)
    elif context in ['global', 'experiment-wide']:
        data = statistics_report(data, outfile, context, "protein", parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, color_palette)

    # Store results
    if infile != outfile:
        copyfile(infile, outfile)

    if is_parquet_file(infile):
        init_df = pl.read_parquet(outfile)
        if context == 'global':
            df = data[['protein_id','score','p_value','q_value','pep']]
            df.columns = ['PROTEIN_ID','SCORE','PVALUE','QVALUE','PEP']
            df = df.rename(columns=lambda x: f'SCORE_PROTEIN_{context.upper().replace("-", "_")}_{x}' if x not in ['PROTEIN_ID'] else x)
            df = init_df.join(pl.from_pandas(df).with_columns(pl.col('PROTEIN_ID').cast(pl.Int64)), on=['PROTEIN_ID'], how='left', coalesce=True)
        else:
            df = data[['run_id', 'protein_id','score','p_value','q_value','pep']]
            df.columns = ['RUN_ID', 'PROTEIN_ID','SCORE','PVALUE','QVALUE','PEP']
            df = df.rename(columns=lambda x: f'SCORE_PROTEIN_{context.upper().replace("-", "_")}_{x}' if x not in ['RUN_ID', 'PROTEIN_ID'] else x)
            df = init_df.join(
                pl.from_pandas(df).with_columns(pl.col('PROTEIN_ID').cast(pl.Int64)), 
                on=['RUN_ID', 'PROTEIN_ID'], 
                how='left', 
                coalesce=True
            )
        df.write_parquet(
            outfile,
            compression="zstd",
            compression_level=11
        )
    else:
        con = sqlite3.connect(outfile)
        c = con.cursor()
        c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_PROTEIN"')
        if c.fetchone()[0] == 1:
            c.execute('DELETE FROM SCORE_PROTEIN WHERE CONTEXT =="%s"' % context)
        c.fetchall()

        df = data[['context','run_id','protein_id','score','p_value','q_value','pep']]
        df.columns = ['CONTEXT','RUN_ID','PROTEIN_ID','SCORE','PVALUE','QVALUE','PEP']
        table = "SCORE_PROTEIN"
        df.to_sql(table, con, index=False, dtype={"RUN_ID": "INTEGER"}, if_exists='append')

        con.close()


def infer_peptides(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, color_palette):

    if is_parquet_file(infile):
        all_column_names = get_parquet_column_names(infile)
        if not any([col.startswith("SCORE_MS2_") for col in all_column_names]):
            raise click.ClickException("Apply scoring to MS2-level data before running peptide-level scoring.")
    else:
        con = sqlite3.connect(infile)

        if not check_sqlite_table(con, "SCORE_MS2"):
            raise click.ClickException("Apply scoring to MS2-level data before running peptide-level scoring.")

    if context in ['global','experiment-wide','run-specific']:
        if context == 'global':
            run_id = 'NULL'
            group_id = 'PEPTIDE.ID'
        else:
            run_id = 'RUN_ID'
            group_id = 'RUN_ID || "_" || PEPTIDE.ID'
    
        if is_parquet_file(infile):
            cols = ['RUN_ID', 'PEPTIDE_ID', 'PRECURSOR_DECOY', 'SCORE_MS2_SCORE']
            data = pl.read_parquet(infile, columns=cols)
            data = (
                data.with_columns(
                    # Common transformations
                    pl.col('PEPTIDE_ID').cast(pl.Utf8),
                    pl.col('PRECURSOR_DECOY').alias('DECOY')
                )
                .with_columns(
                    # Conditional transformations - use different names for struct fields
                    pl.when(pl.lit(context == 'global'))
                    .then(pl.struct([
                        pl.lit(None).cast(pl.Int64).alias('RUN_ID_NEW'),  # Changed name
                        pl.col('PEPTIDE_ID').alias('GROUP_ID')
                    ]))
                    .otherwise(pl.struct([
                        pl.col('RUN_ID').cast(pl.Int64).alias('RUN_ID_NEW'),  # Changed name
                        (pl.col('RUN_ID').cast(pl.Utf8) + "_" + pl.col('PEPTIDE_ID')).alias('GROUP_ID')
                    ]))
                    .alias('fields')
                )
                .unnest('fields')
                .drop('RUN_ID')  # Drop the original RUN_ID column
                .rename({'RUN_ID_NEW': 'RUN_ID'})  # Rename back to RUN_ID
                .group_by('GROUP_ID')
                .agg(
                    pl.col('SCORE_MS2_SCORE').max().alias('SCORE'),
                    pl.first('RUN_ID'),
                    pl.first('PEPTIDE_ID'),
                    pl.first('DECOY')
                )
                .with_columns(
                    pl.lit(context).cast(pl.Utf8).alias('CONTEXT')
                )
                .select([
                    'RUN_ID',
                    'GROUP_ID',
                    'PEPTIDE_ID',
                    'DECOY',
                    'SCORE',
                    'CONTEXT'
                ])
                .sort('SCORE', descending=True)
                .to_pandas()
            )
        else:

            con.executescript('''
        CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID);
        CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);
        CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);
        CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
        CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
        CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
        CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);
        ''')

            data = pd.read_sql_query('''
        SELECT %s AS RUN_ID,
            %s AS GROUP_ID,
            PEPTIDE.ID AS PEPTIDE_ID,
            PRECURSOR.DECOY,
            SCORE,
            "%s" AS CONTEXT
        FROM PEPTIDE
        INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
        INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
        INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        GROUP BY GROUP_ID
        HAVING MAX(SCORE)
        ORDER BY SCORE DESC
        ''' % (run_id, group_id, context), con)
    else:
        raise click.ClickException("Unspecified context selected.")

    data.columns = [col.lower() for col in data.columns]
    
    if is_sqlite_file(infile):
        con.close()

    if context == 'run-specific':
        data = data.groupby('run_id').apply(statistics_report, outfile, context, "peptide", parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, color_palette)

    elif context in ['global', 'experiment-wide']:
        data = statistics_report(data, outfile, context, "peptide", parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, color_palette)

    # store data in table
    if infile != outfile:
        copyfile(infile, outfile)
    
    if is_parquet_file(infile):
        init_df = pl.read_parquet(outfile)
        if context == 'global':
            df = data[['peptide_id','score','p_value','q_value','pep']]
            df.columns = ['PEPTIDE_ID','SCORE','PVALUE','QVALUE','PEP']
            df = df.rename(columns=lambda x: f'SCORE_PEPTIDE_{context.upper().replace("-", "_")}_{x}' if x not in ['PEPTIDE_ID'] else x)
            df = init_df.join(pl.from_pandas(df).with_columns(pl.col('PEPTIDE_ID').cast(pl.Int64)), on=['PEPTIDE_ID'], how='left', coalesce=True)
        else:
            df = data[['run_id', 'peptide_id','score','p_value','q_value','pep']]
            df.columns = ['RUN_ID', 'PEPTIDE_ID','SCORE','PVALUE','QVALUE','PEP']
            df = df.rename(columns=lambda x: f'SCORE_PEPTIDE_{context.upper().replace("-", "_")}_{x}' if x not in ['RUN_ID', 'PEPTIDE_ID'] else x)
            df = init_df.join(pl.from_pandas(df).with_columns(pl.col('PEPTIDE_ID').cast(pl.Int64)), on=['RUN_ID', 'PEPTIDE_ID'], how='left', coalesce=True)
        df.write_parquet(
            outfile,
            compression="zstd",
            compression_level=11
        )
    else:
        con = sqlite3.connect(outfile)

        c = con.cursor()
        c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_PEPTIDE"')
        if c.fetchone()[0] == 1:
            c.execute('DELETE FROM SCORE_PEPTIDE WHERE CONTEXT =="%s"' % context)
        c.fetchall()

        df = data[['context','run_id','peptide_id','score','p_value','q_value','pep']]
        df.columns = ['CONTEXT','RUN_ID','PEPTIDE_ID','SCORE','PVALUE','QVALUE','PEP']
        table = "SCORE_PEPTIDE"
        df.to_sql(table, con, index=False, dtype={"RUN_ID": "INTEGER"}, if_exists='append')

        con.close()


def infer_glycopeptides(infile, outfile, context,
                        density_estimator,
                        grid_size,
                        parametric, pfdr,
                        pi0_lambda,
                        pi0_method, pi0_smooth_df, 
                        pi0_smooth_log_pi0,
                        lfdr_truncate, lfdr_monotone, 
                        # lfdr_transformation, lfdr_adj, lfdr_eps
                        ):
    '''
    Infer glycopeptides 
    Adapted from: https://github.com/lmsac/GproDIA/blob/main/src/glycoprophet/level_contexts.py
    '''
    con = sqlite3.connect(infile)
    
    if not check_sqlite_table(con, "SCORE_MS2") or \
        not check_sqlite_table(con, "SCORE_MS2_PART_PEPTIDE") or \
        not check_sqlite_table(con, "SCORE_MS2_PART_GLYCAN"):
        raise click.ClickException("Apply scoring to MS2-level data before running glycopeptide-level scoring.")
        
    if context not in ['global','experiment-wide','run-specific']:
        raise click.ClickException("Unspecified context selected.")
        
    if context == 'global':
        run_id = 'NULL'
        group_id = 'GLYCOPEPTIDE.ID'
    else:
        run_id = 'RUN_ID'
        group_id = 'RUN_ID || "_" || GLYCOPEPTIDE.ID'
        
    con.executescript('''
CREATE INDEX IF NOT EXISTS idx_glycopeptide_glycopeptide_id ON GLYCOPEPTIDE (ID);
CREATE INDEX IF NOT EXISTS idx_precursor_glycopeptide_mapping_glycopeptide_id ON PRECURSOR_GLYCOPEPTIDE_MAPPING (GLYCOPEPTIDE_ID);
CREATE INDEX IF NOT EXISTS idx_precursor_glycopeptide_mapping_precursor_id ON PRECURSOR_GLYCOPEPTIDE_MAPPING (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_ms2_part_peptide_feature_id ON SCORE_MS2_PART_PEPTIDE (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_ms2_part_glycan_feature_id ON SCORE_MS2_PART_GLYCAN (FEATURE_ID);
''')
    
    data = pd.read_sql_query('''
SELECT %s AS RUN_ID,
       %s AS GROUP_ID,
       GLYCOPEPTIDE.ID AS GLYCOPEPTIDE_ID,
       GLYCOPEPTIDE.DECOY_PEPTIDE,
       GLYCOPEPTIDE.DECOY_GLYCAN,
       SCORE_MS2.SCORE AS d_score_combined,
       SCORE_MS2_PART_PEPTIDE.SCORE AS d_score_peptide,
       SCORE_MS2_PART_GLYCAN.SCORE AS d_score_glycan,
       "%s" AS CONTEXT
FROM GLYCOPEPTIDE
INNER JOIN PRECURSOR_GLYCOPEPTIDE_MAPPING ON GLYCOPEPTIDE.ID = PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID
INNER JOIN PRECURSOR ON PRECURSOR_GLYCOPEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
INNER JOIN SCORE_MS2_PART_PEPTIDE ON FEATURE.ID = SCORE_MS2_PART_PEPTIDE.FEATURE_ID
INNER JOIN SCORE_MS2_PART_GLYCAN ON FEATURE.ID = SCORE_MS2_PART_GLYCAN.FEATURE_ID
GROUP BY GROUP_ID
HAVING MAX(d_score_combined)
ORDER BY d_score_combined DESC
''' % (run_id, group_id, context), con)
    
    data.columns = [col.lower() for col in data.columns]
    
    if context == 'run-specific':
        data = data.groupby('run_id').apply(
            statistics_report, 
            outfile, context, 'glycopeptide',
            density_estimator=density_estimator,
            grid_size=grid_size,
            parametric=parametric, pfdr=pfdr, 
            pi0_lambda=pi0_lambda, pi0_method=pi0_method, 
            pi0_smooth_df=pi0_smooth_df, 
            pi0_smooth_log_pi0=pi0_smooth_log_pi0, 
            lfdr_truncate=lfdr_truncate, 
            lfdr_monotone=lfdr_monotone, 
            # lfdr_transformation=lfdr_transformation, 
            # lfdr_adj=lfdr_adj, lfdr_eps=lfdr_eps
        ).reset_index()

    elif context in ['global', 'experiment-wide']:
        data = glyco_statistics_report(
            data, outfile, context, 'glycopeptide',
            density_estimator=density_estimator,
            grid_size=grid_size,
            parametric=parametric, pfdr=pfdr, 
            pi0_lambda=pi0_lambda, pi0_method=pi0_method, 
            pi0_smooth_df=pi0_smooth_df, 
            pi0_smooth_log_pi0=pi0_smooth_log_pi0, 
            lfdr_truncate=lfdr_truncate, 
            lfdr_monotone=lfdr_monotone, 
            # lfdr_transformation=lfdr_transformation, 
            # lfdr_adj=lfdr_adj, lfdr_eps=lfdr_eps
        )
    
    if infile != outfile:
        copyfile(infile, outfile)
    
    con = sqlite3.connect(outfile)

    c = con.cursor()
    c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_GLYCOPEPTIDE"')
    if c.fetchone()[0] == 1:
        c.execute('DELETE FROM SCORE_GLYCOPEPTIDE WHERE CONTEXT =="%s"' % context)
    c.fetchall()
    c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_GLYCOPEPTIDE_PART_PEPTIDE"')
    if c.fetchone()[0] == 1:
        c.execute('DELETE FROM SCORE_GLYCOPEPTIDE_PART_PEPTIDE WHERE CONTEXT =="%s"' % context)
    c.fetchall()
    c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_GLYCOPEPTIDE_PART_GLYCAN"')
    if c.fetchone()[0] == 1:
        c.execute('DELETE FROM SCORE_GLYCOPEPTIDE_PART_GLYCAN WHERE CONTEXT =="%s"' % context)
    c.fetchall()

    df = data[['context','run_id','glycopeptide_id','d_score_combined','q_value','pep']]
    df.columns = ['CONTEXT','RUN_ID','GLYCOPEPTIDE_ID','SCORE','QVALUE','PEP']
    table = "SCORE_GLYCOPEPTIDE"
    df.to_sql(table, con, index=False, dtype={"RUN_ID": "INTEGER"}, if_exists='append')
    
    for part in ['peptide', 'glycan']:
        df = data[['context','run_id','glycopeptide_id','d_score_' + part,'pep_' + part]]
        df.columns = ['CONTEXT','RUN_ID','GLYCOPEPTIDE_ID','SCORE','PEP']
        table = "SCORE_GLYCOPEPTIDE_PART_" + part.upper()
        df.to_sql(table, con, index=False, dtype={"RUN_ID": "INTEGER"}, if_exists='append')

    con.close()


def subsample_osw(infile, outfile, subsample_ratio, test):
    conn = sqlite3.connect(infile)
    ms1_present = check_sqlite_table(conn, "FEATURE_MS1")
    ms2_present = check_sqlite_table(conn, "FEATURE_MS2")
    transition_present = check_sqlite_table(conn, "FEATURE_TRANSITION")
    ## Check if infile contains multiple entries for run table, if only 1 entry, then infile is a single run, else infile contains multiples run
    n_runs = conn.cursor().execute("SELECT COUNT(*) AS NUMBER_OF_RUNS FROM RUN").fetchall()[0][0]
    multiple_runs = True if n_runs > 1 else False
    if multiple_runs: 
        click.echo("Warn: There are %s runs in %s" %(n_runs, infile))
    conn.close()
    
    conn = sqlite3.connect(outfile)
    c = conn.cursor()

    c.executescript('''
PRAGMA synchronous = OFF;

ATTACH DATABASE "%s" AS sdb;

CREATE TABLE RUN AS SELECT * FROM sdb.RUN;

DETACH DATABASE sdb;
''' % infile)
    click.echo("Info: Propagated runs of file %s to %s." % (infile, outfile))

    if subsample_ratio >= 1.0:
        c.executescript('''
ATTACH DATABASE "%s" AS sdb; 

CREATE TABLE FEATURE AS SELECT * FROM sdb.FEATURE; 

DETACH DATABASE sdb;
''' % infile)
    else:
        if test:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

CREATE TABLE FEATURE AS 
SELECT *
FROM sdb.FEATURE
WHERE PRECURSOR_ID IN
    (SELECT ID
     FROM sdb.PRECURSOR
     LIMIT
       (SELECT ROUND(%s*COUNT(DISTINCT ID))
        FROM sdb.PRECURSOR));

DETACH DATABASE sdb;
''' % (infile, subsample_ratio))
        else:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

CREATE TABLE FEATURE AS 
SELECT *
FROM sdb.FEATURE
WHERE PRECURSOR_ID IN
    (SELECT ID
     FROM sdb.PRECURSOR
     ORDER BY RANDOM()
     LIMIT
       (SELECT ROUND(%s*COUNT(DISTINCT ID))
        FROM sdb.PRECURSOR));

DETACH DATABASE sdb;
''' % (infile, subsample_ratio))
    click.echo("Info: Subsampled generic features of file %s to %s." % (infile, outfile))

    if ms1_present:
        if subsample_ratio >= 1.0:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

CREATE TABLE FEATURE_MS1 AS 
SELECT *
FROM sdb.FEATURE_MS1;

DETACH DATABASE sdb;
''' % infile)
        else:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

CREATE TABLE FEATURE_MS1 AS 
SELECT *
FROM sdb.FEATURE_MS1
WHERE sdb.FEATURE_MS1.FEATURE_ID IN
    (SELECT ID
     FROM FEATURE);

DETACH DATABASE sdb;
''' % infile)
        click.echo("Info: Subsampled MS1 features of file %s to %s." % (infile, outfile))

    if ms2_present:
        if subsample_ratio >= 1.0:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

CREATE TABLE FEATURE_MS2 AS 
SELECT *
FROM sdb.FEATURE_MS2;

DETACH DATABASE sdb;
''' % infile)
        else:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

CREATE TABLE FEATURE_MS2 AS 
SELECT *
FROM sdb.FEATURE_MS2
WHERE sdb.FEATURE_MS2.FEATURE_ID IN
    (SELECT ID
     FROM FEATURE);

DETACH DATABASE sdb;
''' % infile)
        click.echo("Info: Subsampled MS2 features of file %s to %s." % (infile, outfile))

    if transition_present:
        if subsample_ratio >= 1.0:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

CREATE TABLE FEATURE_TRANSITION AS 
SELECT *
FROM sdb.FEATURE_TRANSITION;

DETACH DATABASE sdb;
''' % infile)
        else:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

CREATE TABLE FEATURE_TRANSITION AS 
SELECT *
FROM sdb.FEATURE_TRANSITION
WHERE sdb.FEATURE_TRANSITION.FEATURE_ID IN
    (SELECT ID
     FROM FEATURE);

DETACH DATABASE sdb;
''' % infile)
        click.echo("Info: Subsampled transition features of file %s to %s." % (infile, outfile))

    if multiple_runs:
        c.executescript('''
PRAGMA synchronous = OFF;

ATTACH DATABASE "%s" AS sdb;

CREATE TABLE PRECURSOR AS 
SELECT * 
FROM sdb.PRECURSOR
WHERE sdb.PRECURSOR.ID IN
    (SELECT PRECURSOR_ID
     FROM FEATURE);

DETACH DATABASE sdb;
''' % infile)
        click.echo("Info: Subsampled precursor table of file %s to %s. For scoring merged subsampled file." % (infile, outfile))

        c.executescript('''
PRAGMA synchronous = OFF;

ATTACH DATABASE "%s" AS sdb;

CREATE TABLE TRANSITION_PRECURSOR_MAPPING AS 
SELECT * 
FROM sdb.TRANSITION_PRECURSOR_MAPPING
WHERE sdb.TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID IN
    (SELECT ID
     FROM PRECURSOR);

DETACH DATABASE sdb;
''' % infile)
        click.echo("Info: Subsampled transition_precursor_mapping table of file %s to %s. For scoring merged subsampled file." % (infile, outfile)) 


        c.executescript('''
PRAGMA synchronous = OFF;

ATTACH DATABASE "%s" AS sdb;

CREATE TABLE TRANSITION AS 
SELECT * 
FROM sdb.TRANSITION
WHERE sdb.TRANSITION.ID IN
    (SELECT TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID
     FROM TRANSITION_PRECURSOR_MAPPING);

DETACH DATABASE sdb;
''' % infile)
        click.echo("Info: Subsampled transition table of file %s to %s. For scoring merged subsampled file." % (infile, outfile)) 

    conn.commit()
    conn.close()

    click.echo("Info: OSW file was subsampled.")


def reduce_osw(infile, outfile):
    conn = sqlite3.connect(infile)
    if not check_sqlite_table(conn, "SCORE_MS2"):
        raise click.ClickException("Apply scoring to MS2 data before reducing file for multi-run scoring.")
    conn.close()

    try:
        os.remove(outfile)
    except OSError:
        pass

    conn = sqlite3.connect(outfile)
    c = conn.cursor()

    c.executescript('''
PRAGMA synchronous = OFF;

ATTACH DATABASE "%s" AS sdb;

CREATE TABLE RUN(ID INT PRIMARY KEY NOT NULL,
                 FILENAME TEXT NOT NULL);

INSERT INTO RUN
SELECT *
FROM sdb.RUN;

CREATE TABLE SCORE_MS2(FEATURE_ID INTEGER, SCORE REAL);

INSERT INTO SCORE_MS2 (FEATURE_ID, SCORE)
SELECT FEATURE_ID,
       SCORE
FROM sdb.SCORE_MS2
WHERE RANK == 1;

CREATE TABLE FEATURE(ID INT PRIMARY KEY NOT NULL,
                     RUN_ID INT NOT NULL,
                     PRECURSOR_ID INT NOT NULL);

INSERT INTO FEATURE (ID, RUN_ID, PRECURSOR_ID)
SELECT ID,
       RUN_ID,
       PRECURSOR_ID
FROM sdb.FEATURE
WHERE ID IN
    (SELECT FEATURE_ID
     FROM SCORE_MS2);
''' % infile)

    conn.commit()
    conn.close()

    click.echo("Info: OSW file was reduced for multi-run scoring.")


def merge_osw(infiles, outfile, templatefile, same_run, merge_post_scored_runs):
    conn = sqlite3.connect(infiles[0])
    reduced = check_sqlite_table(conn, "SCORE_MS2")
    conn.close()
    if reduced and not merge_post_scored_runs:
        click.echo("Calling reduced osws merge function")
        merge_oswr(infiles, outfile, templatefile, same_run)
    elif merge_post_scored_runs:
        click.echo("Calling post scored osws merge function")
        merge_oswps(infiles, outfile, templatefile, same_run)
    else:
        click.echo("Calling pre scored osws merge function")
        merge_osws(infiles, outfile, templatefile, same_run)


def merge_osws(infiles, outfile, templatefile, same_run):
    # Copy the first file to have a template
    copyfile(templatefile, outfile)
    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    if same_run:
        c.execute("SELECT ID, FILENAME FROM RUN")
        result = c.fetchall()
        if len(result) != 1:
            raise click.ClickException("Input for same-run merge contains more than one run.")
        runid, rname = result[0]

    c.executescript('''
PRAGMA synchronous = OFF;

DROP TABLE IF EXISTS RUN;

DROP TABLE IF EXISTS FEATURE;

DROP TABLE IF EXISTS FEATURE_MS1;

DROP TABLE IF EXISTS FEATURE_MS2;

DROP TABLE IF EXISTS FEATURE_TRANSITION;

DROP TABLE IF EXISTS SCORE_MS1;

DROP TABLE IF EXISTS SCORE_MS2;

DROP TABLE IF EXISTS SCORE_TRANSITION;

DROP TABLE IF EXISTS SCORE_PEPTIDE;

DROP TABLE IF EXISTS SCORE_PROTEIN;

DROP TABLE IF EXISTS SCORE_IPF;

ATTACH DATABASE "%s" AS sdb;

CREATE TABLE RUN AS SELECT * FROM sdb.RUN LIMIT 0;

CREATE TABLE FEATURE AS SELECT * FROM sdb.FEATURE LIMIT 0;

CREATE TABLE FEATURE_MS1 AS SELECT * FROM sdb.FEATURE_MS1 LIMIT 0;

CREATE TABLE FEATURE_MS2 AS SELECT * FROM sdb.FEATURE_MS2 LIMIT 0;

CREATE TABLE FEATURE_TRANSITION AS SELECT * FROM sdb.FEATURE_TRANSITION LIMIT 0;

DETACH DATABASE sdb;
''' % (infiles[0]))

    conn.commit()
    conn.close()

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Only create a single run entry (all files are presumably from the same run)
        if same_run:
            c.executescript('''INSERT INTO RUN (ID, FILENAME) VALUES (%s, '%s')''' % (runid, rname) )
            break;
        else:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO RUN SELECT * FROM sdb.RUN;

DETACH DATABASE sdb;
''' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged runs of file %s to %s." % (infile, outfile))

    # Now merge the run-specific data into the output file:
    #   Note: only tables FEATURE, FEATURE_MS1, FEATURE_MS2 and FEATURE_TRANSITION are run-specific
    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript('''
ATTACH DATABASE "%s" AS sdb; 

INSERT INTO FEATURE SELECT * FROM sdb.FEATURE; 

DETACH DATABASE sdb;
''' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged generic features of file %s to %s." % (infile, outfile))
        
    if same_run:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Fix run id assuming we only have a single run
        c.executescript('''UPDATE FEATURE SET RUN_ID = %s''' % runid)

        conn.commit()
        conn.close()

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO FEATURE_MS1
SELECT *
FROM sdb.FEATURE_MS1;

DETACH DATABASE sdb;
''' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged MS1 features of file %s to %s." % (infile, outfile))

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO FEATURE_MS2
SELECT *
FROM sdb.FEATURE_MS2;

DETACH DATABASE sdb;
''' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged MS2 features of file %s to %s." % (infile, outfile))

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO FEATURE_TRANSITION
SELECT *
FROM sdb.FEATURE_TRANSITION;

DETACH DATABASE sdb;
''' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged transition features of file %s to %s." % (infile, outfile))

    click.echo("Info: All OSWS files were merged.")


def merge_oswr(infiles, outfile, templatefile, same_run):
    # Copy the template to the output file
    copyfile(templatefile, outfile)
    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    if same_run:
        c.execute("SELECT ID, FILENAME FROM RUN")
        result = c.fetchall()
        if len(result) != 1:
            raise click.ClickException("Input for same-run merge contains more than one run.")
        runid, rname = result[0]

    c.executescript('''
PRAGMA synchronous = OFF;

DROP TABLE IF EXISTS RUN;

DROP TABLE IF EXISTS FEATURE;

DROP TABLE IF EXISTS FEATURE_MS1;

DROP TABLE IF EXISTS FEATURE_MS2;

DROP TABLE IF EXISTS FEATURE_TRANSITION;

DROP TABLE IF EXISTS SCORE_MS1;

DROP TABLE IF EXISTS SCORE_MS2;

DROP TABLE IF EXISTS SCORE_TRANSITION;

DROP TABLE IF EXISTS SCORE_PEPTIDE;

DROP TABLE IF EXISTS SCORE_PROTEIN;

DROP TABLE IF EXISTS SCORE_IPF;

CREATE TABLE RUN(ID INT PRIMARY KEY NOT NULL,
                 FILENAME TEXT NOT NULL);

CREATE TABLE SCORE_MS2(FEATURE_ID INTEGER, SCORE REAL);

CREATE TABLE FEATURE(ID INT PRIMARY KEY NOT NULL,
                     RUN_ID INT NOT NULL,
                     PRECURSOR_ID INT NOT NULL);
''')

    conn.commit()
    conn.close()

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Only create a single run entry (all files are presumably from the same run)
        if same_run:
            c.executescript('''INSERT INTO RUN (ID, FILENAME) VALUES (%s, '%s')''' % (runid, rname) )
            break;
        else:
            c.executescript('ATTACH DATABASE "%s" AS sdb; INSERT INTO RUN SELECT * FROM sdb.RUN; DETACH DATABASE sdb;' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged runs of file %s to %s." % (infile, outfile))

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript('ATTACH DATABASE "%s" AS sdb; INSERT INTO FEATURE SELECT * FROM sdb.FEATURE; DETACH DATABASE sdb;' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged generic features of file %s to %s." % (infile, outfile))

    if same_run:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Fix run id assuming we only have a single run
        c.executescript('''UPDATE FEATURE SET RUN_ID = %s''' % runid)

        conn.commit()
        conn.close()

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript('ATTACH DATABASE "%s" AS sdb; INSERT INTO SCORE_MS2 SELECT * FROM sdb.SCORE_MS2; DETACH DATABASE sdb;' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged MS2 scores of file %s to %s." % (infile, outfile))

    click.echo("Info: All reduced OSWR files were merged.")

def merge_oswps(infiles, outfile, templatefile, same_run):
    click.echo("Info: Merging all Scored Runs.")
    # Copy the first file to have a template
    copyfile(templatefile, outfile)
    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    if same_run:
        c.execute("SELECT ID, FILENAME FROM RUN")
        result = c.fetchall()
        if len(result) != 1:
            raise click.ClickException("Input for same-run merge contains more than one run.")
        runid, rname = result[0]

    original_tables = c.execute(''' SELECT name FROM sqlite_master WHERE type='table'; ''')
    original_tables = [name[0] for name in original_tables]
    ## Get Score tables table_present
    score_tables = [name for name in original_tables if "SCORE" in name]
    if len(score_tables) > 0:
        create_scores_query = '\n'.join( ['CREATE TABLE ' + score_tbl + ' AS SELECT * FROM sdb.' + score_tbl + ' LIMIT 0;' for score_tbl in score_tables] )
    else:
        create_scores_query = ""
        
    ## Get Feature Alignment tables table_present
    feature_alignment_tables = [name for name in original_tables if "ALIGNMENT" in name]
    feature_alignment_tables_present = False
    if len(feature_alignment_tables) > 0:
        create_feature_alignment_query = '\n'.join( ['CREATE TABLE ' + feature_alignment_tbl + ' AS SELECT * FROM sdb.' + feature_alignment_tbl + ' LIMIT 0;' for feature_alignment_tbl in feature_alignment_tables] )
        feature_alignment_tables_present = True
    else:
        create_feature_alignment_query = ""

    click.echo( '''First File input: %s''' %( infiles[0] ) )

    c.executescript('''
    PRAGMA synchronous = OFF;
    DROP TABLE IF EXISTS RUN;
    DROP TABLE IF EXISTS FEATURE;
    DROP TABLE IF EXISTS FEATURE_MS1;
    DROP TABLE IF EXISTS FEATURE_MS2;
    DROP TABLE IF EXISTS FEATURE_TRANSITION;
    DROP TABLE IF EXISTS FEATURE_ALIGNMENT;
    DROP TABLE IF EXISTS FEATURE_MS2_ALIGNMENT;
    DROP TABLE IF EXISTS FEATURE_TRANSITION_ALIGNMENT;
    DROP TABLE IF EXISTS SCORE_MS1;
    DROP TABLE IF EXISTS SCORE_MS2;
    DROP TABLE IF EXISTS SCORE_TRANSITION;
    DROP TABLE IF EXISTS SCORE_PEPTIDE;
    DROP TABLE IF EXISTS SCORE_PROTEIN;
    DROP TABLE IF EXISTS SCORE_IPF;
    ATTACH DATABASE "%s" AS sdb;
    CREATE TABLE RUN AS SELECT * FROM sdb.RUN LIMIT 0;
    CREATE TABLE FEATURE AS SELECT * FROM sdb.FEATURE LIMIT 0;
    CREATE TABLE FEATURE_MS1 AS SELECT * FROM sdb.FEATURE_MS1 LIMIT 0;
    CREATE TABLE FEATURE_MS2 AS SELECT * FROM sdb.FEATURE_MS2 LIMIT 0;
    CREATE TABLE FEATURE_TRANSITION AS SELECT * FROM sdb.FEATURE_TRANSITION
    LIMIT 0;
    %s
    %s
    DETACH DATABASE sdb;
    ''' % (infiles[0], create_feature_alignment_query, create_scores_query) )

    conn.commit()
    conn.close()

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Only create a single run entry (all files are presumably from the same run)
        if same_run:
            c.executescript('''INSERT INTO RUN (ID, FILENAME) VALUES (%s, '%s')''' % (runid, rname) )
            break;
        else:
            c.executescript('''
    ATTACH DATABASE "%s" AS sdb;
    INSERT INTO RUN SELECT * FROM sdb.RUN;
    DETACH DATABASE sdb;
    ''' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged runs of file %s to %s." % (infile, outfile))

    # Now merge the run-specific data into the output file:
    #   Note: only tables FEATURE, FEATURE_MS1, FEATURE_MS2 and FEATURE_TRANSITION are run-specific
    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript('''
    ATTACH DATABASE "%s" AS sdb; 
    INSERT INTO FEATURE SELECT * FROM sdb.FEATURE; 
    DETACH DATABASE sdb;
    ''' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged generic features of file %s to %s." % (infile, outfile))

    if same_run:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Fix run id assuming we only have a single run
        c.executescript('''UPDATE FEATURE SET RUN_ID = %s''' % runid)

        conn.commit()
        conn.close()

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript('''
    ATTACH DATABASE "%s" AS sdb;
    INSERT INTO FEATURE_MS1
    SELECT *
    FROM sdb.FEATURE_MS1;
    DETACH DATABASE sdb;
    ''' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged MS1 features of file %s to %s." % (infile, outfile))

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript('''
    ATTACH DATABASE "%s" AS sdb;
    INSERT INTO FEATURE_MS2
    SELECT *
    FROM sdb.FEATURE_MS2;
    DETACH DATABASE sdb;
    ''' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged MS2 features of file %s to %s." % (infile, outfile))

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript('''
    ATTACH DATABASE "%s" AS sdb;
    INSERT INTO FEATURE_TRANSITION
    SELECT *
    FROM sdb.FEATURE_TRANSITION;
    DETACH DATABASE sdb;
    ''' % infile)

        conn.commit()
        conn.close()

        click.echo("Info: Merged transition features of file %s to %s." % (infile, outfile))
        
    if feature_alignment_tables_present:
        for infile in infiles:
            
            # Check if the infile contains the feature_alignment table
            conn = sqlite3.connect(infile)
            feature_alignment_present = check_sqlite_table(conn, "FEATURE_ALIGNMENT")
            conn.close()
            
            if feature_alignment_present:
                conn = sqlite3.connect(outfile)
                c = conn.cursor()

                c.executescript('''
            ATTACH DATABASE "%s" AS sdb;
            INSERT INTO FEATURE_ALIGNMENT
            SELECT *
            FROM sdb.FEATURE_ALIGNMENT;
            DETACH DATABASE sdb;
            ''' % infile)
                    
                conn.commit()
                conn.close()
                
                click.echo("Info: Merged feature alignment tables of file %s to %s." % (infile, outfile))
            else:
                click.echo("Warn: No feature alignment table found in file %s." % (infile))
                
        # Merge FEATURE_MS2_ALIGNMENT
        for infile in infiles:

            conn = sqlite3.connect(infile)
            feature_ms2_alignment_present = check_sqlite_table(conn, "FEATURE_MS2_ALIGNMENT")
            conn.close()
            
            if feature_ms2_alignment_present:
                conn = sqlite3.connect(outfile)
                c = conn.cursor()

                c.executescript('''
            ATTACH DATABASE "%s" AS sdb;
            INSERT INTO FEATURE_MS2_ALIGNMENT
            SELECT *
            FROM sdb.FEATURE_MS2_ALIGNMENT;
            DETACH DATABASE sdb;
            ''' % infile)
                
                conn.commit()
                conn.close()
                
                click.echo("Info: Merged feature MS2 alignment tables of file %s to %s." % (infile, outfile))
            else:
                click.echo("Warn: No feature MS2 alignment table found in file %s." % (infile))
                
        # Merge FEATURE_TRANSITION_ALIGNMENT
        for infile in infiles:
                
                conn = sqlite3.connect(infile)
                feature_transition_alignment_present = check_sqlite_table(conn, "FEATURE_TRANSITION_ALIGNMENT")
                conn.close()
                
                if feature_transition_alignment_present:
                    conn = sqlite3.connect(outfile)
                    c = conn.cursor()
        
                    c.executescript('''
                ATTACH DATABASE "%s" AS sdb;
                INSERT INTO FEATURE_TRANSITION_ALIGNMENT
                SELECT *
                FROM sdb.FEATURE_TRANSITION_ALIGNMENT;
                DETACH DATABASE sdb;
                ''' % infile)
                    
                    conn.commit()
                    conn.close()
                    
                    click.echo("Info: Merged feature transition alignment tables of file %s to %s." % (infile, outfile))
                else:
                    click.echo("Warn: No feature transition alignment table found in file %s." % (infile))
                

    for infile in infiles:
        for score_tbl in score_tables:
            conn = sqlite3.connect(outfile)
            c = conn.cursor()

            c.executescript('''
    ATTACH DATABASE "%s" AS sdb;
    INSERT INTO %s
    SELECT *
    FROM sdb.%s;
    DETACH DATABASE sdb;
    ''' % (infile, score_tbl, score_tbl) )

            conn.commit()
            conn.close()

            click.echo("Info: Merged %s table of file %s to %s." % (score_tbl, infile, outfile))

    ## Vacuum to clean and re-write rootpage indexes
    conn =  sqlite3.connect(outfile)
    c = conn.cursor()

    c.executescript('VACUUM')

    conn.commit()
    conn.close()
    
    click.echo("Info: Cleaned and re-wrote indexing meta-data for %s.: " % (outfile))

    click.echo("Info: All Post-Scored OSWS files were merged.")

def backpropagate_oswr(infile, outfile, apply_scores):
    # store data in table
    if infile != outfile:
        copyfile(infile, outfile)

    # find out what tables exist in the scores
    score_con = sqlite3.connect(apply_scores)
    peptide_present = check_sqlite_table(score_con, "SCORE_PEPTIDE")
    protein_present = check_sqlite_table(score_con, "SCORE_PROTEIN")
    score_con.close()
    if not (peptide_present or protein_present):
        raise click.ClickException('Backpropagation requires peptide or protein-level contexts.')

    # build up the list
    script = list()
    script.append('PRAGMA synchronous = OFF;')
    script.append('DROP TABLE IF EXISTS SCORE_PEPTIDE;')
    script.append('DROP TABLE IF EXISTS SCORE_PROTEIN;')

    # create the tables
    if peptide_present:
        script.append('CREATE TABLE SCORE_PEPTIDE (CONTEXT TEXT, RUN_ID INTEGER, PEPTIDE_ID INTEGER, SCORE REAL, PVALUE REAL, QVALUE REAL, PEP REAL);')
    if protein_present:
        script.append('CREATE TABLE SCORE_PROTEIN (CONTEXT TEXT, RUN_ID INTEGER, PROTEIN_ID INTEGER, SCORE REAL, PVALUE REAL, QVALUE REAL, PEP REAL);')

    # copy across the tables
    script.append('ATTACH DATABASE "{}" AS sdb;'.format(apply_scores))
    insert_table_fmt = 'INSERT INTO {0}\nSELECT *\nFROM sdb.{0};'
    if peptide_present:
        script.append(insert_table_fmt.format('SCORE_PEPTIDE'))
    if protein_present:
        script.append(insert_table_fmt.format('SCORE_PROTEIN'))

    # execute the script
    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    c.executescript('\n'.join(script))
    conn.commit()
    conn.close()

    click.echo("Info: All multi-run data was backpropagated.")
