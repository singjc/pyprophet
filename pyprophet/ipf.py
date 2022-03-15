import pandas as pd
import numpy as np
import scipy as sp
import sqlite3
import sys
import click
import itertools
import time
	
from scipy.stats import rankdata
from .data_handling import check_sqlite_table
from shutil import copyfile

def generate_peptide_combinations(arr, max_combs):
	combs = []
	for i in range(1, max_combs+1):
		if i > 1:
			# remove h0
			arr = np.delete(arr,-1)	
		for mp in list(itertools.combinations(arr, i)):
			combs.append(pd.DataFrame({'num_multi_peptides': i, 'multi_peptide_id': '_'.join(str(x) for x in mp), 'peptide_id': mp}))
	
	return pd.concat(combs)


def compute_model_fdr(data_in):
	data = np.asarray(data_in)
	
	# compute model based FDR estimates from posterior error probabilities
	order = np.argsort(data)
	
	ranks = np.zeros(data.shape[0], dtype=np.int)
	fdr = np.zeros(data.shape[0])
	
	# rank data with with maximum ranks for ties
	ranks[order] = rankdata(data[order], method='max')
	
	# compute FDR/q-value by using cumulative sum of maximum rank for ties
	fdr[order] = data[order].cumsum()[ranks[order]-1] / ranks[order]
	
	return fdr


def read_pyp_peakgroup_precursor(path, ipf_max_peakgroup_pep, ipf_ms1_scoring, ipf_ms2_scoring):
	click.echo("Info: Reading precursor-level data.")
	# precursors are restricted according to ipf_max_peakgroup_pep to exclude very poor peak groups
	con = sqlite3.connect(path)
	
	# only use MS2 precursors
	if not ipf_ms1_scoring and ipf_ms2_scoring:
		if not check_sqlite_table(con, "SCORE_MS2") or not check_sqlite_table(con, "SCORE_TRANSITION"):
			sys.exit("Error: Apply scoring to MS2 and transition-level data before running IPF.")
		data = pd.read_sql_query('''
								  SELECT FEATURE.ID AS FEATURE_ID,
										 SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
										 NULL AS MS1_PRECURSOR_PEP,
										 SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PEP
								  FROM PRECURSOR
								  INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
								  INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
								  INNER JOIN
									(SELECT FEATURE_ID,
											PEP
									 FROM SCORE_TRANSITION
									 INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
									 WHERE TRANSITION.TYPE=''
									   AND TRANSITION.DECOY=0) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID
								  WHERE PRECURSOR.DECOY=0
									AND SCORE_MS2.PEP < %s;
								  ''' % ipf_max_peakgroup_pep, con)
	
	# only use MS1 precursors
	elif ipf_ms1_scoring and not ipf_ms2_scoring:
		if not check_sqlite_table(con, "SCORE_MS1") or not check_sqlite_table(con, "SCORE_MS2") or not check_sqlite_table(con, "SCORE_TRANSITION"):
			sys.exit("Error: Apply scoring to MS1, MS2 and transition-level data before running IPF.")
		data = pd.read_sql_query('''
								  SELECT FEATURE.ID AS FEATURE_ID,
										 SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
										 SCORE_MS1.PEP AS MS1_PRECURSOR_PEP,
										 NULL AS MS2_PRECURSOR_PEP
								  FROM PRECURSOR
								  INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
								  INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
								  INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
								  WHERE PRECURSOR.DECOY=0
									AND SCORE_MS2.PEP < %s;
								  ''' % ipf_max_peakgroup_pep, con)
	
	# use both MS1 and MS2 precursors
	elif ipf_ms1_scoring and ipf_ms2_scoring:
		if not check_sqlite_table(con, "SCORE_MS1") or not check_sqlite_table(con, "SCORE_MS2") or not check_sqlite_table(con, "SCORE_TRANSITION"):
			sys.exit("Error: Apply scoring to MS1, MS2 and transition-level data before running IPF.")
		data = pd.read_sql_query('''
								  SELECT FEATURE.ID AS FEATURE_ID,
										 SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
										 SCORE_MS1.PEP AS MS1_PRECURSOR_PEP,
										 SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PEP
								  FROM PRECURSOR
								  INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
								  INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
								  INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
								  INNER JOIN
									(SELECT FEATURE_ID,
											PEP
									 FROM SCORE_TRANSITION
									 INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
									 WHERE TRANSITION.TYPE=''
									   AND TRANSITION.DECOY=0) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID
								  WHERE PRECURSOR.DECOY=0
									AND SCORE_MS2.PEP < %s;
								  ''' % ipf_max_peakgroup_pep, con)
	
	# do not use any precursor information
	else:
		if not check_sqlite_table(con, "SCORE_MS2") or not check_sqlite_table(con, "SCORE_TRANSITION"):
			sys.exit("Error: Apply scoring to MS2  and transition-level data before running IPF.")
		data = pd.read_sql_query('''
								  SELECT FEATURE.ID AS FEATURE_ID,
										 SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
										 NULL AS MS1_PRECURSOR_PEP,
										 NULL AS MS2_PRECURSOR_PEP
								  FROM PRECURSOR
								  INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
								  INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
								  WHERE PRECURSOR.DECOY=0
									AND SCORE_MS2.PEP < %s;
								  ''' % ipf_max_peakgroup_pep, con)
	
	data.columns = [col.lower() for col in data.columns]
	con.close()
	
	return data


def read_pyp_transition(path, ipf_max_transition_pep, ipf_h0, ipf_multi):
	click.echo("Info: Reading peptidoform-level data.")
	# only the evidence is restricted to ipf_max_transition_pep, the peptidoform-space is complete
	con = sqlite3.connect(path)
	
	# transition-level evidence
	evidence = pd.read_sql_query('''
								SELECT FEATURE_ID,
									   TRANSITION_ID,
									   PEP
								FROM SCORE_TRANSITION
								INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
								WHERE TRANSITION.TYPE!=''
								  AND TRANSITION.DECOY=0
								  AND PEP < %s;
								 ''' % ipf_max_transition_pep, con)
	evidence.columns = [col.lower() for col in evidence.columns]
	
	# transition-level bitmask
	bitmask = pd.read_sql_query('''
								SELECT DISTINCT TRANSITION.ID AS TRANSITION_ID,
												PEPTIDE_ID,
												1 AS BMASK
								FROM TRANSITION
								INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
								WHERE TRANSITION.TYPE!=''
								  AND TRANSITION.DECOY=0;
								''', con)
	bitmask.columns = [col.lower() for col in bitmask.columns]
	
	# potential peptidoforms per feature
	num_peptidoforms = pd.read_sql_query('''
										  SELECT FEATURE_ID,
												 COUNT(DISTINCT PEPTIDE_ID) AS NUM_PEPTIDOFORMS
										  FROM SCORE_TRANSITION
										  INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
										  INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
										  WHERE TRANSITION.TYPE!=''
											AND TRANSITION.DECOY=0
										  GROUP BY FEATURE_ID
										  ORDER BY FEATURE_ID;
										  ''', con)
	num_peptidoforms.columns = [col.lower() for col in num_peptidoforms.columns]
	
	# peptidoform space per feature
	peptidoforms = pd.read_sql_query('''
									  SELECT DISTINCT FEATURE_ID,
													  PEPTIDE_ID
									  FROM SCORE_TRANSITION
									  INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
									  INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
									  WHERE TRANSITION.TYPE!=''
										AND TRANSITION.DECOY=0
									  ORDER BY FEATURE_ID;
									  ''', con)
	peptidoforms.columns = [col.lower() for col in peptidoforms.columns]
	
	con.close()
	
	# add h0 (peptide_id: -1) to peptidoform-space if necessary
	if ipf_h0:
		peptidoforms = pd.concat([peptidoforms, pd.DataFrame({'feature_id': peptidoforms['feature_id'].unique(), 'peptide_id': -1})])
	
	if ipf_multi > 1:
		peptidoforms = peptidoforms.groupby('feature_id').apply(lambda x: generate_peptide_combinations(x['peptide_id'].values, ipf_multi)).reset_index(level='feature_id')
		num_peptidoforms = peptidoforms.groupby(['feature_id']).apply(lambda x: pd.Series({'num_peptidoforms': x['multi_peptide_id'].unique().shape[0]})).reset_index(level=['feature_id'])
		# ensure that only site-specific transitions are used, else we can't differentiate if peptidoforms are actually present or IPF can't distinguish them
		transition_specificity = bitmask.groupby('transition_id').size().reset_index(name='num_peptidoforms')
		transition_specificity = transition_specificity[transition_specificity['num_peptidoforms'] == 1]
		evidence = evidence.merge(transition_specificity[['transition_id']], how='inner', on='transition_id')
		bitmask = bitmask.merge(transition_specificity[['transition_id']], how='inner', on='transition_id')
	else:
		peptidoforms['multi_peptide_id'] = peptidoforms['peptide_id']
		peptidoforms['num_multi_peptides'] = 1
	
	# generate transition-peptidoform table
	trans_pf = pd.merge(evidence, peptidoforms, how='outer', on='feature_id')
	
	# apply bitmask
	trans_pf_bm = pd.merge(trans_pf, bitmask, how='left', on=['transition_id','peptide_id']).fillna(0)
	
	# append number of peptidoforms
	data = pd.merge(trans_pf_bm, num_peptidoforms, how='inner', on='feature_id')
	
	return data


def prepare_precursor_bm(data):
	# MS1-level precursors
	ms1_precursor_data = data[['feature_id','ms2_peakgroup_pep','ms1_precursor_pep']].dropna(axis=0, how='any')
	ms1_bm_data = pd.concat([pd.DataFrame({'feature_id': ms1_precursor_data['feature_id'], 'prior': 1-ms1_precursor_data['ms2_peakgroup_pep'], 'evidence': 1-ms1_precursor_data['ms1_precursor_pep'], 'hypothesis': True}), pd.DataFrame({'feature_id': ms1_precursor_data['feature_id'], 'prior': ms1_precursor_data['ms2_peakgroup_pep'], 'evidence': ms1_precursor_data['ms1_precursor_pep'], 'hypothesis': False})])
	
	# MS2-level precursors
	ms2_precursor_data = data[['feature_id','ms2_peakgroup_pep','ms2_precursor_pep']].dropna(axis=0, how='any')
	ms2_bm_data = pd.concat([pd.DataFrame({'feature_id': ms2_precursor_data['feature_id'], 'prior': 1-ms2_precursor_data['ms2_peakgroup_pep'], 'evidence': 1-ms2_precursor_data['ms2_precursor_pep'], 'hypothesis': True}), pd.DataFrame({'feature_id': ms2_precursor_data['feature_id'], 'prior': ms2_precursor_data['ms2_peakgroup_pep'], 'evidence': ms2_precursor_data['ms2_precursor_pep'], 'hypothesis': False})])
	
	# missing precursor data
	missing_precursor_data = data[['feature_id','ms2_peakgroup_pep']].dropna(axis=0, how='any').drop_duplicates()
	missing_bm_data = pd.concat([pd.DataFrame({'feature_id': missing_precursor_data['feature_id'], 'prior': 1-missing_precursor_data['ms2_peakgroup_pep'], 'evidence': 0, 'hypothesis': True}), pd.DataFrame({'feature_id': missing_precursor_data['feature_id'], 'prior': missing_precursor_data['ms2_peakgroup_pep'], 'evidence': 1, 'hypothesis': False})])
	
	# combine precursor data
	precursor_bm_data = pd.concat([ms1_bm_data, ms2_bm_data])
	# append missing precursors if no MS1/MS2 evidence is available
	precursor_bm_data = pd.concat([precursor_bm_data, missing_bm_data.loc[~missing_bm_data['feature_id'].isin(precursor_bm_data['feature_id'])]])
	
	return(precursor_bm_data)

def transfer_confident_evidence_across_runs(df1, df2, across_run_confidence_threshold):
	# Check to see if current feature data was aligned or not, based on the fact that there should be multiple feature_ids for run across_run_feature_id
	# TODO: find a better way to check for this
	unique_feature_ids_per_across_run_feature_id = np.unique(df2['feature_id'].loc[ np.isin(df2['across_run_feature_id'], df1['across_run_feature_id']) ])
	if unique_feature_ids_per_across_run_feature_id.shape[0]==1:
		new_df = df1
	else:
		across_run_data = df2.loc[df2.across_run_feature_id==df1.across_run_feature_id.unique()[0]]
		new_df = pd.concat([df1, across_run_data.loc[across_run_data.pep <= across_run_confidence_threshold]], ignore_index=True)
		new_df['feature_id'] = df1['feature_id'].iloc[0]
	return new_df

def prepare_transition_bm(data, propagate_signal_across_runs, across_run_confidence_threshold):
	# Propagate peps <= threshold for aligned feature groups across runs
	if ( propagate_signal_across_runs ){
  	## Separate out features that need propagation and those that don't to avoid calling apply on the features that don't need propagated peps
  	non_prop_data = data.loc[ data['feature_id']==data['across_run_feature_id']]
  	prop_data = data.loc[ data['feature_id']!=data['across_run_feature_id']]
  	start = time.time()
  	data_with_confidence = prop_data.groupby(["feature_id"]).apply(transfer_confident_evidence_across_runs, prop_data, across_run_confidence_threshold)
  	end = time.time()
  	click.echo(f"INFO: Elapsed time for propagating peps for aligned features across runs {end-start} seconds")
  	## Concat non prop data with prop data
  	data = pd.concat([non_prop_data, data_with_confidence], ignore_index=True)
	}
	
	# peptide_id = -1 indicates h0, i.e. the peak group is wrong!
	# initialize priors
	data.loc[data.peptide_id != -1, 'prior'] = (1-data.loc[data.peptide_id != -1, 'precursor_peakgroup_pep']) / data.loc[data.peptide_id != -1, 'num_peptidoforms'] # potential peptidoforms
	data.loc[data.peptide_id == -1, 'prior'] = data.loc[data.peptide_id == -1, 'precursor_peakgroup_pep'] # h0
	
	# set evidence
	data.loc[data.bmask == 1, 'evidence'] = (1-data.loc[data.bmask == 1, 'pep']) # we have evidence FOR this peptidoform or h0
	data.loc[data.bmask == 0, 'evidence'] = data.loc[data.bmask == 0, 'pep'] # we have evidence AGAINST this peptidoform or h0
	
	data = data[['feature_id', 'across_run_feature_id', 'num_peptidoforms','prior','evidence','multi_peptide_id','peptide_id']]
	data = data.rename(columns=lambda x: x.replace('multi_peptide_id','hypothesis'))
	
	return data


def apply_bm(data):
	
	# tmp = data.loc[data.feature_id==-3487633823378371770]
	
	# compute likelihood * prior per feature & hypothesis
	# all priors are identical but pandas DF multiplication requires aggregation, so we use min()
	pp_data = (data.groupby(['feature_id', "hypothesis"])["evidence"].prod() * data.groupby(['feature_id',"hypothesis"])["prior"].min()).reset_index()
	pp_data.columns = ['feature_id','hypothesis','likelihood_prior']
	
	# compute likelihood sum per feature
	pp_data['likelihood_sum'] = pp_data.groupby('feature_id')['likelihood_prior'].transform(np.sum)
	
	# compute posterior hypothesis probability
	pp_data['posterior'] = pp_data['likelihood_prior'] / pp_data['likelihood_sum']
	
	return pp_data.fillna(value = 0)


def precursor_inference(data, ipf_ms1_scoring, ipf_ms2_scoring, ipf_max_precursor_pep, ipf_max_precursor_peakgroup_pep):
	# prepare MS1-level precursor data
	if ipf_ms1_scoring:
		ms1_precursor_data = data[data['ms1_precursor_pep'] < ipf_max_precursor_pep][['feature_id','ms1_precursor_pep']].drop_duplicates()
	else:
		ms1_precursor_data = data[['feature_id']].drop_duplicates()
		ms1_precursor_data['ms1_precursor_pep'] = np.nan
	
	# prepare MS2-level precursor data
	if ipf_ms2_scoring:
		ms2_precursor_data = data[data['ms2_precursor_pep'] < ipf_max_precursor_pep][['feature_id','ms2_precursor_pep']].drop_duplicates()
	else:
		ms2_precursor_data = data[['feature_id']].drop_duplicates()
		ms2_precursor_data['ms2_precursor_pep'] = np.nan
	
	# prepare MS2-level peak group data
	ms2_pg_data = data[['feature_id','ms2_peakgroup_pep']].drop_duplicates()
	
	if ipf_ms1_scoring or ipf_ms2_scoring:
		# merge MS1- & MS2-level precursor and peak group data
		precursor_data = ms2_precursor_data.merge(ms1_precursor_data, on=['feature_id'], how='outer').merge(ms2_pg_data, on=['feature_id'], how='outer')
		
		# prepare precursor-level Bayesian model
		click.echo("Info: Preparing precursor-level data.")
		precursor_data_bm = prepare_precursor_bm(precursor_data)
		
		# compute posterior precursor probability
		click.echo("Info: Conducting precursor-level inference.")
		prec_pp_data = apply_bm(precursor_data_bm)
		prec_pp_data['precursor_peakgroup_pep'] = 1 - prec_pp_data['posterior']
		
		inferred_precursors = prec_pp_data[prec_pp_data['hypothesis']][['feature_id','precursor_peakgroup_pep']]
	else:
		# no precursor-level data on MS1 and/or MS2 should be used; use peak group-level data
		click.echo("Info: Skipping precursor-level inference.")
		inferred_precursors = ms2_pg_data.rename(columns=lambda x: x.replace('ms2_peakgroup_pep', 'precursor_peakgroup_pep'))
	
	inferred_precursors = inferred_precursors[(inferred_precursors['precursor_peakgroup_pep'] < ipf_max_precursor_peakgroup_pep)]
	
	return inferred_precursors


def peptidoform_inference(transition_table, precursor_data, ipf_grouped_fdr, propagate_signal_across_runs, across_run_confidence_threshold):
	transition_table = pd.merge(transition_table, precursor_data, on='feature_id')
	# transition_table.loc[ transition_table.feature_id==1922353247390079399 ]
	# compute transition posterior probabilities
	click.echo("Info: Preparing peptidoform-level data.")
	transition_data_bm = prepare_transition_bm(transition_table, propagate_signal_across_runs, across_run_confidence_threshold)
	transition_data_bm = transition_data_bm.reset_index(drop=True)
	# transition_data_bm.rename(columns={'feature_id': 'feature_id_original', 'across_run_feature_id': 'feature_id'}, inplace=True)
	# transition_data_bm.loc[ np.logical_and(transition_data_bm.feature_id==1922353247390079399, transition_data_bm.peptide_id!=-1) ]
	# compute posterior peptidoform probability
	click.echo("Info: Conducting peptidoform-level inference.")
	pf_pp_data = apply_bm(transition_data_bm.reset_index(drop=True))
	pf_pp_data['pep'] = 1 - pf_pp_data['posterior']
	# compute model-based FDR
	if ipf_grouped_fdr:
		pf_pp_data['qvalue'] = pd.merge(pf_pp_data, transition_data_bm[['feature_id', 'num_peptidoforms']].drop_duplicates(), on=['feature_id'], how='inner').groupby('num_peptidoforms')['pep'].transform(compute_model_fdr)
	else:
		pf_pp_data['qvalue'] = compute_model_fdr(pf_pp_data['pep'])
	# merge precursor-level data with UIS data
	result = pf_pp_data.merge(precursor_data[['feature_id','precursor_peakgroup_pep']].drop_duplicates(), on=['feature_id'], how='inner')
	# merge peptide identifiers for multi-mode
	peptide_ids = transition_table[['feature_id', 'multi_peptide_id','peptide_id']].drop_duplicates()
	peptide_ids.columns = ['feature_id', 'hypothesis','peptide_id']
	result = result.merge(peptide_ids, on=['feature_id','hypothesis'], how='inner')
	return result

def get_feature_mapping_across_runs(infile):
	click.echo("Info: Reading Feature Across Run Alignment Mapping.")
	# precursors are restricted according to ipf_max_peakgroup_pep to exclude very poor peak groups
	con = sqlite3.connect(infile)
	
	data = pd.read_sql_query('''SELECT * FROM REFERENCE_EXPERIMENT_ALIGNMENT_FEATURE_MAPPING''', con)
	
	data.columns = [col.lower() for col in data.columns]
	con.close()
	
	return data

def generate_new_feature_id_mapping(grouped_feature_ids, total_set_of_feature_ids):
	# np.random.randint(-9223372036854775808, 9223372036854775807, size=1, dtype=np.int64)[0]
	# return {np.int64(random.randint(10000, 1000000000000)):grouped_feature_id for grouped_feature_id in grouped_feature_ids}
	#total_set_of_feature_ids = np.append(total_set_of_feature_ids, [5706483448744010246, 3908523129301711414, 8345796704896399892])
	
	# Generate new ids
	new_feature_ids = {grouped_feature_id:np.random.randint(-9223372036854775808, 9223372036854775807, size=1, dtype=np.int64)[0] for grouped_feature_id in grouped_feature_ids}
	# Check to see if there is any overlap with existing ids
	existing_ids = np.intersect1d(np.fromiter(new_feature_ids.values(), dtype=int), total_set_of_feature_ids)
	# TODO: should I make this a conditional while loop?
	if existing_ids.shape[0] > 0:
		keys_to_alter = [list(new_feature_ids.keys())[list(new_feature_ids.values()).index(existing_id)] for existing_id in existing_ids]
		tmp = dict((k, np.random.randint(-9223372036854775808, 9223372036854775807, size=1, dtype=np.int64)[0]) for k in keys_to_alter)
		new_feature_ids.update(tmp)
	return new_feature_ids

def across_run_feature_merge(df, df2):

	unique_feature_ids = np.unique(df.to_numpy().flatten())

	new_across_run_feature_id = generate_new_feature_id_mapping([0], [0])

	df_sub_index = df2.loc[np.in1d(df2[["across_run_feature_id"]].to_numpy().flatten(), unique_feature_ids), 'across_run_feature_id'].index

	return {new_across_run_feature_id[0]:df_sub_index}

def infer_peptidoforms(infile, outfile, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0, ipf_grouped_fdr, ipf_max_precursor_pep, ipf_max_peakgroup_pep, ipf_max_precursor_peakgroup_pep, ipf_max_transition_pep, ipf_multi, propagate_signal_across_runs, across_run_confidence_threshold):
	click.echo("Info: Starting IPF (Inference of PeptidoForms).")
	click.echo("Info: Assess probabilities of %s coeluting peptidoform(s) generating a peak group." % ipf_multi)
	
	# precursor level
	precursor_table = read_pyp_peakgroup_precursor(infile, ipf_max_peakgroup_pep, ipf_ms1_scoring, ipf_ms2_scoring)
	precursor_data = precursor_inference(precursor_table, ipf_ms1_scoring, ipf_ms2_scoring, ipf_max_precursor_pep, ipf_max_precursor_peakgroup_pep)
	
	# peptidoform level
	peptidoform_table = read_pyp_transition(infile, ipf_max_transition_pep, ipf_h0, ipf_multi)
	
	## prepare for propagating signal across runs for aligned features
	if ( propagate_signal_across_runs ){
  	across_run_feature_map = get_feature_mapping_across_runs(infile)
  	peptidoform_table['across_run_feature_id'] = peptidoform_table['feature_id']
  	
    # Generate new id to group aligned feature groups
  	start = time.time()
  	new_mapping = across_run_feature_map.groupby("reference_feature_id").apply(across_run_feature_merge, peptidoform_table[["across_run_feature_id"]]).tolist()
  	end = time.time()
  	click.echo(f"INFO: Elapsed time for generating feature alignment grouping mapping {end-start} seconds")
  	
  	# Update across run feature id with aligned feature group ids
  	start = time.time()
  	for mapping in new_mapping:
  		peptidoform_table.loc[ list(mapping.values())[0].tolist() , ["across_run_feature_id"]] = list(mapping.keys())[0]
  	end = time.time()
  	click.echo(f"INFO: Elapsed time for update across run feature ids {end-start} seconds")
	}
	
	peptidoform_data = peptidoform_inference(peptidoform_table, precursor_data, ipf_grouped_fdr, propagate_signal_across_runs, across_run_confidence_threshold)
	
	# finalize results and write to table
	click.echo("Info: Storing results.")
	peptidoform_data = peptidoform_data[peptidoform_data['peptide_id']!=-1][['feature_id','hypothesis','peptide_id','precursor_peakgroup_pep','qvalue','pep']]
	peptidoform_data.columns = ['FEATURE_ID','MULTI_PEPTIDE_ID','PEPTIDE_ID','PRECURSOR_PEAKGROUP_PEP','QVALUE','PEP']
	
	if infile != outfile:
		copyfile(infile, outfile)
	
	con = sqlite3.connect(outfile)
	
	peptidoform_data.to_sql("SCORE_IPF", con, index=False, if_exists='replace')
	con.close()
