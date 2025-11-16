# encoding: utf-8
"""Tests for transfer_confident_evidence_across_runs function."""

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from pyprophet.ipf import transfer_confident_evidence_across_runs


def test_transfer_confident_evidence_basic():
    """Test basic functionality of transfer_confident_evidence_across_runs."""
    # Create test data with 3 features in the same alignment group
    # Feature 0: has confident evidence (pep=0.05)
    # Feature 1: has less confident evidence (pep=0.15)
    # Feature 2: has no confident evidence (pep=0.25)
    
    data = pd.DataFrame({
        'feature_id': [0, 0, 1, 1, 2, 2],
        'transition_id': [10, 11, 10, 11, 10, 11],
        'peptide_id': [100, 100, 100, 100, 100, 100],
        'bmask': [1, 1, 1, 1, 1, 1],
        'num_peptidoforms': [2, 2, 2, 2, 2, 2],
        'alignment_group_id': [1, 1, 1, 1, 1, 1],
        'pep': [0.05, 0.06, 0.15, 0.16, 0.25, 0.26],
        'precursor_peakgroup_pep': [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]
    })
    
    result = transfer_confident_evidence_across_runs(data, 0.1)
    
    # After propagation, all features should have the minimum pep values
    # from confident evidence (threshold = 0.1)
    # Feature 0's confident rows (pep <= 0.1) should propagate to all features
    
    # Check that we have results for all 3 features
    assert result['feature_id'].nunique() == 3
    
    # Each feature should have 2 transitions
    for feature_id in [0, 1, 2]:
        feature_data = result[result['feature_id'] == feature_id]
        assert len(feature_data) == 2
        
        # All features should have at least the confident evidence from feature 0
        # which means they should have pep values <= 0.06
        assert feature_data['pep'].min() <= 0.06


def test_transfer_confident_evidence_no_propagation():
    """Test when no evidence meets the confidence threshold."""
    data = pd.DataFrame({
        'feature_id': [0, 1],
        'transition_id': [10, 10],
        'peptide_id': [100, 100],
        'bmask': [1, 1],
        'num_peptidoforms': [2, 2],
        'alignment_group_id': [1, 1],
        'pep': [0.5, 0.6],
        'precursor_peakgroup_pep': [0.5, 0.6]
    })
    
    result = transfer_confident_evidence_across_runs(data, 0.1)
    
    # No confident evidence to propagate, so each feature should only have its own data
    assert len(result) == 2
    assert set(result['pep'].values) == {0.5, 0.6}


def test_transfer_confident_evidence_multiple_alignment_groups():
    """Test with multiple alignment groups."""
    # Two alignment groups, each with 2 features
    data = pd.DataFrame({
        'feature_id': [0, 1, 2, 3],
        'transition_id': [10, 10, 10, 10],
        'peptide_id': [100, 100, 100, 100],
        'bmask': [1, 1, 1, 1],
        'num_peptidoforms': [2, 2, 2, 2],
        'alignment_group_id': [1, 1, 2, 2],
        'pep': [0.05, 0.15, 0.08, 0.18],
        'precursor_peakgroup_pep': [0.1, 0.2, 0.15, 0.25]
    })
    
    result = transfer_confident_evidence_across_runs(data, 0.1)
    
    # All features should be present
    assert result['feature_id'].nunique() == 4
    
    # Feature 0's confident evidence (0.05) should propagate to feature 1
    feature_1_data = result[result['feature_id'] == 1]
    assert feature_1_data['pep'].min() <= 0.05
    
    # Feature 2's confident evidence (0.08) should propagate to feature 3
    feature_3_data = result[result['feature_id'] == 3]
    assert feature_3_data['pep'].min() <= 0.08


def test_transfer_confident_evidence_preserves_columns():
    """Test that the function preserves all required columns."""
    data = pd.DataFrame({
        'feature_id': [0, 1],
        'transition_id': [10, 10],
        'peptide_id': [100, 100],
        'bmask': [1, 1],
        'num_peptidoforms': [2, 2],
        'alignment_group_id': [1, 1],
        'pep': [0.05, 0.15],
        'precursor_peakgroup_pep': [0.1, 0.2]
    })
    
    result = transfer_confident_evidence_across_runs(data, 0.1)
    
    # Check that all required columns are present
    required_cols = ['feature_id', 'transition_id', 'peptide_id', 'bmask', 
                     'num_peptidoforms', 'alignment_group_id', 'pep', 
                     'precursor_peakgroup_pep']
    assert all(col in result.columns for col in required_cols)


def test_transfer_confident_evidence_min_reduction():
    """Test that the function correctly applies min reduction."""
    # Create data where confident evidence from multiple sources should be reduced to minimum
    data = pd.DataFrame({
        'feature_id': [0, 0, 1, 1, 2, 2],
        'transition_id': [10, 11, 10, 11, 10, 11],
        'peptide_id': [100, 100, 100, 100, 100, 100],
        'bmask': [1, 1, 1, 1, 1, 1],
        'num_peptidoforms': [2, 2, 2, 2, 2, 2],
        'alignment_group_id': [1, 1, 1, 1, 1, 1],
        'pep': [0.05, 0.08, 0.06, 0.09, 0.07, 0.04],  # Note: Feature 2, transition 11 has lowest (0.04)
        'precursor_peakgroup_pep': [0.1, 0.1, 0.1, 0.1, 0.1, 0.05]
    })
    
    result = transfer_confident_evidence_across_runs(data, 0.1)
    
    # For transition 11, all features should get the minimum pep (0.04) and 
    # minimum precursor_peakgroup_pep (0.05)
    for feature_id in [0, 1, 2]:
        feature_data = result[(result['feature_id'] == feature_id) & 
                              (result['transition_id'] == 11)]
        assert len(feature_data) == 1
        assert feature_data['pep'].values[0] == 0.04
        assert feature_data['precursor_peakgroup_pep'].values[0] == 0.05
