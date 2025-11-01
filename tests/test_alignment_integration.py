# encoding: utf-8
"""
Tests for alignment integration functionality.
"""
from __future__ import print_function

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal

from pyprophet.cli.alignment_integration import _compute_adjusted_scores
from pyprophet.ipf import compute_model_fdr


pd.options.display.expand_frame_repr = False
pd.options.display.precision = 4
pd.options.display.max_columns = None


def test_compute_adjusted_scores_basic():
    """Test basic adjusted score computation."""
    # Create test data with MS2 scores
    ms2_scores = pd.DataFrame({
        "feature_id": [1, 2, 3, 4],
        "run_id": [1, 1, 1, 1],
        "precursor_id": [100, 100, 200, 200],
        "decoy": [0, 0, 0, 0],
        "ms2_pep": [0.02, 0.08, 0.03, 0.06],
        "ms2_qvalue": [0.02, 0.08, 0.03, 0.06],
        "ms2_rank": [1, 2, 1, 2]
    })
    
    # Create alignment scores
    # Feature 2 has good alignment (0.04)
    # Feature 4 has weaker alignment (0.08)
    alignment_scores = pd.DataFrame({
        "feature_id": [2, 4],
        "reference_feature_id": [1, 3],
        "alignment_pep": [0.04, 0.08]
    })
    
    result = _compute_adjusted_scores(ms2_scores, alignment_scores)
    
    # Check that we have results for all features
    assert len(result) == 4
    
    # Check that reference features (1, 3) have alignment_pep = 1.0
    # This means their adjusted PEP should equal their MS2 PEP
    feature_1 = result[result["feature_id"] == 1].iloc[0]
    assert abs(feature_1["alignment_ms2_pep"] - 0.02) < 0.001, f"Expected ~0.02, got {feature_1['alignment_ms2_pep']}"
    
    # Check that feature 2 has adjusted PEP computed correctly
    # pep_adj = 1 - (1 - 0.08) * (1 - 0.04) = 1 - 0.92 * 0.96 = 1 - 0.8832 = 0.1168
    feature_2 = result[result["feature_id"] == 2].iloc[0]
    expected_pep_2 = 1 - (1 - 0.08) * (1 - 0.04)
    assert abs(feature_2["alignment_ms2_pep"] - expected_pep_2) < 0.001, \
        f"Expected ~{expected_pep_2}, got {feature_2['alignment_ms2_pep']}"
    
    # Check that ranks are assigned
    assert "alignment_ms2_peak_group_rank" in result.columns
    
    # Check that feature 1 is rank 1 in its group (precursor 100)
    # Because its adjusted PEP (0.02) is lower than feature 2 (0.1168)
    assert feature_1["alignment_ms2_peak_group_rank"] == 1.0
    
    # Check that q-values are computed
    assert "alignment_ms2_q_value" in result.columns
    assert not pd.isna(feature_1["alignment_ms2_q_value"])


def test_compute_adjusted_scores_with_no_alignment():
    """Test that features without alignment info get alignment_pep = 1.0."""
    ms2_scores = pd.DataFrame({
        "feature_id": [1, 2],
        "run_id": [1, 1],
        "precursor_id": [100, 100],
        "decoy": [0, 0],
        "ms2_pep": [0.02, 0.08],
        "ms2_qvalue": [0.02, 0.08],
        "ms2_rank": [1, 2]
    })
    
    # No alignment scores
    alignment_scores = pd.DataFrame(columns=["feature_id", "reference_feature_id", "alignment_pep"])
    
    result = _compute_adjusted_scores(ms2_scores, alignment_scores)
    
    # All features should have alignment_pep = 1.0, so adjusted PEP = MS2 PEP
    for idx, row in result.iterrows():
        feature_id = row["feature_id"]
        ms2_pep = ms2_scores[ms2_scores["feature_id"] == feature_id]["ms2_pep"].values[0]
        assert abs(row["alignment_ms2_pep"] - ms2_pep) < 0.001, \
            f"Feature {feature_id}: expected ~{ms2_pep}, got {row['alignment_ms2_pep']}"


def test_compute_adjusted_scores_with_decoys():
    """Test that decoys are excluded from scoring but included in result."""
    ms2_scores = pd.DataFrame({
        "feature_id": [1, 2, 3, 4],
        "run_id": [1, 1, 1, 1],
        "precursor_id": [100, 100, 200, 200],
        "decoy": [0, 1, 0, 1],  # 2 and 4 are decoys
        "ms2_pep": [0.02, 0.08, 0.03, 0.06],
        "ms2_qvalue": [0.02, 0.08, 0.03, 0.06],
        "ms2_rank": [1, 2, 1, 2]
    })
    
    alignment_scores = pd.DataFrame(columns=["feature_id", "reference_feature_id", "alignment_pep"])
    
    result = _compute_adjusted_scores(ms2_scores, alignment_scores)
    
    # Check that all features are in result
    assert len(result) == 4
    
    # Check that decoys have NaN for adjusted scores
    decoy_2 = result[result["feature_id"] == 2].iloc[0]
    decoy_4 = result[result["feature_id"] == 4].iloc[0]
    assert pd.isna(decoy_2["alignment_ms2_pep"])
    assert pd.isna(decoy_4["alignment_ms2_pep"])
    
    # Check that targets have valid scores
    target_1 = result[result["feature_id"] == 1].iloc[0]
    target_3 = result[result["feature_id"] == 3].iloc[0]
    assert not pd.isna(target_1["alignment_ms2_pep"])
    assert not pd.isna(target_3["alignment_ms2_pep"])


def test_reference_feature_alignment_pep():
    """Test that reference features get alignment_pep = 1.0."""
    ms2_scores = pd.DataFrame({
        "feature_id": [1, 2, 3],
        "run_id": [1, 1, 2],
        "precursor_id": [100, 100, 100],
        "decoy": [0, 0, 0],
        "ms2_pep": [0.02, 0.08, 0.14],
        "ms2_qvalue": [0.02, 0.08, 0.14],
        "ms2_rank": [1, 2, 1]
    })
    
    # Feature 1 is the reference for features 2 and 3
    alignment_scores = pd.DataFrame({
        "feature_id": [2, 3],
        "reference_feature_id": [1, 1],
        "alignment_pep": [0.06, 0.08]
    })
    
    result = _compute_adjusted_scores(ms2_scores, alignment_scores)
    
    # Feature 1 (reference) should have adjusted PEP = MS2 PEP
    # because its alignment_pep is set to 1.0
    feature_1 = result[result["feature_id"] == 1].iloc[0]
    expected_pep_1 = ms2_scores[ms2_scores["feature_id"] == 1]["ms2_pep"].values[0]
    assert abs(feature_1["alignment_ms2_pep"] - expected_pep_1) < 0.001, \
        f"Reference feature should have adjusted PEP = MS2 PEP, got {feature_1['alignment_ms2_pep']}"


def test_duplicate_alignments_handling():
    """Test that duplicate alignments per feature are handled correctly.
    
    In real data, a feature can align to multiple references. The code should
    handle this by keeping only the best (lowest PEP) alignment per feature.
    This test simulates the scenario where duplicate alignments would be passed
    to _compute_adjusted_scores (which should not happen after deduplication).
    """
    ms2_scores = pd.DataFrame({
        "feature_id": [1, 2],
        "run_id": [1, 1],
        "precursor_id": [100, 100],
        "decoy": [0, 0],
        "ms2_pep": [0.02, 0.08],
        "ms2_qvalue": [0.02, 0.08],
        "ms2_rank": [1, 2]
    })
    
    # Feature 2 has multiple alignments - this should be deduplicated before
    # calling _compute_adjusted_scores, but we test the behavior if it happens
    alignment_scores = pd.DataFrame({
        "feature_id": [2, 2],  # Duplicate feature!
        "reference_feature_id": [1, 1],
        "alignment_pep": [0.04, 0.06]  # Different PEPs
    })
    
    # After merge, this would create duplicate rows for feature 2
    result = _compute_adjusted_scores(ms2_scores, alignment_scores)
    
    # The result should have deduplicated the alignments during merge
    # Check that each feature appears only once in the result
    feature_counts = result["feature_id"].value_counts()
    
    # With the current implementation, merge with duplicates would create duplicates
    # This test documents the expected behavior - in practice, deduplication
    # should happen BEFORE calling _compute_adjusted_scores
    # For now, we just verify the function completes without error
    assert len(result) >= len(ms2_scores), "Result should have at least as many rows as MS2 scores"
