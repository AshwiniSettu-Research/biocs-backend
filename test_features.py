"""
Unit tests for the MLPT feature extraction pipeline.
Tests physicochemical features, Kolaskar-Tongaonkar scores, normalization,
and aggregated features.

Run with: python3 -m pytest test_features.py -v
"""

import numpy as np
import pytest

from mlpt.features import (
    extract_physicochemical_features,
    normalize_features,
    compute_kt_scores,
    compute_kt_feature_vector,
    extract_aggregated_features,
    PHYSICOCHEMICAL_PROPERTIES,
    PROPERTY_NAMES,
    KT_PROPENSITY,
)
from mlpt.config import MAX_SEQ_LEN, AMINO_ACIDS


# ============================================================
# extract_physicochemical_features
# ============================================================

class TestExtractPhysicochemicalFeatures:
    """Tests for the physicochemical feature extraction function."""

    def test_extract_physicochemical_features_shape(self):
        """Output should be (39, 64) for a normal-length sequence."""
        seq = "FIASNGVKLV"  # 10 residues
        features = extract_physicochemical_features(seq)
        assert features.shape == (39, MAX_SEQ_LEN)
        assert features.shape == (39, 64)

    def test_extract_physicochemical_features_short_sequence(self):
        """Output should still be (39, 64) for a 3-residue sequence, with zero-padding."""
        seq = "ACE"
        features = extract_physicochemical_features(seq)
        assert features.shape == (39, 64)

        # First 3 columns should have non-zero values for some properties
        # (at least hydrophobicity should be non-zero for A, C, E)
        assert np.any(features[:, :3] != 0)

        # Remaining columns (indices 3..63) should be all zeros (padding)
        assert np.all(features[:, 3:] == 0)

    def test_extract_physicochemical_features_long_sequence(self):
        """Output should still be (39, 64) for a 100-char sequence (truncated to 64)."""
        seq = "ACDEFGHIKL" * 10  # 100 residues
        features = extract_physicochemical_features(seq)
        assert features.shape == (39, 64)

        # All 64 positions should have data (no trailing zeros since seq > MAX_SEQ_LEN)
        # Check that the last column has non-zero entries
        assert np.any(features[:, 63] != 0)

    def test_extract_physicochemical_features_all_amino_acids(self):
        """All 20 standard amino acids should produce non-zero features."""
        for aa in AMINO_ACIDS:
            seq = aa * 3  # need at least a few characters
            features = extract_physicochemical_features(seq)
            # The first position should have at least some non-zero property values
            col = features[:, 0]
            assert np.any(col != 0), f"Amino acid {aa} produced all-zero features"


# ============================================================
# compute_kt_scores
# ============================================================

class TestComputeKtScores:
    """Tests for the Kolaskar-Tongaonkar scoring function."""

    def test_compute_kt_scores_basic(self):
        """Should return (scores_array, avg_score, regions_list) for a normal peptide."""
        scores, avg, regions = compute_kt_scores("FIASNGVKLV")

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (MAX_SEQ_LEN,)
        assert isinstance(avg, float)
        assert isinstance(regions, list)

        # Average antigenicity should be positive for a real peptide
        assert avg > 0.0

        # The first 10 positions should have non-zero scores
        assert np.all(scores[:10] > 0)

        # Positions beyond the sequence should be zero
        assert np.all(scores[10:] == 0)

    def test_compute_kt_scores_short_sequence(self):
        """Should work for a 3-residue sequence."""
        scores, avg, regions = compute_kt_scores("FIA")

        assert scores.shape == (MAX_SEQ_LEN,)
        assert avg > 0.0
        # First 3 positions should have scores
        assert np.all(scores[:3] > 0)

    def test_compute_kt_scores_antigenic_regions(self):
        """Antigenic regions should be list of (start, end) pairs with valid indices."""
        # Use a longer sequence to increase chance of getting regions
        seq = "FIASNGVKLVACDEFGHIKL"
        scores, avg, regions = compute_kt_scores(seq)

        assert isinstance(regions, list)
        for region in regions:
            assert len(region) == 2
            start, end = region
            assert isinstance(start, (int, np.integer))
            assert isinstance(end, (int, np.integer))
            assert 0 <= start <= end < len(seq)


# ============================================================
# compute_kt_feature_vector
# ============================================================

class TestComputeKtFeatureVector:
    """Tests for the weighted K-T feature vector function."""

    def test_compute_kt_feature_vector_shape(self):
        """Output shape should be (64,) = MAX_SEQ_LEN."""
        vec = compute_kt_feature_vector("FIASNGVKLV")
        assert vec.shape == (MAX_SEQ_LEN,)
        assert vec.shape == (64,)

    def test_compute_kt_feature_vector_default_uses_propensity(self):
        """Without weights, the scores should match KT_PROPENSITY values."""
        seq = "FIA"
        vec = compute_kt_feature_vector(seq, kt_weights=None)

        # First three positions should match raw KT_PROPENSITY
        assert abs(vec[0] - KT_PROPENSITY["F"]) < 1e-5
        assert abs(vec[1] - KT_PROPENSITY["I"]) < 1e-5
        assert abs(vec[2] - KT_PROPENSITY["A"]) < 1e-5

        # Padded positions should be zero
        assert vec[3] == 0.0

    def test_compute_kt_feature_vector_with_weights(self):
        """Providing weights should produce a different output than default."""
        seq = "FIASNGVKLV"
        vec_default = compute_kt_feature_vector(seq, kt_weights=None)

        # Create non-trivial weights (39 weights, one per property)
        weights = np.ones(39, dtype=np.float32) * 0.1
        vec_weighted = compute_kt_feature_vector(seq, kt_weights=weights)

        assert vec_weighted.shape == (MAX_SEQ_LEN,)
        # The weighted vector should differ from the default
        assert not np.allclose(vec_default[:10], vec_weighted[:10])


# ============================================================
# normalize_features
# ============================================================

class TestNormalizeFeatures:
    """Tests for the batch normalization function."""

    def test_normalize_features_zero_variance(self):
        """Should handle zero-variance columns without producing NaN."""
        # Create a batch where one feature has constant values
        batch = np.zeros((5, 39, 64), dtype=np.float32)
        # Set one feature to a constant non-zero value
        batch[:, 0, :] = 3.0
        # Set another feature to varying values
        batch[:, 1, :10] = np.arange(10, dtype=np.float32)

        normalized, means, stds = normalize_features(batch)

        # There should be no NaN values in the output
        assert not np.any(np.isnan(normalized))
        # The zero-variance feature should be normalized to 0 (since (x - mean)/1 = 0 when all x = mean)
        assert np.allclose(normalized[:, 0, :], 0.0)

    def test_normalize_features_preserves_shape(self):
        """Output shape should match input shape."""
        batch = np.random.randn(10, 39, 64).astype(np.float32)
        normalized, means, stds = normalize_features(batch)

        assert normalized.shape == batch.shape
        assert means.shape == (39,)
        assert stds.shape == (39,)


# ============================================================
# extract_aggregated_features
# ============================================================

class TestExtractAggregatedFeatures:
    """Tests for the sequence-level aggregated feature extraction."""

    def test_extract_aggregated_features_shape(self):
        """Output should be (156,) = 4 * 39 (mean, std, min, max per property)."""
        seq = "FIASNGVKLV"
        agg = extract_aggregated_features(seq)
        assert agg.shape == (156,)

    def test_extract_aggregated_features_values(self):
        """Verify mean/std/min/max are correct for a known single-AA sequence."""
        # For a single amino acid repeated, std should be 0 and mean = min = max
        seq = "AAA"
        agg = extract_aggregated_features(seq)

        # First 39 values are means, next 39 are stds, next 39 are mins, next 39 are maxs
        means = agg[:39]
        stds = agg[39:78]
        mins = agg[78:117]
        maxs = agg[117:156]

        # For a uniform sequence "AAA", std should be 0 (or negligible float32 noise)
        assert np.allclose(stds, 0.0, atol=1e-6)

        # Mean, min, and max should all be equal for a uniform sequence
        assert np.allclose(means, mins, atol=1e-6)
        assert np.allclose(means, maxs, atol=1e-6)

        # Verify one known property value: hydrophobicity of A = 1.8
        hydro_idx = PROPERTY_NAMES.index("hydrophobicity")
        assert abs(means[hydro_idx] - 1.8) < 1e-5

    def test_extract_aggregated_features_mixed_sequence(self):
        """For a mixed sequence, std should be > 0 for properties with varied values."""
        seq = "ACDEFGHIKLMNPQRSTVWY"  # all 20 amino acids
        agg = extract_aggregated_features(seq)

        means = agg[:39]
        stds = agg[39:78]
        mins = agg[78:117]
        maxs = agg[117:156]

        # With all different amino acids, many properties should have non-zero std
        # Hydrophobicity varies a lot across amino acids
        hydro_idx = PROPERTY_NAMES.index("hydrophobicity")
        assert stds[hydro_idx] > 0.0

        # Min should be <= mean and max should be >= mean
        for i in range(39):
            assert mins[i] <= means[i] + 1e-6
            assert maxs[i] >= means[i] - 1e-6


# ============================================================
# Property dictionary integrity
# ============================================================

class TestPropertyDictionaryIntegrity:
    """Tests to verify correctness of the physicochemical property dictionaries."""

    def test_no_duplicate_properties(self):
        """All 39 property dictionaries should have distinct values (not copied)."""
        # Extract value tuples for each property (sorted by amino acid)
        sorted_aas = sorted(AMINO_ACIDS)
        value_signatures = []

        for prop_name in PROPERTY_NAMES:
            prop_dict = PHYSICOCHEMICAL_PROPERTIES[prop_name]
            signature = tuple(prop_dict[aa] for aa in sorted_aas)
            value_signatures.append(signature)

        # Every signature should be unique
        unique_sigs = set(value_signatures)
        assert len(unique_sigs) == 39, (
            f"Found {len(unique_sigs)} unique property signatures out of 39. "
            "Some properties may have duplicated value dictionaries."
        )

    def test_all_properties_have_20_amino_acids(self):
        """Every property dictionary should have exactly the 20 standard amino acids."""
        expected_aas = set(AMINO_ACIDS)
        for prop_name in PROPERTY_NAMES:
            prop_dict = PHYSICOCHEMICAL_PROPERTIES[prop_name]
            actual_aas = set(prop_dict.keys())
            assert actual_aas == expected_aas, (
                f"Property '{prop_name}' has keys {actual_aas} "
                f"but expected {expected_aas}"
            )

    def test_kt_propensity_has_20_amino_acids(self):
        """KT_PROPENSITY should have all 20 standard amino acids."""
        expected_aas = set(AMINO_ACIDS)
        actual_aas = set(KT_PROPENSITY.keys())
        assert actual_aas == expected_aas

    def test_property_count(self):
        """There should be exactly 39 properties defined."""
        assert len(PROPERTY_NAMES) == 39
        assert len(PHYSICOCHEMICAL_PROPERTIES) == 39
