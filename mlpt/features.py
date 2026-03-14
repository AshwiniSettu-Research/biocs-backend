"""
Feature extraction for antigenic peptide prediction.

Level 1: Kolaskar-Tongaonkar antigenicity scoring
Level 2: 39 physicochemical property vectors per amino acid
"""

import numpy as np
from .config import MAX_SEQ_LEN, AMINO_ACIDS

# ============================================================
# 39 Physicochemical Properties per Amino Acid
# Sources: AAIndex, Kyte-Doolittle, Chou-Fasman, Zimmerman,
#          Hopp-Woods, Kidera, etc.
# ============================================================

# fmt: off
PHYSICOCHEMICAL_PROPERTIES = {
    # 1. Hydrophobicity (Kyte-Doolittle scale)
    "hydrophobicity": {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    },
    # 2. Hydrophilicity (Hopp-Woods)
    "hydrophilicity": {
        "A": -0.5, "C": -1.0, "D": 3.0, "E": 3.0, "F": -2.5,
        "G": 0.0, "H": -0.5, "I": -1.8, "K": 3.0, "L": -1.8,
        "M": -1.3, "N": 0.2, "P": 0.0, "Q": 0.2, "R": 3.0,
        "S": 0.3, "T": -0.4, "V": -1.5, "W": -3.4, "Y": -2.3,
    },
    # 3. Molecular Weight
    "molecular_weight": {
        "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
        "G": 75.03, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
        "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
        "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
    },
    # 4. Isoelectric Point (pI)
    "pI": {
        "A": 6.00, "C": 5.07, "D": 2.77, "E": 3.22, "F": 5.48,
        "G": 5.97, "H": 7.59, "I": 6.02, "K": 9.74, "L": 5.98,
        "M": 5.74, "N": 5.41, "P": 6.30, "Q": 5.65, "R": 10.76,
        "S": 5.68, "T": 5.60, "V": 5.96, "W": 5.89, "Y": 5.66,
    },
    # 5. Volume (Zamyatnin)
    "volume": {
        "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
        "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
        "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
        "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
    },
    # 6. Bulkiness (Zimmerman)
    "bulkiness": {
        "A": 11.50, "C": 13.46, "D": 11.68, "E": 13.57, "F": 19.80,
        "G": 3.40, "H": 13.69, "I": 21.40, "K": 15.71, "L": 21.40,
        "M": 16.25, "N": 12.82, "P": 17.43, "Q": 14.45, "R": 14.28,
        "S": 9.47, "T": 15.77, "V": 21.57, "W": 21.67, "Y": 18.03,
    },
    # 7. Flexibility (Vihinen)
    "flexibility": {
        "A": 0.984, "C": 0.906, "D": 1.068, "E": 1.094, "F": 0.915,
        "G": 1.031, "H": 0.950, "I": 0.927, "K": 1.102, "L": 0.935,
        "M": 0.952, "N": 1.048, "P": 1.049, "Q": 1.037, "R": 1.008,
        "S": 1.046, "T": 0.997, "V": 0.931, "W": 0.904, "Y": 0.929,
    },
    # 8. Alpha-helix propensity (Chou-Fasman)
    "alpha_helix": {
        "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13,
        "G": 0.57, "H": 1.00, "I": 1.08, "K": 1.16, "L": 1.21,
        "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11, "R": 0.98,
        "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
    },
    # 9. Beta-sheet propensity (Chou-Fasman)
    "beta_sheet": {
        "A": 0.83, "C": 1.19, "D": 0.54, "E": 0.37, "F": 1.38,
        "G": 0.75, "H": 0.87, "I": 1.60, "K": 0.74, "L": 1.30,
        "M": 1.05, "N": 0.89, "P": 0.55, "Q": 1.10, "R": 0.93,
        "S": 0.75, "T": 1.19, "V": 1.70, "W": 1.37, "Y": 1.47,
    },
    # 10. Turn propensity (Chou-Fasman)
    "turn": {
        "A": 0.66, "C": 1.19, "D": 1.46, "E": 0.74, "F": 0.60,
        "G": 1.56, "H": 0.95, "I": 0.47, "K": 1.01, "L": 0.59,
        "M": 0.60, "N": 1.56, "P": 1.52, "Q": 0.98, "R": 0.95,
        "S": 1.43, "T": 0.96, "V": 0.50, "W": 0.96, "Y": 1.14,
    },
    # 11. Coil propensity
    "coil": {
        "A": 0.82, "C": 0.94, "D": 1.15, "E": 0.85, "F": 0.87,
        "G": 1.27, "H": 1.08, "I": 0.73, "K": 0.96, "L": 0.81,
        "M": 0.82, "N": 1.17, "P": 1.32, "Q": 0.88, "R": 1.03,
        "S": 1.07, "T": 0.98, "V": 0.72, "W": 0.85, "Y": 1.01,
    },
    # 12. Surface accessibility (Emini)
    "surface_accessibility": {
        "A": 0.815, "C": 0.394, "D": 1.263, "E": 1.594, "F": 0.695,
        "G": 0.714, "H": 1.312, "I": 0.603, "K": 2.958, "L": 0.603,
        "M": 0.714, "N": 1.655, "P": 0.714, "Q": 1.655, "R": 2.958,
        "S": 1.263, "T": 1.045, "V": 0.603, "W": 0.695, "Y": 1.045,
    },
    # 13. Polarity (Zimmerman)
    "polarity": {
        "A": 0.00, "C": 1.48, "D": 49.70, "E": 49.90, "F": 0.35,
        "G": 0.00, "H": 51.60, "I": 0.13, "K": 49.50, "L": 0.13,
        "M": 1.43, "N": 3.38, "P": 1.58, "Q": 3.53, "R": 52.00,
        "S": 1.67, "T": 1.66, "V": 0.13, "W": 2.10, "Y": 1.61,
    },
    # 14. Mutability (Jones et al.)
    "mutability": {
        "A": 100, "C": 20, "D": 106, "E": 102, "F": 41,
        "G": 49, "H": 66, "I": 96, "K": 56, "L": 40,
        "M": 94, "N": 134, "P": 56, "Q": 93, "R": 65,
        "S": 120, "T": 97, "V": 74, "W": 18, "Y": 41,
    },
    # 15. pK1 (alpha-COOH)
    "pK1": {
        "A": 2.34, "C": 1.96, "D": 1.88, "E": 2.19, "F": 1.83,
        "G": 2.34, "H": 1.82, "I": 2.36, "K": 2.18, "L": 2.36,
        "M": 2.28, "N": 2.02, "P": 1.99, "Q": 2.17, "R": 2.17,
        "S": 2.21, "T": 2.09, "V": 2.32, "W": 2.38, "Y": 2.20,
    },
    # 16. pK2 (alpha-NH3+)
    "pK2": {
        "A": 9.69, "C": 10.28, "D": 9.60, "E": 9.67, "F": 9.13,
        "G": 9.60, "H": 9.17, "I": 9.60, "K": 8.95, "L": 9.60,
        "M": 9.21, "N": 8.80, "P": 10.60, "Q": 9.13, "R": 9.04,
        "S": 9.15, "T": 9.10, "V": 9.62, "W": 9.39, "Y": 9.11,
    },
    # 17. Net charge at pH 7
    "net_charge": {
        "A": 0.0, "C": 0.0, "D": -1.0, "E": -1.0, "F": 0.0,
        "G": 0.0, "H": 0.1, "I": 0.0, "K": 1.0, "L": 0.0,
        "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0, "R": 1.0,
        "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0, "Y": 0.0,
    },
    # 18. Hydrogen bond donors
    "hbond_donors": {
        "A": 1, "C": 1, "D": 1, "E": 1, "F": 1,
        "G": 1, "H": 2, "I": 1, "K": 2, "L": 1,
        "M": 1, "N": 2, "P": 0, "Q": 2, "R": 4,
        "S": 2, "T": 2, "V": 1, "W": 2, "Y": 2,
    },
    # 19. Hydrogen bond acceptors
    "hbond_acceptors": {
        "A": 2, "C": 2, "D": 4, "E": 4, "F": 2,
        "G": 2, "H": 3, "I": 2, "K": 3, "L": 2,
        "M": 2, "N": 4, "P": 2, "Q": 4, "R": 3,
        "S": 3, "T": 3, "V": 2, "W": 2, "Y": 3,
    },
    # 20. Number of carbon atoms in side chain
    "carbon_atoms": {
        "A": 1, "C": 1, "D": 2, "E": 3, "F": 7,
        "G": 0, "H": 4, "I": 4, "K": 4, "L": 4,
        "M": 3, "N": 2, "P": 3, "Q": 3, "R": 4,
        "S": 1, "T": 2, "V": 3, "W": 9, "Y": 7,
    },
    # 21. Number of hydrogen atoms in side chain
    "hydrogen_atoms": {
        "A": 3, "C": 1, "D": 2, "E": 4, "F": 5,
        "G": 1, "H": 3, "I": 7, "K": 8, "L": 7,
        "M": 5, "N": 2, "P": 5, "Q": 4, "R": 6,
        "S": 1, "T": 3, "V": 5, "W": 6, "Y": 5,
    },
    # 22. Tiny residue indicator (G, A, S)
    "tiny": {
        "A": 1, "C": 0, "D": 0, "E": 0, "F": 0,
        "G": 1, "H": 0, "I": 0, "K": 0, "L": 0,
        "M": 0, "N": 0, "P": 0, "Q": 0, "R": 0,
        "S": 1, "T": 0, "V": 0, "W": 0, "Y": 0,
    },
    # 23. Small residue indicator
    "small": {
        "A": 1, "C": 1, "D": 1, "E": 0, "F": 0,
        "G": 1, "H": 0, "I": 0, "K": 0, "L": 0,
        "M": 0, "N": 1, "P": 1, "Q": 0, "R": 0,
        "S": 1, "T": 1, "V": 1, "W": 0, "Y": 0,
    },
    # 24. Aliphatic indicator (I, L, V)
    "aliphatic": {
        "A": 0, "C": 0, "D": 0, "E": 0, "F": 0,
        "G": 0, "H": 0, "I": 1, "K": 0, "L": 1,
        "M": 0, "N": 0, "P": 0, "Q": 0, "R": 0,
        "S": 0, "T": 0, "V": 1, "W": 0, "Y": 0,
    },
    # 25. Aromatic indicator (F, H, W, Y)
    "aromatic": {
        "A": 0, "C": 0, "D": 0, "E": 0, "F": 1,
        "G": 0, "H": 1, "I": 0, "K": 0, "L": 0,
        "M": 0, "N": 0, "P": 0, "Q": 0, "R": 0,
        "S": 0, "T": 0, "V": 0, "W": 1, "Y": 1,
    },
    # 26. NonPolar indicator
    "nonpolar": {
        "A": 1, "C": 1, "D": 0, "E": 0, "F": 1,
        "G": 1, "H": 0, "I": 1, "K": 0, "L": 1,
        "M": 1, "N": 0, "P": 1, "Q": 0, "R": 0,
        "S": 0, "T": 0, "V": 1, "W": 1, "Y": 0,
    },
    # 27. Polar indicator
    "polar": {
        "A": 0, "C": 0, "D": 1, "E": 1, "F": 0,
        "G": 0, "H": 1, "I": 0, "K": 1, "L": 0,
        "M": 0, "N": 1, "P": 0, "Q": 1, "R": 1,
        "S": 1, "T": 1, "V": 0, "W": 0, "Y": 1,
    },
    # 28. Charged indicator (D, E, H, K, R)
    "charged": {
        "A": 0, "C": 0, "D": 1, "E": 1, "F": 0,
        "G": 0, "H": 1, "I": 0, "K": 1, "L": 0,
        "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1,
        "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0,
    },
    # 29. Polarizability (CHAM820101)
    "polarizability": {
        "A": 0.046, "C": 0.128, "D": 0.105, "E": 0.151, "F": 0.290,
        "G": 0.000, "H": 0.230, "I": 0.186, "K": 0.219, "L": 0.186,
        "M": 0.221, "N": 0.134, "P": 0.131, "Q": 0.180, "R": 0.291,
        "S": 0.062, "T": 0.108, "V": 0.140, "W": 0.409, "Y": 0.298,
    },
    # 30. Alpha-helix frequency (CHOP780201)
    "alpha_freq": {
        "A": 0.77, "C": 0.53, "D": 0.62, "E": 0.90, "F": 0.68,
        "G": 0.39, "H": 0.64, "I": 0.64, "K": 0.70, "L": 0.72,
        "M": 0.80, "N": 0.50, "P": 0.39, "Q": 0.70, "R": 0.60,
        "S": 0.50, "T": 0.55, "V": 0.64, "W": 0.68, "Y": 0.45,
    },
    # 31. Polarity Grantham (GRAR740102)
    "polarity_grantham": {
        "A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3, "F": 5.2,
        "G": 9.0, "H": 10.4, "I": 5.2, "K": 11.3, "L": 4.9,
        "M": 5.7, "N": 11.6, "P": 8.0, "Q": 10.5, "R": 10.5,
        "S": 9.2, "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2,
    },
    # 32. Normalized van der Waals Volume (FASG760101)
    "vdw_volume_norm": {
        "A": 1.00, "C": 2.43, "D": 2.78, "E": 3.78, "F": 5.89,
        "G": 0.00, "H": 4.66, "I": 4.00, "K": 4.77, "L": 4.00,
        "M": 4.43, "N": 2.95, "P": 2.72, "Q": 3.95, "R": 6.13,
        "S": 1.60, "T": 2.60, "V": 3.00, "W": 8.08, "Y": 6.47,
    },
    # 33. Side chain volume (KRIW790101)
    "side_chain_volume": {
        "A": 27.5, "C": 44.6, "D": 40.0, "E": 62.0, "F": 115.5,
        "G": 0.0, "H": 79.0, "I": 93.5, "K": 100.0, "L": 93.5,
        "M": 94.1, "N": 58.7, "P": 41.9, "Q": 80.7, "R": 105.0,
        "S": 29.3, "T": 51.3, "V": 71.5, "W": 145.5, "Y": 117.3,
    },
    # 34. Average flexibility (RACS820108)
    "avg_flexibility": {
        "A": 0.357, "C": 0.346, "D": 0.511, "E": 0.497, "F": 0.314,
        "G": 0.544, "H": 0.323, "I": 0.462, "K": 0.466, "L": 0.365,
        "M": 0.295, "N": 0.463, "P": 0.509, "Q": 0.493, "R": 0.529,
        "S": 0.507, "T": 0.444, "V": 0.386, "W": 0.305, "Y": 0.420,
    },
    # 35. Information value for accessibility (ROSM880101)
    "info_accessibility": {
        "A": -0.02, "C": -0.42, "D": 0.78, "E": 0.83, "F": -0.77,
        "G": 0.49, "H": 0.17, "I": -1.13, "K": 1.40, "L": -1.04,
        "M": -0.40, "N": 0.76, "P": 0.34, "Q": 0.59, "R": 1.38,
        "S": 0.53, "T": 0.21, "V": -0.76, "W": -0.55, "Y": -0.11,
    },
    # 36. Kidera factor 1 (alpha-helix / bend)
    "kidera_f1": {
        "A": -1.56, "C": 0.12, "D": 0.58, "E": -1.45, "F": -0.21,
        "G": 1.46, "H": -0.20, "I": -0.73, "K": -1.34, "L": -1.04,
        "M": -0.46, "N": 1.14, "P": 2.06, "Q": -0.47, "R": 0.22,
        "S": 0.81, "T": 0.26, "V": -0.46, "W": -0.52, "Y": 0.26,
    },
    # 37. Kidera factor 2 (side-chain size)
    "kidera_f2": {
        "A": -1.67, "C": 0.45, "D": -0.87, "E": 0.71, "F": 1.53,
        "G": -1.96, "H": 0.62, "I": -0.16, "K": 0.37, "L": 0.00,
        "M": -0.19, "N": 0.18, "P": -1.18, "Q": -0.24, "R": 1.27,
        "S": -1.03, "T": 0.10, "V": -0.46, "W": 1.15, "Y": 0.83,
    },
    # 38. Kidera factor 3 (extended structure)
    "kidera_f3": {
        "A": -0.97, "C": -1.54, "D": 0.22, "E": -0.49, "F": 1.24,
        "G": -1.64, "H": -0.28, "I": 1.79, "K": -0.14, "L": -0.24,
        "M": 1.20, "N": -1.35, "P": -1.24, "Q": 0.07, "R": 1.37,
        "S": 0.93, "T": -0.43, "V": 1.62, "W": 1.14, "Y": -0.36,
    },
    # 39. Kidera factor 4 (hydrophobicity)
    "kidera_f4": {
        "A": -0.27, "C": -0.30, "D": -1.39, "E": -0.58, "F": -0.96,
        "G": 0.07, "H": -0.30, "I": 0.19, "K": -0.12, "L": 1.21,
        "M": -0.54, "N": 0.20, "P": 0.55, "Q": 1.10, "R": 1.11,
        "S": 0.62, "T": -0.12, "V": -0.21, "W": 0.50, "Y": -0.01,
    },
}
# fmt: on

# Ordered list of property names for consistent indexing
PROPERTY_NAMES = list(PHYSICOCHEMICAL_PROPERTIES.keys())
assert len(PROPERTY_NAMES) == 39, f"Expected 39 properties, got {len(PROPERTY_NAMES)}"

# Kolaskar-Tongaonkar antigenic propensity values
# Based on Kolaskar & Tongaonkar (1990)
KT_PROPENSITY = {
    "A": 1.064, "C": 1.412, "D": 0.866, "E": 0.851, "F": 1.091,
    "G": 0.874, "H": 1.105, "I": 1.152, "K": 0.930, "L": 1.250,
    "M": 0.826, "N": 0.776, "P": 1.064, "Q": 1.015, "R": 0.873,
    "S": 1.012, "T": 0.909, "V": 1.383, "W": 0.893, "Y": 1.161,
}


def extract_physicochemical_features(sequence, max_len=MAX_SEQ_LEN):
    """
    Extract 39 physicochemical features for each position in a peptide sequence.

    Args:
        sequence: amino acid string (uppercase)
        max_len: pad/truncate to this length

    Returns:
        numpy array of shape (39, max_len) - features x positions
    """
    features = np.zeros((39, max_len), dtype=np.float32)
    seq = sequence[:max_len].upper()

    for pos, aa in enumerate(seq):
        if aa in AMINO_ACIDS:
            for feat_idx, prop_name in enumerate(PROPERTY_NAMES):
                features[feat_idx, pos] = PHYSICOCHEMICAL_PROPERTIES[prop_name].get(aa, 0.0)

    return features


def normalize_features(features_batch):
    """
    Z-score normalize features across the batch.

    Args:
        features_batch: numpy array of shape (N, 39, max_len)

    Returns:
        normalized array, means, stds
    """
    # Compute stats per feature across all positions and samples
    means = features_batch.mean(axis=(0, 2), keepdims=True)
    stds = features_batch.std(axis=(0, 2), keepdims=True)
    stds[stds == 0] = 1.0  # avoid division by zero
    normalized = (features_batch - means) / stds
    return normalized, means.squeeze(), stds.squeeze()


def compute_kt_scores(sequence, max_len=MAX_SEQ_LEN, window_size=7):
    """
    Compute Kolaskar-Tongaonkar antigenicity scores for a peptide.

    Args:
        sequence: amino acid string
        max_len: pad to this length
        window_size: sliding window size for averaging

    Returns:
        per_residue_scores: numpy array of shape (max_len,)
        avg_score: average antigenicity score for the peptide
        antigenic_regions: list of (start, end) tuples for antigenic regions
    """
    seq = sequence[:max_len].upper()
    n = len(seq)

    # Raw per-residue propensity scores
    raw_scores = np.zeros(max_len, dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in KT_PROPENSITY:
            raw_scores[i] = KT_PROPENSITY[aa]

    if n == 0:
        return raw_scores, 0.0, []

    # Sliding window average
    smoothed = np.zeros(max_len, dtype=np.float32)
    half_w = window_size // 2
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        smoothed[i] = raw_scores[start:end].mean()

    # Determine antigenic regions (above threshold)
    active_scores = smoothed[:n]
    threshold = active_scores.mean() + active_scores.std()
    avg_score = float(active_scores.mean())

    # Find contiguous antigenic regions
    antigenic_regions = []
    in_region = False
    region_start = 0
    for i in range(n):
        if smoothed[i] >= threshold:
            if not in_region:
                region_start = i
                in_region = True
        else:
            if in_region:
                antigenic_regions.append((region_start, i - 1))
                in_region = False
    if in_region:
        antigenic_regions.append((region_start, n - 1))

    return smoothed, avg_score, antigenic_regions


def compute_kt_feature_vector(sequence, kt_weights=None, max_len=MAX_SEQ_LEN):
    """
    Compute weighted K-T feature vector for a peptide.
    Optionally uses SA-BWK optimized weights.

    Args:
        sequence: amino acid string
        kt_weights: optional weight vector of length 39 from SA-BWK optimization
        max_len: pad to this length

    Returns:
        numpy array of shape (max_len,) - weighted K-T scores per position
    """
    seq = sequence[:max_len].upper()
    n = len(seq)
    scores = np.zeros(max_len, dtype=np.float32)

    if kt_weights is None:
        # Use default K-T propensity
        for i, aa in enumerate(seq):
            if aa in KT_PROPENSITY:
                scores[i] = KT_PROPENSITY[aa]
    else:
        # Weighted combination: sum of (weight_j * feature_j) for each position
        features = extract_physicochemical_features(sequence, max_len)
        for i in range(n):
            scores[i] = np.dot(kt_weights, features[:, i])

    return scores


def extract_aggregated_features(sequence):
    """
    Compute aggregated (sequence-level) statistics from physicochemical features.

    Returns:
        numpy array of shape (156,) - mean, std, min, max of each of 39 features
    """
    features = extract_physicochemical_features(sequence)
    seq_len = min(len(sequence), MAX_SEQ_LEN)

    if seq_len == 0:
        return np.zeros(156, dtype=np.float32)

    active = features[:, :seq_len]
    means = active.mean(axis=1)
    stds = active.std(axis=1)
    mins = active.min(axis=1)
    maxs = active.max(axis=1)

    return np.concatenate([means, stds, mins, maxs]).astype(np.float32)
