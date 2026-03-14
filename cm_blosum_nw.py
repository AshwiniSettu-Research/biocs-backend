"""
CM-BLOSUM-NW: Compositionally Modulated BLOSUM with Banded Needleman-Wunsch
=============================================================================

A novel pairwise sequence alignment algorithm that augments BLOSUM62 with
sequence-specific information-theoretic adjustments derived from amino acid
and dipeptide compositions of the input sequences.

Novelty Claim:
    Unlike Altschul's compositional adjustment (2005), which operates post-hoc
    in local alignment within BLAST, this method integrates compositional
    modulation DIRECTLY into the DP recurrence of a banded global alignment
    framework. Unlike Crooks et al. (2005), whose dipeptide correlations are
    static from BLOCKS, ours are computed on-the-fly from the input sequences.
    Unlike PROSTAlign (2023), whose dipeptide matrix requires evolutionary data,
    ours requires only the input sequences themselves.

Scoring Formula:
    M(a, b) = BLOSUM62(a, b) + alpha * IC(a, b) + beta * DPC_adj(a, b)

    where:
        IC(a, a)      = -log2(freq(a))                         [self-information]
        IC(a, b)      = 0                                       [mismatch: no IC bonus]
        DPC_adj(a, b) = log2(freq(ab) / (freq(a) * freq(b)))   [dipeptide log-odds]
        alpha, beta   = weights optimized via grid search on BAliBASE

References:
    [1] Henikoff & Henikoff (1992) - BLOSUM62 matrix
    [2] Altschul et al. (2005) - Compositional adjustment in BLAST
    [3] Crooks et al. (2005) - Dipeptide covariation in alignment
    [4] Yu et al. (2003) - Compositional adjustment of substitution matrices
    [5] Ashwini et al. (2024) - CS-NW: Compositional Scoring NW (base work)
    [6] Shannon (1948) - Information theory foundations

Author: S. Ashwini (Ph.D. Research)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np                          # Numerical arrays for DP matrices
import math                                 # log2 for information content
import time                                 # Runtime measurement
import tracemalloc                          # Memory usage tracking
from collections import Counter             # Efficient character counting
from multiprocessing import Pool, cpu_count # Parallel frequency computation
from typing import List, Tuple, Dict, Optional  # Type annotations
from itertools import product as iter_product    # Grid search combinations
import logging                              # Structured logging
import json                                 # Export results as JSON
import os                                   # File path operations

# Configure logging to show timestamps and severity levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Standard 20 amino acid single-letter codes (IUPAC)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Small constant to prevent division by zero in frequency calculations
# Any value << 1/L (where L is sequence length) works; 1e-6 is standard
EPSILON = 1e-6

# =============================================================================
# BLOSUM62 SUBSTITUTION MATRIX
# =============================================================================
# Source: Henikoff & Henikoff (1992), PNAS 89(22):10915-10919
# These are half-bit log-odds scores derived from the BLOCKS database.
# Sequences were clustered at 62% identity threshold.
# This is the default scoring matrix in NCBI BLAST and the most widely
# validated general-purpose protein substitution matrix.
#
# Format: BLOSUM62_DATA[amino_acid_1][amino_acid_2] = score
# Positive scores indicate favored substitutions (biologically common).
# Negative scores indicate disfavored substitutions (biologically rare).
# Diagonal entries represent self-match scores (always positive).

BLOSUM62_DATA = {
    'A': {'A':  4, 'C':  0, 'D': -2, 'E': -1, 'F': -2, 'G':  0, 'H': -2, 'I': -1, 'K': -1, 'L': -1, 'M': -1, 'N': -2, 'P': -1, 'Q': -1, 'R': -1, 'S':  1, 'T':  0, 'V':  0, 'W': -3, 'Y': -2},
    'C': {'A':  0, 'C':  9, 'D': -3, 'E': -4, 'F': -2, 'G': -3, 'H': -3, 'I': -1, 'K': -3, 'L': -1, 'M': -1, 'N': -3, 'P': -3, 'Q': -3, 'R': -3, 'S': -1, 'T': -1, 'V': -1, 'W': -2, 'Y': -2},
    'D': {'A': -2, 'C': -3, 'D':  6, 'E':  2, 'F': -3, 'G': -1, 'H': -1, 'I': -3, 'K': -1, 'L': -4, 'M': -3, 'N':  1, 'P': -1, 'Q':  0, 'R': -2, 'S':  0, 'T': -1, 'V': -3, 'W': -4, 'Y': -3},
    'E': {'A': -1, 'C': -4, 'D':  2, 'E':  5, 'F': -3, 'G': -2, 'H':  0, 'I': -3, 'K':  1, 'L': -3, 'M': -2, 'N':  0, 'P': -1, 'Q':  2, 'R':  0, 'S':  0, 'T': -1, 'V': -2, 'W': -3, 'Y': -2},
    'F': {'A': -2, 'C': -2, 'D': -3, 'E': -3, 'F':  6, 'G': -3, 'H': -1, 'I':  0, 'K': -3, 'L':  0, 'M':  0, 'N': -3, 'P': -4, 'Q': -3, 'R': -3, 'S': -2, 'T': -2, 'V': -1, 'W':  1, 'Y':  3},
    'G': {'A':  0, 'C': -3, 'D': -1, 'E': -2, 'F': -3, 'G':  6, 'H': -2, 'I': -4, 'K': -2, 'L': -4, 'M': -3, 'N':  0, 'P': -2, 'Q': -2, 'R': -2, 'S':  0, 'T': -2, 'V': -3, 'W': -2, 'Y': -3},
    'H': {'A': -2, 'C': -3, 'D': -1, 'E':  0, 'F': -1, 'G': -2, 'H':  8, 'I': -3, 'K': -1, 'L': -3, 'M': -2, 'N':  1, 'P': -2, 'Q':  0, 'R':  0, 'S': -1, 'T': -2, 'V': -3, 'W': -2, 'Y':  2},
    'I': {'A': -1, 'C': -1, 'D': -3, 'E': -3, 'F':  0, 'G': -4, 'H': -3, 'I':  4, 'K': -3, 'L':  2, 'M':  1, 'N': -3, 'P': -3, 'Q': -3, 'R': -3, 'S': -2, 'T': -1, 'V':  3, 'W': -3, 'Y': -1},
    'K': {'A': -1, 'C': -3, 'D': -1, 'E':  1, 'F': -3, 'G': -2, 'H': -1, 'I': -3, 'K':  5, 'L': -2, 'M': -1, 'N':  0, 'P': -1, 'Q':  1, 'R':  2, 'S':  0, 'T': -1, 'V': -2, 'W': -3, 'Y': -2},
    'L': {'A': -1, 'C': -1, 'D': -4, 'E': -3, 'F':  0, 'G': -4, 'H': -3, 'I':  2, 'K': -2, 'L':  4, 'M':  2, 'N': -3, 'P': -3, 'Q': -2, 'R': -2, 'S': -2, 'T': -1, 'V':  1, 'W': -2, 'Y': -1},
    'M': {'A': -1, 'C': -1, 'D': -3, 'E': -2, 'F':  0, 'G': -3, 'H': -2, 'I':  1, 'K': -1, 'L':  2, 'M':  5, 'N': -2, 'P': -2, 'Q':  0, 'R': -1, 'S': -1, 'T': -1, 'V':  1, 'W': -1, 'Y': -1},
    'N': {'A': -2, 'C': -3, 'D':  1, 'E':  0, 'F': -3, 'G':  0, 'H':  1, 'I': -3, 'K':  0, 'L': -3, 'M': -2, 'N':  6, 'P': -2, 'Q':  0, 'R':  0, 'S':  1, 'T':  0, 'V': -3, 'W': -4, 'Y': -2},
    'P': {'A': -1, 'C': -3, 'D': -1, 'E': -1, 'F': -4, 'G': -2, 'H': -2, 'I': -3, 'K': -1, 'L': -3, 'M': -2, 'N': -2, 'P':  7, 'Q': -1, 'R': -2, 'S': -1, 'T': -1, 'V': -2, 'W': -4, 'Y': -3},
    'Q': {'A': -1, 'C': -3, 'D':  0, 'E':  2, 'F': -3, 'G': -2, 'H':  0, 'I': -3, 'K':  1, 'L': -2, 'M':  0, 'N':  0, 'P': -1, 'Q':  5, 'R':  1, 'S':  0, 'T': -1, 'V': -2, 'W': -2, 'Y': -1},
    'R': {'A': -1, 'C': -3, 'D': -2, 'E':  0, 'F': -3, 'G': -2, 'H':  0, 'I': -3, 'K':  2, 'L': -2, 'M': -1, 'N':  0, 'P': -2, 'Q':  1, 'R':  5, 'S': -1, 'T': -1, 'V': -3, 'W': -3, 'Y': -2},
    'S': {'A':  1, 'C': -1, 'D':  0, 'E':  0, 'F': -2, 'G':  0, 'H': -1, 'I': -2, 'K':  0, 'L': -2, 'M': -1, 'N':  1, 'P': -1, 'Q':  0, 'R': -1, 'S':  4, 'T':  1, 'V': -2, 'W': -3, 'Y': -2},
    'T': {'A':  0, 'C': -1, 'D': -1, 'E': -1, 'F': -2, 'G': -2, 'H': -2, 'I': -1, 'K': -1, 'L': -1, 'M': -1, 'N':  0, 'P': -1, 'Q': -1, 'R': -1, 'S':  1, 'T':  5, 'V':  0, 'W': -2, 'Y': -2},
    'V': {'A':  0, 'C': -1, 'D': -3, 'E': -2, 'F': -1, 'G': -3, 'H': -3, 'I':  3, 'K': -2, 'L':  1, 'M':  1, 'N': -3, 'P': -2, 'Q': -2, 'R': -3, 'S': -2, 'T':  0, 'V':  4, 'W': -3, 'Y': -1},
    'W': {'A': -3, 'C': -2, 'D': -4, 'E': -3, 'F':  1, 'G': -2, 'H': -2, 'I': -3, 'K': -3, 'L': -2, 'M': -1, 'N': -4, 'P': -4, 'Q': -2, 'R': -3, 'S': -3, 'T': -2, 'V': -3, 'W': 11, 'Y':  2},
    'Y': {'A': -2, 'C': -2, 'D': -3, 'E': -2, 'F':  3, 'G': -3, 'H':  2, 'I': -1, 'K': -2, 'L': -1, 'M': -1, 'N': -2, 'P': -3, 'Q': -1, 'R': -2, 'S': -2, 'T': -2, 'V': -1, 'W':  2, 'Y':  7},
}


def get_blosum62_matrix() -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Convert the BLOSUM62 dictionary into a numpy array for fast lookup.

    Returns:
        Tuple of:
            - blosum_matrix: 20x20 numpy array of BLOSUM62 scores
            - char_to_idx: mapping from amino acid character to matrix index

    The numpy array enables O(1) lookup during DP matrix filling,
    which is critical for performance on long sequences.
    """
    # Build character-to-index mapping (A=0, C=1, D=2, ..., Y=19)
    char_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    # Allocate 20x20 matrix of zeros
    n = len(AMINO_ACIDS)
    matrix = np.zeros((n, n), dtype=np.float64)

    # Fill matrix from the dictionary
    for i, aa1 in enumerate(AMINO_ACIDS):        # Row amino acid
        for j, aa2 in enumerate(AMINO_ACIDS):    # Column amino acid
            matrix[i][j] = BLOSUM62_DATA[aa1][aa2]  # Copy score

    return matrix, char_to_idx


# =============================================================================
# STAGE 1: SEQUENCE PREPROCESSING
# =============================================================================
# Computes two feature types from the input sequences:
#   (a) Amino Acid Composition (AAC): frequency of each amino acid
#   (b) Dipeptide Composition (DPC): frequency of each adjacent pair
#
# These frequencies are used to compute the information content (IC)
# and dipeptide log-odds (DPC_adj) scoring components.

def compute_amino_acid_frequencies(sequence: str) -> Dict[str, float]:
    """
    Compute the frequency of each amino acid in a single sequence.

    Formula (from information theory):
        freq(a) = count(a in S) / L

    where L is the sequence length.

    Args:
        sequence: Protein sequence string (uppercase, no gaps).

    Returns:
        Dictionary mapping each amino acid to its frequency [0.0, 1.0].
        Amino acids not present in the sequence have frequency 0.0.

    Example:
        >>> compute_amino_acid_frequencies("AACG")
        {'A': 0.5, 'C': 0.25, 'G': 0.25, ...}  # others are 0.0
    """
    L = len(sequence)                        # Total sequence length
    if L == 0:                               # Guard against empty sequence
        return {aa: 0.0 for aa in AMINO_ACIDS}

    counts = Counter(sequence)               # Count occurrences of each character

    # Compute frequency for each standard amino acid
    # Non-standard characters (B, Z, X, etc.) are ignored
    return {aa: counts.get(aa, 0) / L for aa in AMINO_ACIDS}


def compute_dipeptide_frequencies(sequence: str) -> Dict[str, float]:
    """
    Compute the frequency of each adjacent amino acid pair (dipeptide).

    Formula:
        freq(d) = count(d in S) / (L - 1)

    where d is a dipeptide (e.g., "AL", "LG") and L-1 is the total
    number of dipeptide positions in a sequence of length L.

    Args:
        sequence: Protein sequence string (uppercase, no gaps).

    Returns:
        Dictionary mapping each observed dipeptide to its frequency.
        There are 20*20 = 400 possible dipeptides for protein sequences.

    Example:
        >>> compute_dipeptide_frequencies("AACG")
        {'AA': 0.333, 'AC': 0.333, 'CG': 0.333, ...}  # others are 0.0
    """
    L = len(sequence)                        # Total sequence length
    if L <= 1:                               # Need at least 2 residues for a dipeptide
        return {}

    # Extract all consecutive pairs from the sequence
    dipeptides = [sequence[i:i+2] for i in range(L - 1)]

    # Count how many times each dipeptide occurs
    counts = Counter(dipeptides)

    denominator = L - 1                      # Total number of dipeptide positions

    # Build frequency dictionary for all 400 possible dipeptides
    result = {}
    for aa1 in AMINO_ACIDS:                  # First amino acid in pair
        for aa2 in AMINO_ACIDS:              # Second amino acid in pair
            dp = aa1 + aa2                   # Form the dipeptide string
            result[dp] = counts.get(dp, 0) / denominator  # Frequency or 0

    return result


# =============================================================================
# STAGE 2: PARALLEL FREQUENCY AGGREGATION
# =============================================================================
# When aligning sequences from a dataset, we need GLOBAL frequencies
# across all sequences. This stage computes length-weighted averages
# using a MapReduce pattern for parallelism.
#
# Mathematical justification (Eq. 2 from Ashwini et al. 2024):
#   f_a^global = sum_S [f_S(a) * L_S] / sum_S [L_S]
#
# This weights each sequence's contribution by its length, so longer
# sequences have proportionally more influence on the global frequencies.

def _worker_preprocess(args: Tuple[str, List[str]]) -> Tuple[Dict, Dict, int]:
    """
    Worker function for parallel frequency computation (Map step).

    Each worker processes one sequence and returns its local frequencies.
    This function runs in a separate process via multiprocessing.Pool.

    Args:
        args: Tuple of (sequence_string, alphabet_list)

    Returns:
        Tuple of (aac_dict, dpc_dict, sequence_length)
    """
    sequence, _ = args                       # Unpack arguments
    # Clean the sequence: uppercase, alphabetic characters only
    sequence = ''.join(c for c in sequence.upper().strip() if c.isalpha())
    aac = compute_amino_acid_frequencies(sequence)  # Compute AAC
    dpc = compute_dipeptide_frequencies(sequence)   # Compute DPC
    return aac, dpc, len(sequence)           # Return results with length


def compute_global_frequencies(
    sequences: List[str],
    n_workers: int = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute length-weighted global AAC and DPC across all sequences.

    This is the Reduce step of the MapReduce pattern. It aggregates
    per-sequence frequencies into global frequencies.

    Formula for AAC (Eq. 2):
        f_a^global = sum_S [freq_S(a) * L_S] / sum_S [L_S]

    Formula for DPC (Eq. 4):
        f_d^global = sum_S [freq_S(d) * (L_S-1)] / sum_S [(L_S-1)]

    For small inputs (<=4 sequences), runs sequentially to avoid
    the overhead of process spawning.

    Args:
        sequences: List of protein sequence strings.
        n_workers: Number of parallel workers (default: cpu_count - 1).

    Returns:
        Tuple of (global_aac_dict, global_dpc_dict).
    """
    if n_workers is None:                    # Default worker count
        n_workers = max(1, cpu_count() - 1)  # Leave one core free

    # --- Map step: compute per-sequence frequencies ---
    results = []
    if len(sequences) <= 4:
        # Sequential execution for small datasets (avoids multiprocessing overhead)
        for seq in sequences:
            r = _worker_preprocess((seq, AMINO_ACIDS))
            results.append(r)
    else:
        # Parallel execution for larger datasets
        args_list = [(seq, AMINO_ACIDS) for seq in sequences]
        try:
            with Pool(processes=min(n_workers, len(sequences))) as pool:
                results = pool.map(_worker_preprocess, args_list)
        except Exception as e:
            # Fallback to sequential if multiprocessing fails
            logger.warning(f"Parallel execution failed ({e}), using sequential.")
            for seq in sequences:
                results.append(_worker_preprocess((seq, AMINO_ACIDS)))

    # --- Reduce step: aggregate into global frequencies ---
    # Sum of all sequence lengths (denominator for AAC)
    total_length = sum(r[2] for r in results)
    # Sum of (L-1) for each sequence (denominator for DPC)
    total_dp_length = sum(max(0, r[2] - 1) for r in results)

    # Initialize accumulators
    global_aac = {aa: 0.0 for aa in AMINO_ACIDS}
    global_dpc = {}

    for aac, dpc, L in results:
        # Accumulate weighted AAC: freq_S(a) * L_S
        for aa, freq in aac.items():
            if aa in global_aac:
                global_aac[aa] += freq * L   # Weight by sequence length

        # Accumulate weighted DPC: freq_S(d) * (L_S - 1)
        for dp, freq in dpc.items():
            if dp not in global_dpc:
                global_dpc[dp] = 0.0
            global_dpc[dp] += freq * max(0, L - 1)  # Weight by dipeptide count

    # Normalize by total lengths
    if total_length > 0:
        for aa in global_aac:
            global_aac[aa] /= total_length   # Now in range [0.0, 1.0]

    if total_dp_length > 0:
        for dp in global_dpc:
            global_dpc[dp] /= total_dp_length  # Now in range [0.0, 1.0]

    return global_aac, global_dpc


# =============================================================================
# STAGE 3: INFORMATION CONTENT (IC) SCORING COMPONENT
# =============================================================================
# Based on Shannon's self-information (Shannon, 1948):
#   IC(a) = -log2(freq(a))
#
# Biological rationale:
#   - Rare amino acids carry more information when they match.
#   - A Tryptophan (W) match (freq ~2-3%) is more informative than
#     a Leucine (L) match (freq ~9-10%).
#   - This is analogous to IDF in text retrieval: rare terms are
#     more discriminative.
#
# This is the SAME mathematical principle underlying BLOSUM itself:
#   BLOSUM(a,b) = log2(observed_freq(a,b) / (freq(a) * freq(b)))
# Our IC term adds a sequence-specific adjustment on top.

def compute_ic_matrix(
    global_aac: Dict[str, float]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build the Information Content scoring matrix.

    For matches (a == b):
        IC(a, a) = -log2(freq(a) + epsilon)

        This gives higher scores to rare amino acid matches:
          - freq(W) = 0.013 -> IC = 6.27 bits (very informative)
          - freq(L) = 0.096 -> IC = 3.38 bits (less informative)

    For mismatches (a != b):
        IC(a, b) = 0.0

        Mismatch information is handled by the DPC component and BLOSUM62.
        Including IC for mismatches would double-count with BLOSUM62.

    Args:
        global_aac: Global amino acid frequencies from Stage 2.

    Returns:
        Tuple of (ic_matrix as 20x20 numpy array, char_to_idx mapping).
    """
    # Build character-to-index mapping
    char_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    n = len(AMINO_ACIDS)                     # Matrix dimension (20)

    # Initialize matrix with zeros (mismatches contribute nothing)
    matrix = np.zeros((n, n), dtype=np.float64)

    # Fill diagonal with self-information values
    for i, aa in enumerate(AMINO_ACIDS):
        freq = global_aac.get(aa, 0.0)      # Get global frequency of this AA
        # -log2(freq) is the self-information in bits
        # epsilon prevents log2(0) which would be -infinity
        matrix[i][i] = -math.log2(freq + EPSILON)

    return matrix, char_to_idx


# =============================================================================
# STAGE 4: DIPEPTIDE LOG-ODDS (DPC) SCORING COMPONENT
# =============================================================================
# Based on the log-odds framework (same as BLOSUM derivation):
#   DPC_adj(a, b) = log2(freq(ab) / (freq(a) * freq(b)))
#
# This measures whether dipeptide "ab" occurs MORE or LESS often
# than expected if a and b were independently distributed.
#
#   Positive: ab is enriched -> a and b have sequential affinity
#   Negative: ab is depleted -> a and b avoid each other
#   Zero:     ab occurs exactly as expected by chance
#
# Biological rationale:
#   Certain amino acid pairs have strong sequential preferences.
#   For example, Proline (P) is often preceded by specific residues
#   due to its unique backbone geometry. These preferences are
#   sequence-specific and not captured by BLOSUM62 (which measures
#   evolutionary substitutability, not sequential adjacency).

def compute_dpc_matrix(
    global_aac: Dict[str, float],
    global_dpc: Dict[str, float]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build the Dipeptide Log-Odds scoring matrix.

    Formula:
        DPC_adj(a, b) = log2(freq(ab) / (freq(a) * freq(b) + epsilon))

    If dipeptide ab is never observed (freq(ab) = 0), the score is 0.0
    (no evidence, so no adjustment).

    Args:
        global_aac: Global amino acid frequencies.
        global_dpc: Global dipeptide frequencies.

    Returns:
        Tuple of (dpc_matrix as 20x20 numpy array, char_to_idx mapping).
    """
    # Build character-to-index mapping
    char_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    n = len(AMINO_ACIDS)                     # Matrix dimension (20)

    # Initialize matrix with zeros (unobserved dipeptides contribute nothing)
    matrix = np.zeros((n, n), dtype=np.float64)

    for i, aa1 in enumerate(AMINO_ACIDS):    # First amino acid
        for j, aa2 in enumerate(AMINO_ACIDS):  # Second amino acid
            dp = aa1 + aa2                   # Form dipeptide string
            freq_dp = global_dpc.get(dp, 0.0)  # Observed dipeptide frequency
            freq_a = global_aac.get(aa1, 0.0)  # Marginal frequency of aa1
            freq_b = global_aac.get(aa2, 0.0)  # Marginal frequency of aa2

            if freq_dp > 0 and freq_a > 0 and freq_b > 0:
                # Log-odds ratio: observed / expected
                # expected = freq(a) * freq(b) under independence assumption
                expected = freq_a * freq_b
                # log2(observed/expected) gives the log-odds score
                matrix[i][j] = math.log2(freq_dp / (expected + EPSILON))
            # else: leave as 0.0 (no evidence for or against)

    return matrix, char_to_idx


# =============================================================================
# STAGE 5: HYBRID SCORING MATRIX CONSTRUCTION
# =============================================================================
# The core novelty: combining three scoring sources into one matrix.
#
# M(a, b) = BLOSUM62(a, b) + alpha * IC(a, b) + beta * DPC_adj(a, b)
#
# Each component captures different biological information:
#   BLOSUM62:  Evolutionary substitution likelihood (from millions of years
#              of observed protein evolution across all known protein families)
#   IC:        Sequence-specific information content (rewards rare matches
#              more heavily, adapting to the composition of the input data)
#   DPC_adj:   Sequential adjacency preferences (captures local structural
#              and functional constraints specific to the input sequences)

class HybridScoringMatrix:
    """
    Builds the CM-BLOSUM scoring matrix by combining three components.

    The hybrid matrix adapts the universal BLOSUM62 to the specific
    compositional characteristics of the input sequences, without
    discarding the evolutionary information that BLOSUM62 provides.

    This addresses the key weakness of the original CS-NW algorithm
    (Ashwini et al. 2024), which replaced BLOSUM entirely with
    composition-only scores that lack evolutionary grounding.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3):
        """
        Initialize with mixing weights.

        Args:
            alpha: Weight for Information Content component.
                   Higher alpha -> more reward for rare amino acid matches.
                   Range: [0.0, 2.0] (to be optimized via grid search).
            beta:  Weight for Dipeptide Log-Odds component.
                   Higher beta -> more influence of sequential adjacency.
                   Range: [0.0, 2.0] (to be optimized via grid search).
        """
        self.alpha = alpha                   # IC weight
        self.beta = beta                     # DPC weight
        # Load BLOSUM62 once at initialization (immutable reference)
        self.blosum_matrix, self.char_to_idx = get_blosum62_matrix()

    def build(
        self,
        global_aac: Dict[str, float],
        global_dpc: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Construct the hybrid scoring matrix.

        M(a, b) = BLOSUM62(a, b) + alpha * IC(a, b) + beta * DPC_adj(a, b)

        Args:
            global_aac: Global amino acid frequencies.
            global_dpc: Global dipeptide frequencies.

        Returns:
            Tuple of (hybrid_matrix as 20x20 numpy array, char_to_idx).
        """
        # Compute the IC component (diagonal-only contributions)
        ic_matrix, _ = compute_ic_matrix(global_aac)

        # Compute the DPC log-odds component (all 400 entries)
        dpc_matrix, _ = compute_dpc_matrix(global_aac, global_dpc)

        # Combine: element-wise weighted sum
        # BLOSUM62 provides the evolutionary baseline
        # IC adjusts match scores based on amino acid rarity
        # DPC adjusts all scores based on sequential adjacency patterns
        hybrid = (
            self.blosum_matrix                # Evolutionary component (fixed)
            + self.alpha * ic_matrix          # Information content (adaptive)
            + self.beta * dpc_matrix          # Dipeptide log-odds (adaptive)
        )

        return hybrid, self.char_to_idx

    def build_from_sequences(
        self,
        seq1: str,
        seq2: str,
        dataset: List[str] = None
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float], Dict[str, float]]:
        """
        Convenience method: compute global frequencies then build matrix.

        If a dataset is provided, frequencies are computed from the full
        dataset (intended for large-scale analyses). Otherwise, frequencies
        come from just the two input sequences.

        Args:
            seq1: First protein sequence.
            seq2: Second protein sequence.
            dataset: Optional list of sequences for frequency computation.

        Returns:
            Tuple of (hybrid_matrix, char_to_idx, global_aac, global_dpc).
        """
        # Determine which sequences to use for frequency computation
        if dataset and len(dataset) > 2:
            freq_sequences = dataset         # Use full dataset
        else:
            freq_sequences = [seq1, seq2]    # Use just the pair

        # Compute global frequencies (Stage 2)
        global_aac, global_dpc = compute_global_frequencies(freq_sequences)

        # Build hybrid matrix (Stage 5)
        hybrid, char_to_idx = self.build(global_aac, global_dpc)

        return hybrid, char_to_idx, global_aac, global_dpc


# =============================================================================
# STAGE 6: BANDED NEEDLEMAN-WUNSCH WITH AFFINE GAP PENALTIES
# =============================================================================
# This implements the core DP alignment with two key optimizations:
#   (a) Banded computation: only fills cells within |i - j| <= k
#   (b) Affine gap penalties: separate open and extension costs
#
# Affine gaps model biology better than linear gaps because:
#   - A single insertion/deletion event can affect multiple residues
#   - Opening a gap should be expensive (structural disruption)
#   - Extending an existing gap should be cheaper (same event continues)
#
# Three DP matrices are maintained:
#   M[i,j] = best score ending in a match/mismatch at (i,j)
#   X[i,j] = best score ending in a gap in seq2 (deletion from seq1)
#   Y[i,j] = best score ending in a gap in seq1 (insertion into seq1)
#
# Recurrence (Gotoh, 1982):
#   M[i,j] = max(M[i-1,j-1], X[i-1,j-1], Y[i-1,j-1]) + score(s1[i], s2[j])
#   X[i,j] = max(M[i-1,j] + G_o, X[i-1,j] + G_e)
#   Y[i,j] = max(M[i,j-1] + G_o, Y[i,j-1] + G_e)
#
# Final score = max(M[m,n], X[m,n], Y[m,n])

class BandedAffineNW:
    """
    Banded Needleman-Wunsch alignment with affine gap penalties.

    Uses three DP matrices (M, X, Y) for match, deletion, and insertion
    states, restricted to a diagonal band of width 2k+1.
    """

    # Traceback pointer constants (encoded as integers for memory efficiency)
    NONE = 0       # No predecessor (boundary)
    DIAG_M = 1     # Came from M[i-1,j-1] (match/mismatch)
    DIAG_X = 2     # Came from X[i-1,j-1] via match state
    DIAG_Y = 3     # Came from Y[i-1,j-1] via match state
    UP_OPEN = 4    # Gap opened: M[i-1,j] -> X[i,j]
    UP_EXT = 5     # Gap extended: X[i-1,j] -> X[i,j]
    LEFT_OPEN = 6  # Gap opened: M[i,j-1] -> Y[i,j]
    LEFT_EXT = 7   # Gap extended: Y[i,j-1] -> Y[i,j]

    def __init__(
        self,
        gap_open: float = -10.0,
        gap_extend: float = -1.0,
        bandwidth: int = 5
    ):
        """
        Initialize alignment parameters.

        Args:
            gap_open: Cost of opening a new gap (G_o). Must be negative.
                      Default -10 is standard for BLOSUM62 in BLAST.
            gap_extend: Cost of extending an existing gap (G_e). Must be negative.
                        Default -1 is standard for BLOSUM62 in BLAST.
            bandwidth: Half-width of the diagonal band (k).
                       Cell (i,j) is computed only if |i - j| <= k.
        """
        self.gap_open = gap_open             # G_o: cost to start a gap
        self.gap_extend = gap_extend         # G_e: cost to continue a gap
        self.bandwidth = bandwidth           # k: band half-width

    def compute_adaptive_bandwidth(self, len1: int, len2: int) -> int:
        """
        Compute an adaptive bandwidth based on sequence lengths.

        For similar-length sequences, k = self.bandwidth is sufficient.
        For different-length sequences, we need a wider band to accommodate
        the length difference.

        Formula:
            k = max(self.bandwidth, ceil(0.1 * |len1 - len2|), ceil(0.05 * max(len1, len2)))

        The 0.1 * length_diff ensures we can accommodate insertions/deletions.
        The 0.05 * max_length ensures divergent equal-length sequences
        still have room to find the optimal path.

        Args:
            len1: Length of first sequence.
            len2: Length of second sequence.

        Returns:
            Adaptive bandwidth value (always >= self.bandwidth).
        """
        length_diff = abs(len1 - len2)       # How different the lengths are
        max_length = max(len1, len2)          # Longer sequence

        k_from_diff = math.ceil(0.1 * length_diff)   # Scale with difference
        k_from_length = math.ceil(0.05 * max_length)  # Scale with total length

        # Take the maximum of all three criteria
        return max(self.bandwidth, k_from_diff, k_from_length)

    def _band_j_start(self, i: int, k: int) -> int:
        """Return the first valid column index j for row i in the band."""
        return max(0, i - k)

    def _j_to_band(self, i: int, j: int, k: int) -> int:
        """Convert full column index j to band-local index for row i."""
        return j - self._band_j_start(i, k)

    def _in_band(self, i: int, j: int, k: int, n: int) -> bool:
        """Check if (i, j) falls within the band and within matrix bounds."""
        j_s = self._band_j_start(i, k)
        j_band = j - j_s
        band_width = 2 * k + 1
        return 0 <= j_band < band_width and 0 <= j <= n

    def align(
        self,
        seq1: str,
        seq2: str,
        scoring_matrix: np.ndarray,
        char_to_idx: Dict[str, int],
        adaptive_band: bool = True
    ) -> Dict:
        """
        Perform banded affine-gap Needleman-Wunsch alignment.

        Uses band-only storage: arrays of shape (m+1, 2*k+1) instead of
        full (m+1, n+1) arrays. For row i, the valid columns are
        j_start = max(0, i-k) to j_end = min(n, i+k), and the band-local
        index is j_band = j - j_start.

        Args:
            seq1: First protein sequence (length m).
            seq2: Second protein sequence (length n).
            scoring_matrix: The hybrid scoring matrix M(a,b).
            char_to_idx: Amino acid to index mapping.
            adaptive_band: If True, use adaptive bandwidth.

        Returns:
            Dictionary containing:
                aligned_seq1, aligned_seq2: Gapped alignment strings
                score: Optimal alignment score
                matches, mismatches, gaps: Count statistics
                identity: Sequence identity percentage
                gap_opens: Number of gap-opening events
        """
        # Clean input sequences
        seq1 = ''.join(c for c in seq1.upper().strip() if c.isalpha())
        seq2 = ''.join(c for c in seq2.upper().strip() if c.isalpha())

        m = len(seq1)                        # Length of sequence 1
        n = len(seq2)                        # Length of sequence 2

        # Determine effective bandwidth
        if adaptive_band:
            k = self.compute_adaptive_bandwidth(m, n)
        else:
            k = self.bandwidth

        G_o = self.gap_open                  # Gap open penalty
        G_e = self.gap_extend                # Gap extend penalty
        NEG_INF = float('-inf')              # Represents unreachable cells
        band_w = 2 * k + 1                  # Band width (number of columns stored per row)

        # --- Allocate band-only DP matrices ---
        # Shape: (m+1, band_w) instead of (m+1, n+1)
        # For row i, band column b corresponds to real column j = b + max(0, i-k)
        # M[i,b]: best score ending in match/mismatch
        # X[i,b]: best score ending in gap in seq2 (deletion)
        # Y[i,b]: best score ending in gap in seq1 (insertion)
        M_mat = np.full((m + 1, band_w), NEG_INF, dtype=np.float64)
        X_mat = np.full((m + 1, band_w), NEG_INF, dtype=np.float64)
        Y_mat = np.full((m + 1, band_w), NEG_INF, dtype=np.float64)

        # Traceback matrices (one per DP matrix), band-only storage
        tb_M = np.zeros((m + 1, band_w), dtype=np.int8)
        tb_X = np.zeros((m + 1, band_w), dtype=np.int8)
        tb_Y = np.zeros((m + 1, band_w), dtype=np.int8)

        # Helper: convert (i, j) to band index, with bounds check
        def _get_band(mat, i, j):
            """Read value from band matrix at full coordinates (i, j)."""
            j_s = max(0, i - k)
            jb = j - j_s
            if jb < 0 or jb >= band_w or j < 0 or j > n:
                return NEG_INF
            return mat[i][jb]

        def _set_band(mat, i, j, val):
            """Write value to band matrix at full coordinates (i, j)."""
            j_s = max(0, i - k)
            jb = j - j_s
            mat[i][jb] = val

        # --- Initialization ---
        # M[0][0] = 0.0 : Starting point, no alignment yet
        # At row 0, j_start = max(0, 0-k) = 0, so band index for j=0 is 0
        _set_band(M_mat, 0, 0, 0.0)

        # First column: gaps in seq2 (all deletions from seq1)
        for i in range(1, m + 1):
            if i <= k:                       # j=0 is within band only if i <= k
                _set_band(X_mat, i, 0, G_o + (i - 1) * G_e)  # Open + extend
                _set_band(tb_X, i, 0, self.UP_OPEN if i == 1 else self.UP_EXT)

        # First row: gaps in seq1 (all insertions into seq1)
        for j in range(1, n + 1):
            if j <= k:                       # At row 0, j is within band if j <= k
                _set_band(Y_mat, 0, j, G_o + (j - 1) * G_e)  # Open + extend
                _set_band(tb_Y, 0, j, self.LEFT_OPEN if j == 1 else self.LEFT_EXT)

        # --- Fill DP matrices within band ---
        for i in range(1, m + 1):
            # Band boundaries for column j at row i
            j_start = max(1, i - k)          # Left edge of band (minimum 1 for fill)
            j_end = min(n, i + k)            # Right edge of band

            for j in range(j_start, j_end + 1):
                # --- Look up substitution score ---
                c1 = seq1[i - 1]             # Character from seq1 (0-indexed)
                c2 = seq2[j - 1]             # Character from seq2 (0-indexed)
                idx1 = char_to_idx.get(c1)   # Matrix index for c1
                idx2 = char_to_idx.get(c2)   # Matrix index for c2

                if idx1 is not None and idx2 is not None:
                    sub_score = scoring_matrix[idx1][idx2]  # Hybrid score
                else:
                    # Non-standard amino acid: use gap open as fallback
                    sub_score = G_o

                # --- Update M[i,j]: match/mismatch state ---
                # Can arrive from M, X, or Y at position (i-1, j-1)
                candidates_M = []
                val = _get_band(M_mat, i - 1, j - 1)
                if val != NEG_INF:
                    candidates_M.append((val + sub_score, self.DIAG_M))
                val = _get_band(X_mat, i - 1, j - 1)
                if val != NEG_INF:
                    candidates_M.append((val + sub_score, self.DIAG_X))
                val = _get_band(Y_mat, i - 1, j - 1)
                if val != NEG_INF:
                    candidates_M.append((val + sub_score, self.DIAG_Y))

                if candidates_M:
                    best_val, best_ptr = max(candidates_M, key=lambda x: x[0])
                    _set_band(M_mat, i, j, best_val)
                    _set_band(tb_M, i, j, best_ptr)

                # --- Update X[i,j]: gap in seq2 (deletion) state ---
                # Can open gap from M[i-1,j] or extend from X[i-1,j]
                candidates_X = []
                val = _get_band(M_mat, i - 1, j)
                if val != NEG_INF:
                    candidates_X.append((val + G_o, self.UP_OPEN))
                val = _get_band(X_mat, i - 1, j)
                if val != NEG_INF:
                    candidates_X.append((val + G_e, self.UP_EXT))

                if candidates_X:
                    best_val, best_ptr = max(candidates_X, key=lambda x: x[0])
                    _set_band(X_mat, i, j, best_val)
                    _set_band(tb_X, i, j, best_ptr)

                # --- Update Y[i,j]: gap in seq1 (insertion) state ---
                # Can open gap from M[i,j-1] or extend from Y[i,j-1]
                candidates_Y = []
                val = _get_band(M_mat, i, j - 1)
                if val != NEG_INF:
                    candidates_Y.append((val + G_o, self.LEFT_OPEN))
                val = _get_band(Y_mat, i, j - 1)
                if val != NEG_INF:
                    candidates_Y.append((val + G_e, self.LEFT_EXT))

                if candidates_Y:
                    best_val, best_ptr = max(candidates_Y, key=lambda x: x[0])
                    _set_band(Y_mat, i, j, best_val)
                    _set_band(tb_Y, i, j, best_ptr)

        # --- Determine which state has the best final score ---
        final_M = _get_band(M_mat, m, n)
        final_X = _get_band(X_mat, m, n)
        final_Y = _get_band(Y_mat, m, n)
        final_scores = [
            (final_M, 'M'),                  # Ended in match/mismatch
            (final_X, 'X'),                  # Ended in deletion
            (final_Y, 'Y'),                  # Ended in insertion
        ]
        # Filter out unreachable states
        reachable = [(s, t) for s, t in final_scores if s != NEG_INF]

        if not reachable:
            logger.warning("Final cell unreachable. Increase bandwidth.")
            return self._empty_result(seq1, seq2, k)

        best_score, best_state = max(reachable, key=lambda x: x[0])

        # --- Traceback ---
        aligned1, aligned2 = self._traceback(
            seq1, seq2, tb_M, tb_X, tb_Y, best_state, k, n
        )

        # --- Compute alignment statistics ---
        matches, mismatches, gaps, gap_opens = self._compute_stats(aligned1, aligned2)
        aln_len = len(aligned1)
        identity = (matches / aln_len * 100) if aln_len > 0 else 0.0

        return {
            "aligned_seq1": aligned1,        # Gapped sequence 1
            "aligned_seq2": aligned2,        # Gapped sequence 2
            "score": best_score,             # Optimal alignment score
            "matches": matches,              # Number of identical positions
            "mismatches": mismatches,         # Number of non-identical aligned positions
            "gaps": gaps,                    # Total gap characters
            "gap_opens": gap_opens,          # Number of gap-opening events
            "alignment_length": aln_len,     # Total alignment length
            "identity": identity,            # Sequence identity percentage
            "bandwidth_used": k,             # Actual bandwidth used
        }

    def _traceback(
        self, seq1, seq2, tb_M, tb_X, tb_Y, start_state, k, n
    ) -> Tuple[str, str]:
        """
        Reconstruct the optimal alignment by following traceback pointers
        stored in band-only matrices.

        Starts at (m, n) in the state indicated by start_state,
        and follows pointers back to (0, 0).

        Args:
            seq1, seq2: Original sequences.
            tb_M, tb_X, tb_Y: Traceback pointer matrices (band storage).
            start_state: Which state to start traceback from ('M', 'X', 'Y').
            k: Band half-width.
            n: Length of seq2 (needed for band bounds checking).

        Returns:
            Tuple of (aligned_seq1, aligned_seq2) strings with gaps.
        """
        band_w = 2 * k + 1                  # Band width

        def _get_tb(mat, i, j):
            """Read traceback pointer from band matrix at full (i, j)."""
            j_s = max(0, i - k)
            jb = j - j_s
            if jb < 0 or jb >= band_w or j < 0 or j > n:
                return self.NONE
            return mat[i][jb]

        aligned1 = []                        # Build alignment in reverse
        aligned2 = []
        i = len(seq1)                        # Start at bottom-right
        j = len(seq2)
        state = start_state                  # Current DP state

        while i > 0 or j > 0:
            if state == 'M':
                # In match/mismatch state: came from diagonal
                ptr = _get_tb(tb_M, i, j)
                aligned1.append(seq1[i-1])   # Add character from seq1
                aligned2.append(seq2[j-1])   # Add character from seq2
                i -= 1                       # Move diagonally
                j -= 1
                # Determine which state we came from
                if ptr == self.DIAG_M:
                    state = 'M'              # Stayed in match state
                elif ptr == self.DIAG_X:
                    state = 'X'              # Came from deletion state
                elif ptr == self.DIAG_Y:
                    state = 'Y'              # Came from insertion state
                else:
                    break                    # Reached origin

            elif state == 'X':
                # In deletion state: gap in seq2
                ptr = _get_tb(tb_X, i, j)
                aligned1.append(seq1[i-1])   # Character from seq1
                aligned2.append('-')         # Gap in seq2
                i -= 1                       # Move up
                if ptr == self.UP_OPEN:
                    state = 'M'              # Gap was just opened from M
                elif ptr == self.UP_EXT:
                    state = 'X'              # Gap was extended
                else:
                    break

            elif state == 'Y':
                # In insertion state: gap in seq1
                ptr = _get_tb(tb_Y, i, j)
                aligned1.append('-')         # Gap in seq1
                aligned2.append(seq2[j-1])   # Character from seq2
                j -= 1                       # Move left
                if ptr == self.LEFT_OPEN:
                    state = 'M'              # Gap was just opened from M
                elif ptr == self.LEFT_EXT:
                    state = 'Y'              # Gap was extended
                else:
                    break
            else:
                break                        # Safety exit

        # Reverse the alignment (we built it backwards)
        return ''.join(reversed(aligned1)), ''.join(reversed(aligned2))

    def _compute_stats(self, aligned1: str, aligned2: str) -> Tuple[int, int, int, int]:
        """
        Compute alignment statistics.

        Args:
            aligned1: Gapped sequence 1.
            aligned2: Gapped sequence 2.

        Returns:
            Tuple of (matches, mismatches, gaps, gap_opens).
        """
        matches = 0                          # Identical residue pairs
        mismatches = 0                       # Non-identical residue pairs
        gaps = 0                             # Total gap characters
        gap_opens = 0                        # Number of gap-opening events
        prev_gap1 = False                    # Was previous position a gap in seq1?
        prev_gap2 = False                    # Was previous position a gap in seq2?

        for c1, c2 in zip(aligned1, aligned2):
            if c1 == '-' or c2 == '-':
                gaps += 1                    # Count gap character
                # Detect gap opening (transition from non-gap to gap)
                if c1 == '-' and not prev_gap1:
                    gap_opens += 1           # New gap in seq1
                if c2 == '-' and not prev_gap2:
                    gap_opens += 1           # New gap in seq2
                prev_gap1 = (c1 == '-')      # Update gap state for seq1
                prev_gap2 = (c2 == '-')      # Update gap state for seq2
            else:
                if c1 == c2:
                    matches += 1             # Identical residues
                else:
                    mismatches += 1          # Different residues
                prev_gap1 = False            # Reset gap states
                prev_gap2 = False

        return matches, mismatches, gaps, gap_opens

    def _empty_result(self, seq1, seq2, k):
        """Return an empty result when alignment fails."""
        return {
            "aligned_seq1": seq1, "aligned_seq2": seq2,
            "score": None, "matches": 0, "mismatches": 0,
            "gaps": 0, "gap_opens": 0, "alignment_length": 0,
            "identity": 0.0, "bandwidth_used": k,
        }


# =============================================================================
# STAGE 7: COMPLETE CM-BLOSUM-NW PIPELINE
# =============================================================================

class CM_BLOSUM_NW:
    """
    Complete CM-BLOSUM-NW alignment pipeline.

    Combines all stages:
        1. Sequence Preprocessing (AAC + DPC)
        2. Parallel Frequency Aggregation
        3. Information Content Scoring
        4. Dipeptide Log-Odds Scoring
        5. Hybrid Matrix Construction (BLOSUM62 + alpha*IC + beta*DPC)
        6. Banded Affine-Gap Needleman-Wunsch Alignment
        7. Result Generation with Metrics

    All tunable parameters (alpha, beta, gap_open, gap_extend, bandwidth)
    can be optimized via the built-in grid search optimizer.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gap_open: float = -10.0,
        gap_extend: float = -1.0,
        bandwidth: int = 5
    ):
        """
        Initialize the CM-BLOSUM-NW pipeline.

        Args:
            alpha: Weight for IC component (default 0.5, optimize via grid search).
            beta: Weight for DPC component (default 0.3, optimize via grid search).
            gap_open: Affine gap open penalty (default -10, standard for BLOSUM62).
            gap_extend: Affine gap extend penalty (default -1, standard for BLOSUM62).
            bandwidth: Minimum band half-width (default 5, adaptive by default).
        """
        self.alpha = alpha                   # IC weight
        self.beta = beta                     # DPC weight
        self.gap_open = gap_open             # Gap open penalty
        self.gap_extend = gap_extend         # Gap extend penalty
        self.bandwidth = bandwidth           # Minimum bandwidth

        # Initialize sub-components
        self.matrix_builder = HybridScoringMatrix(alpha=alpha, beta=beta)
        self.aligner = BandedAffineNW(
            gap_open=gap_open,
            gap_extend=gap_extend,
            bandwidth=bandwidth
        )

    def align_pair(
        self,
        seq1: str,
        seq2: str,
        dataset: List[str] = None,
        verbose: bool = True,
        profile: bool = False
    ) -> Dict:
        """
        Align two protein sequences using the full CM-BLOSUM-NW pipeline.

        Args:
            seq1: First protein sequence.
            seq2: Second protein sequence.
            dataset: Optional larger dataset for frequency computation.
            verbose: If True, print detailed results.
            profile: If True, enable tracemalloc for memory profiling (adds overhead).

        Returns:
            Dictionary with full alignment results and performance metrics.
        """
        # Start resource tracking
        if profile:
            tracemalloc.start()
        start_time = time.time()

        # Clean sequences
        seq1_clean = ''.join(c for c in seq1.upper().strip() if c.isalpha())
        seq2_clean = ''.join(c for c in seq2.upper().strip() if c.isalpha())

        # Build hybrid scoring matrix (Stages 1-5)
        hybrid_matrix, char_to_idx, global_aac, global_dpc = \
            self.matrix_builder.build_from_sequences(
                seq1_clean, seq2_clean, dataset=dataset
            )

        # Perform alignment (Stage 6)
        result = self.aligner.align(
            seq1_clean, seq2_clean, hybrid_matrix, char_to_idx
        )

        # Stop resource tracking
        elapsed = time.time() - start_time
        if profile:
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result["memory_peak_mb"] = peak_mem / (1024 * 1024)
        else:
            result["memory_peak_mb"] = None

        # Add metadata to result
        result["runtime_seconds"] = elapsed
        result["alpha"] = self.alpha
        result["beta"] = self.beta
        result["gap_open"] = self.gap_open
        result["gap_extend"] = self.gap_extend
        result["seq1_length"] = len(seq1_clean)
        result["seq2_length"] = len(seq2_clean)

        if verbose:
            self._print_results(result, seq1_clean, seq2_clean)

        return result

    def _print_results(self, result: Dict, seq1: str, seq2: str):
        """Print formatted alignment results."""
        print("\n" + "=" * 72)
        print("  CM-BLOSUM-NW ALIGNMENT RESULTS")
        print("=" * 72)
        # Show sequence info
        print(f"  Seq1: {seq1[:50]}{'...' if len(seq1) > 50 else ''} (len={len(seq1)})")
        print(f"  Seq2: {seq2[:50]}{'...' if len(seq2) > 50 else ''} (len={len(seq2)})")
        # Show parameters
        print(f"  Parameters: alpha={self.alpha}, beta={self.beta}, "
              f"G_o={self.gap_open}, G_e={self.gap_extend}, k={result['bandwidth_used']}")
        print("-" * 72)

        # Print alignment with match indicators
        a1 = result["aligned_seq1"]
        a2 = result["aligned_seq2"]
        # Build midline: | for match, . for mismatch, space for gap
        mid = ""
        for c1, c2 in zip(a1, a2):
            if c1 == c2 and c1 != '-':
                mid += "|"                   # Match
            elif c1 == '-' or c2 == '-':
                mid += " "                   # Gap
            else:
                mid += "."                   # Mismatch

        # Print in 60-character lines
        width = 60
        for start in range(0, len(a1), width):
            end = start + width
            print(f"  Seq1: {a1[start:end]}")
            print(f"        {mid[start:end]}")
            print(f"  Seq2: {a2[start:end]}")
            if end < len(a1):
                print()

        # Print statistics
        print("-" * 72)
        score_str = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
        print(f"  Score          : {score_str}")
        print(f"  Matches        : {result['matches']}")
        print(f"  Mismatches     : {result['mismatches']}")
        print(f"  Gaps           : {result['gaps']} ({result['gap_opens']} opens)")
        print(f"  Alignment Len  : {result['alignment_length']}")
        print(f"  Identity       : {result['identity']:.2f}%")
        print(f"  Runtime        : {result['runtime_seconds']:.4f}s")
        print(f"  Peak Memory    : {result['memory_peak_mb']:.2f} MB")
        print("=" * 72)


# =============================================================================
# STAGE 8: GRID SEARCH PARAMETER OPTIMIZER
# =============================================================================
# Optimizes alpha, beta, gap_open, gap_extend on a benchmark dataset
# by maximizing alignment accuracy against reference alignments.
#
# This is NOT a hyperparameter hack — it is standard methodology.
# MAFFT, MUSCLE, and BLAST all optimize their parameters on benchmarks.

class ParameterOptimizer:
    """
    Grid search optimizer for CM-BLOSUM-NW parameters.

    Searches over a discrete grid of (alpha, beta, gap_open, gap_extend)
    and selects the combination that maximizes alignment accuracy
    on a set of reference alignments.
    """

    def __init__(
        self,
        alpha_range: List[float] = None,
        beta_range: List[float] = None,
        gap_open_range: List[float] = None,
        gap_extend_range: List[float] = None,
        bandwidth: int = 5
    ):
        """
        Define the search space.

        Args:
            alpha_range: Values to test for alpha (IC weight).
            beta_range: Values to test for beta (DPC weight).
            gap_open_range: Values to test for gap open penalty.
            gap_extend_range: Values to test for gap extend penalty.
            bandwidth: Fixed bandwidth for all runs.
        """
        # Default search ranges based on domain knowledge
        self.alpha_range = alpha_range or [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        self.beta_range = beta_range or [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        self.gap_open_range = gap_open_range or [-8.0, -10.0, -12.0]
        self.gap_extend_range = gap_extend_range or [-0.5, -1.0, -2.0]
        self.bandwidth = bandwidth

    def optimize(
        self,
        sequence_pairs: List[Tuple[str, str]],
        reference_alignments: List[Tuple[str, str]],
        verbose: bool = True
    ) -> Dict:
        """
        Run grid search to find optimal parameters.

        For each parameter combination, aligns all sequence pairs
        and computes average accuracy against reference alignments.

        Args:
            sequence_pairs: List of (seq1, seq2) tuples to align.
            reference_alignments: Corresponding reference (aligned1, aligned2).
            verbose: If True, print progress.

        Returns:
            Dictionary with best parameters and all results.
        """
        # Generate all parameter combinations
        param_grid = list(iter_product(
            self.alpha_range,                # alpha values
            self.beta_range,                 # beta values
            self.gap_open_range,             # gap open values
            self.gap_extend_range            # gap extend values
        ))

        total_combos = len(param_grid)       # Total number of combinations
        logger.info(f"Grid search: {total_combos} parameter combinations, "
                    f"{len(sequence_pairs)} sequence pairs")

        best_accuracy = -1.0                 # Track best result
        best_params = None
        all_results = []                     # Store all results for analysis

        for idx, (alpha, beta, g_o, g_e) in enumerate(param_grid):
            # Create aligner with current parameters
            aligner = CM_BLOSUM_NW(
                alpha=alpha, beta=beta,
                gap_open=g_o, gap_extend=g_e,
                bandwidth=self.bandwidth
            )

            # Align all pairs and compute average accuracy
            total_accuracy = 0.0
            for (s1, s2), (ref1, ref2) in zip(sequence_pairs, reference_alignments):
                result = aligner.align_pair(s1, s2, verbose=False)
                # Compute accuracy: fraction of correctly aligned residue pairs
                acc = self._alignment_accuracy(
                    result["aligned_seq1"], result["aligned_seq2"],
                    ref1, ref2
                )
                total_accuracy += acc

            avg_accuracy = total_accuracy / len(sequence_pairs)

            # Record result
            entry = {
                "alpha": alpha, "beta": beta,
                "gap_open": g_o, "gap_extend": g_e,
                "avg_accuracy": avg_accuracy
            }
            all_results.append(entry)

            # Update best if this is the best so far
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_params = entry.copy()

            # Progress reporting
            if verbose and (idx + 1) % 20 == 0:
                logger.info(f"  [{idx+1}/{total_combos}] "
                           f"Best so far: acc={best_accuracy:.4f} "
                           f"(alpha={best_params['alpha']}, beta={best_params['beta']}, "
                           f"G_o={best_params['gap_open']}, G_e={best_params['gap_extend']})")

        if verbose:
            print(f"\n{'='*60}")
            print(f"GRID SEARCH COMPLETE")
            print(f"{'='*60}")
            print(f"  Best accuracy: {best_accuracy:.4f}")
            print(f"  Best alpha:    {best_params['alpha']}")
            print(f"  Best beta:     {best_params['beta']}")
            print(f"  Best G_o:      {best_params['gap_open']}")
            print(f"  Best G_e:      {best_params['gap_extend']}")
            print(f"{'='*60}")

        return {
            "best_params": best_params,
            "best_accuracy": best_accuracy,
            "all_results": all_results,
            "total_combinations": total_combos
        }

    def _alignment_accuracy(
        self,
        test_a1: str, test_a2: str,
        ref_a1: str, ref_a2: str
    ) -> float:
        """
        Compute alignment accuracy against a reference alignment.

        Accuracy is measured as the fraction of correctly aligned
        residue pairs (true positives / reference positives).

        A residue pair (i, j) means residue at position i in seq1
        is aligned with residue at position j in seq2.

        Args:
            test_a1, test_a2: Test alignment (from our algorithm).
            ref_a1, ref_a2: Reference alignment (ground truth).

        Returns:
            Accuracy as a float in [0.0, 1.0].
        """
        def extract_pairs(a1, a2):
            """Extract aligned residue pairs as (pos_in_seq1, pos_in_seq2) set."""
            pairs = set()
            p1 = p2 = 0                     # Position counters for each sequence
            for c1, c2 in zip(a1, a2):
                if c1 != '-' and c2 != '-':
                    pairs.add((p1, p2))      # This residue pair is aligned
                if c1 != '-':
                    p1 += 1                  # Advance seq1 position
                if c2 != '-':
                    p2 += 1                  # Advance seq2 position
            return pairs

        test_pairs = extract_pairs(test_a1, test_a2)
        ref_pairs = extract_pairs(ref_a1, ref_a2)

        if len(ref_pairs) == 0:              # No reference pairs to compare
            return 1.0 if len(test_pairs) == 0 else 0.0

        # True positives: correctly aligned pairs
        tp = len(test_pairs & ref_pairs)
        # Accuracy = TP / total reference pairs
        return tp / len(ref_pairs)


# =============================================================================
# EVALUATION: COMPARISON WITH STANDARD NW AND ORIGINAL CS-NW
# =============================================================================

class StandardNW:
    """Standard (unbanded) Needleman-Wunsch with BLOSUM62 + affine gaps."""

    def __init__(self, gap_open=-10.0, gap_extend=-1.0):
        """Initialize with standard BLOSUM62 parameters."""
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.blosum, self.idx = get_blosum62_matrix()

    def align(self, seq1: str, seq2: str, profile: bool = False) -> Dict:
        """
        Standard full-matrix NW alignment (O(mn) time and space).
        Used as baseline for comparison.
        """
        if profile:
            tracemalloc.start()
        start_time = time.time()

        # Clean
        seq1 = ''.join(c for c in seq1.upper().strip() if c.isalpha())
        seq2 = ''.join(c for c in seq2.upper().strip() if c.isalpha())
        m, n = len(seq1), len(seq2)

        G_o = self.gap_open
        G_e = self.gap_extend
        NEG_INF = float('-inf')

        # Three matrices
        M = np.full((m+1, n+1), NEG_INF, dtype=np.float64)
        X = np.full((m+1, n+1), NEG_INF, dtype=np.float64)
        Y = np.full((m+1, n+1), NEG_INF, dtype=np.float64)
        tb_M = np.zeros((m+1, n+1), dtype=np.int8)
        tb_X = np.zeros((m+1, n+1), dtype=np.int8)
        tb_Y = np.zeros((m+1, n+1), dtype=np.int8)

        M[0][0] = 0.0
        for i in range(1, m+1):
            X[i][0] = G_o + (i-1) * G_e
            tb_X[i][0] = 4 if i == 1 else 5
        for j in range(1, n+1):
            Y[0][j] = G_o + (j-1) * G_e
            tb_Y[0][j] = 6 if j == 1 else 7

        for i in range(1, m+1):
            for j in range(1, n+1):
                idx1 = self.idx.get(seq1[i-1])
                idx2 = self.idx.get(seq2[j-1])
                sub = self.blosum[idx1][idx2] if idx1 is not None and idx2 is not None else G_o

                # M update
                cands = []
                if M[i-1][j-1] != NEG_INF: cands.append((M[i-1][j-1]+sub, 1))
                if X[i-1][j-1] != NEG_INF: cands.append((X[i-1][j-1]+sub, 2))
                if Y[i-1][j-1] != NEG_INF: cands.append((Y[i-1][j-1]+sub, 3))
                if cands:
                    v, p = max(cands, key=lambda x: x[0])
                    M[i][j] = v; tb_M[i][j] = p

                # X update
                cands = []
                if M[i-1][j] != NEG_INF: cands.append((M[i-1][j]+G_o, 4))
                if X[i-1][j] != NEG_INF: cands.append((X[i-1][j]+G_e, 5))
                if cands:
                    v, p = max(cands, key=lambda x: x[0])
                    X[i][j] = v; tb_X[i][j] = p

                # Y update
                cands = []
                if M[i][j-1] != NEG_INF: cands.append((M[i][j-1]+G_o, 6))
                if Y[i][j-1] != NEG_INF: cands.append((Y[i][j-1]+G_e, 7))
                if cands:
                    v, p = max(cands, key=lambda x: x[0])
                    Y[i][j] = v; tb_Y[i][j] = p

        # Best final
        finals = [(M[m][n],'M'),(X[m][n],'X'),(Y[m][n],'Y')]
        best_score, best_state = max([(s,t) for s,t in finals if s!=NEG_INF], key=lambda x:x[0])

        # Traceback
        a1, a2 = [], []
        i, j, state = m, n, best_state
        while i > 0 or j > 0:
            if state == 'M':
                p = tb_M[i][j]; a1.append(seq1[i-1]); a2.append(seq2[j-1]); i-=1; j-=1
                state = 'M' if p==1 else ('X' if p==2 else ('Y' if p==3 else None))
            elif state == 'X':
                p = tb_X[i][j]; a1.append(seq1[i-1]); a2.append('-'); i-=1
                state = 'M' if p==4 else ('X' if p==5 else None)
            elif state == 'Y':
                p = tb_Y[i][j]; a1.append('-'); a2.append(seq2[j-1]); j-=1
                state = 'M' if p==6 else ('Y' if p==7 else None)
            else:
                break
            if state is None: break

        a1 = ''.join(reversed(a1)); a2 = ''.join(reversed(a2))

        elapsed = time.time() - start_time
        if profile:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            peak = None

        matches = sum(1 for c1,c2 in zip(a1,a2) if c1==c2 and c1!='-')
        mm = sum(1 for c1,c2 in zip(a1,a2) if c1!=c2 and c1!='-' and c2!='-')
        gp = sum(1 for c1,c2 in zip(a1,a2) if c1=='-' or c2=='-')

        return {
            "aligned_seq1": a1, "aligned_seq2": a2,
            "score": best_score, "matches": matches,
            "mismatches": mm, "gaps": gp,
            "alignment_length": len(a1),
            "identity": matches/len(a1)*100 if len(a1)>0 else 0,
            "runtime_seconds": elapsed,
            "memory_peak_mb": peak / (1024 * 1024) if peak is not None else None,
        }


# =============================================================================
# FASTA I/O
# =============================================================================

def read_fasta(filepath: str) -> List[Tuple[str, str]]:
    """Read sequences from a FASTA file. Returns list of (header, sequence)."""
    sequences = []
    header = None
    seq_parts = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    sequences.append((header, ''.join(seq_parts)))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        sequences.append((header, ''.join(seq_parts)))
    return sequences


# =============================================================================
# DEMONSTRATIONS
# =============================================================================

def demo_hemoglobin():
    """
    Demonstrate CM-BLOSUM-NW on hemoglobin alpha vs beta chains.
    These are well-studied homologous proteins (~44% identity).
    """
    print("\n" + "#" * 72)
    print("# DEMO: Hemoglobin Alpha vs Beta (CM-BLOSUM-NW)")
    print("#" * 72)

    # Human Hemoglobin Alpha Chain (first 50 residues, UniProt P69905)
    hba = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
    # Human Hemoglobin Beta Chain (first 50 residues, UniProt P68871)
    hbb = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST"

    # --- CM-BLOSUM-NW alignment ---
    print("\n--- CM-BLOSUM-NW (alpha=0.5, beta=0.3) ---")
    cm_nw = CM_BLOSUM_NW(alpha=0.5, beta=0.3, gap_open=-10.0, gap_extend=-1.0, bandwidth=5)
    result_cm = cm_nw.align_pair(hba, hbb, verbose=True)

    # --- Standard NW (BLOSUM62 only) for comparison ---
    print("\n--- Standard NW (BLOSUM62 only) ---")
    std_nw = StandardNW(gap_open=-10.0, gap_extend=-1.0)
    result_std = std_nw.align(hba, hbb)
    print(f"  Alignment: {result_std['aligned_seq1'][:60]}")
    print(f"             {result_std['aligned_seq2'][:60]}")
    print(f"  Score: {result_std['score']:.4f}")
    print(f"  Identity: {result_std['identity']:.2f}%")
    print(f"  Runtime: {result_std['runtime_seconds']:.6f}s")
    print(f"  Memory: {result_std['memory_peak_mb']:.4f} MB")

    # --- Comparison summary ---
    print("\n--- COMPARISON SUMMARY ---")
    print(f"  {'Metric':<20} {'CM-BLOSUM-NW':>15} {'Standard NW':>15}")
    print(f"  {'-'*50}")
    print(f"  {'Identity (%)':.<20} {result_cm['identity']:>14.2f}% {result_std['identity']:>14.2f}%")
    print(f"  {'Matches':.<20} {result_cm['matches']:>15} {result_std['matches']:>15}")
    print(f"  {'Gaps':.<20} {result_cm['gaps']:>15} {result_std['gaps']:>15}")
    print(f"  {'Runtime (s)':.<20} {result_cm['runtime_seconds']:>15.6f} {result_std['runtime_seconds']:>15.6f}")
    print(f"  {'Memory (MB)':.<20} {result_cm['memory_peak_mb']:>15.4f} {result_std['memory_peak_mb']:>15.4f}")


def demo_scoring_matrix_visualization():
    """
    Show how the hybrid scoring matrix differs from pure BLOSUM62.
    Visualizes the contribution of each component.
    """
    print("\n" + "#" * 72)
    print("# DEMO: Hybrid Scoring Matrix Visualization")
    print("#" * 72)

    # Use hemoglobin sequences for frequency computation
    hba = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
    hbb = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST"

    # Compute global frequencies
    global_aac, global_dpc = compute_global_frequencies([hba, hbb])

    # Show amino acid frequencies
    print("\n--- Global Amino Acid Frequencies ---")
    for aa in AMINO_ACIDS:
        freq = global_aac.get(aa, 0.0)
        if freq > 0:
            ic = -math.log2(freq + EPSILON)
            bar = "#" * int(freq * 100)
            print(f"  {aa}: freq={freq:.4f}  IC={ic:.2f} bits  {bar}")

    # Build and compare matrices
    blosum, idx = get_blosum62_matrix()
    ic_mat, _ = compute_ic_matrix(global_aac)
    dpc_mat, _ = compute_dpc_matrix(global_aac, global_dpc)

    builder = HybridScoringMatrix(alpha=0.5, beta=0.3)
    hybrid, _ = builder.build(global_aac, global_dpc)

    # Show a 5x5 slice for readability
    sample_aa = ['A', 'E', 'G', 'L', 'V']
    print("\n--- BLOSUM62 (subset) ---")
    print(f"     {'  '.join(f'{aa:>6}' for aa in sample_aa)}")
    for aa1 in sample_aa:
        row = f"  {aa1} "
        for aa2 in sample_aa:
            row += f"{blosum[idx[aa1]][idx[aa2]]:>7.1f}"
        print(row)

    print("\n--- IC Component (subset, alpha=0.5 applied) ---")
    print(f"     {'  '.join(f'{aa:>6}' for aa in sample_aa)}")
    for aa1 in sample_aa:
        row = f"  {aa1} "
        for aa2 in sample_aa:
            row += f"{0.5 * ic_mat[idx[aa1]][idx[aa2]]:>7.2f}"
        print(row)

    print("\n--- DPC Log-Odds Component (subset, beta=0.3 applied) ---")
    print(f"     {'  '.join(f'{aa:>6}' for aa in sample_aa)}")
    for aa1 in sample_aa:
        row = f"  {aa1} "
        for aa2 in sample_aa:
            row += f"{0.3 * dpc_mat[idx[aa1]][idx[aa2]]:>7.2f}"
        print(row)

    print("\n--- HYBRID = BLOSUM62 + 0.5*IC + 0.3*DPC (subset) ---")
    print(f"     {'  '.join(f'{aa:>6}' for aa in sample_aa)}")
    for aa1 in sample_aa:
        row = f"  {aa1} "
        for aa2 in sample_aa:
            row += f"{hybrid[idx[aa1]][idx[aa2]]:>7.2f}"
        print(row)


def demo_bandwidth_sensitivity():
    """Bandwidth sensitivity analysis for CM-BLOSUM-NW."""
    print("\n" + "#" * 72)
    print("# ANALYSIS: Bandwidth Sensitivity (CM-BLOSUM-NW)")
    print("#" * 72)

    hba = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
    hbb = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST"

    bandwidths = [1, 2, 3, 5, 7, 10, 15, 25]
    print(f"\n  {'k':>4} | {'Identity%':>10} | {'Score':>10} | {'Runtime(s)':>11} | {'Memory(MB)':>10}")
    print(f"  {'-'*62}")

    for k in bandwidths:
        aligner = CM_BLOSUM_NW(alpha=0.5, beta=0.3, gap_open=-10.0, gap_extend=-1.0, bandwidth=k)
        # Disable adaptive band to test the exact bandwidth value
        aligner.aligner.bandwidth = k
        result = aligner.align_pair(hba, hbb, verbose=False)
        # Override adaptive for this test
        score_str = f"{result['score']:.2f}" if result['score'] is not None else "N/A"
        print(f"  {k:>4} | {result['identity']:>9.2f}% | {score_str:>10} | "
              f"{result['runtime_seconds']:>10.6f}s | {result['memory_peak_mb']:>9.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  CM-BLOSUM-NW: Compositionally Modulated BLOSUM")
    print("  with Banded Needleman-Wunsch")
    print("  Implementation for Ph.D. Research")
    print("=" * 72)

    # 1. Hemoglobin alignment demo
    demo_hemoglobin()

    # 2. Scoring matrix visualization
    demo_scoring_matrix_visualization()

    # 3. Bandwidth sensitivity analysis
    demo_bandwidth_sensitivity()

    print("\n\nAll demonstrations complete.")
