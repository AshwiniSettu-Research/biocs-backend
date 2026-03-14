"""
Compositional Scoring Needleman-Wunsch (CS-NW) Algorithm
=========================================================
Implementation based on:
    "Sequence Alignment Using Compositional Scoring-Based Needleman-Wunsch Algorithm"
    S. Ashwini, R. I. Minu, and Jeevan Kumar
    IEEE Access, DOI: 10.1109/ACCESS.2024.0429000

This module implements the six-stage CS-NW pipeline:
    (i)   Sequence Preprocessing (AAC + Dipeptide frequencies)
    (ii)  Parallel Frequency Aggregation (MapReduce pattern)
    (iii) Compositional Scoring Matrix Construction
    (iv)  Banded Needleman-Wunsch Alignment
    (v)   Traceback
    (vi)  Result Generation with Evaluation Metrics

Author: Implementation for Ph.D. research validation
"""

import numpy as np
import time
import tracemalloc
from collections import Counter
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Optional
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Standard 20 amino acid alphabet
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# For nucleotide sequences (used in the paper's example)
NUCLEOTIDES = list("ACGT")

EPSILON = 1e-6  # Prevents division by zero (paper Section V, step iii)


# =============================================================================
# Stage (i): Sequence Preprocessing
# =============================================================================

class SequencePreprocessor:
    """
    Computes amino acid composition (AAC) and dipeptide composition (DPC)
    for a single sequence.

    Reference: Paper Section V, step (i)
        freq(a) = count(a in S) / L                    -- Eq. (1)
        freq(d) = count(d in S) / (L - 1)              -- Eq. (3)
    """

    def __init__(self, alphabet: List[str] = None):
        """
        Args:
            alphabet: The character alphabet. If None, auto-detected
                      from the first sequence processed.
        """
        self.alphabet = alphabet

    def detect_alphabet(self, sequence: str) -> List[str]:
        """Auto-detect whether sequence is protein or nucleotide."""
        unique_chars = set(sequence.upper())
        nucleotide_chars = set("ACGTU")
        if unique_chars.issubset(nucleotide_chars):
            if "U" in unique_chars:
                return list("ACGU")  # RNA
            return list("ACGT")      # DNA
        return AMINO_ACIDS           # Protein

    def compute_aac(self, sequence: str) -> Dict[str, float]:
        """
        Compute Amino Acid Composition (or nucleotide composition).

        freq(a) = count(a in S) / L   -- Eq. (1)

        Args:
            sequence: Input sequence string (uppercase).

        Returns:
            Dictionary mapping each character to its frequency.
        """
        L = len(sequence)
        if L == 0:
            return {}
        counts = Counter(sequence)
        return {char: counts.get(char, 0) / L for char in self.alphabet}

    def compute_dpc(self, sequence: str) -> Dict[str, float]:
        """
        Compute Dipeptide Composition.

        freq(d) = count(d in S) / (L - 1)   -- Eq. (3)

        Args:
            sequence: Input sequence string (uppercase).

        Returns:
            Dictionary mapping each dipeptide to its frequency.
        """
        L = len(sequence)
        if L <= 1:
            return {}
        dipeptides = [sequence[i:i+2] for i in range(L - 1)]
        counts = Counter(dipeptides)
        denominator = L - 1
        # Only return observed dipeptides and all possible ones with 0 default
        result = {}
        for a in self.alphabet:
            for b in self.alphabet:
                dp = a + b
                result[dp] = counts.get(dp, 0) / denominator
        return result

    def preprocess(self, sequence: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Full preprocessing for a single sequence.

        Args:
            sequence: Raw sequence string.

        Returns:
            Tuple of (aac_dict, dpc_dict).
        """
        sequence = sequence.upper().strip()
        # Remove any non-alphabetic characters (gaps, numbers, whitespace)
        sequence = ''.join(c for c in sequence if c.isalpha())

        if self.alphabet is None:
            self.alphabet = self.detect_alphabet(sequence)

        aac = self.compute_aac(sequence)
        dpc = self.compute_dpc(sequence)
        return aac, dpc


# =============================================================================
# Stage (ii): Parallel Frequency Aggregation
# =============================================================================

def _worker_compute_frequencies(args):
    """
    Worker function for parallel frequency computation.
    This is the 'Map' step of the MapReduce pattern.

    Args:
        args: Tuple of (sequence, alphabet)

    Returns:
        Tuple of (aac_dict, dpc_dict, sequence_length)
    """
    sequence, alphabet = args
    preprocessor = SequencePreprocessor(alphabet=alphabet)
    aac, dpc = preprocessor.preprocess(sequence)
    return aac, dpc, len(sequence.upper().strip())


class ParallelFrequencyAggregator:
    """
    Computes global amino acid and dipeptide frequencies across a dataset
    using parallel MapReduce.

    Reference: Paper Section V, step (ii)
        global_freq(x) = (1/N) * sum_{S in D} freq_S(x)   -- simplified

    And the weighted version from the analytical framework:
        f_a^global = sum_S f_S(a) * L_S / sum_S L_S        -- Eq. (2)
        f_d^global = sum_S f_S(d) * (L_S - 1) / sum_S (L_S - 1)  -- Eq. (4)
    """

    def __init__(self, n_workers: int = None):
        """
        Args:
            n_workers: Number of parallel workers. Defaults to cpu_count().
        """
        self.n_workers = n_workers or max(1, cpu_count() - 1)

    def aggregate(self, sequences: List[str], alphabet: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute length-weighted global frequencies across all sequences.

        Uses Eq. (2) and Eq. (4) from the paper's analytical framework,
        which weight each sequence's contribution by its length.

        Args:
            sequences: List of sequence strings.
            alphabet: Character alphabet.

        Returns:
            Tuple of (global_aac, global_dpc).
        """
        if len(sequences) <= 2:
            # For small inputs, no need for multiprocessing overhead
            return self._aggregate_sequential(sequences, alphabet)

        args_list = [(seq, alphabet) for seq in sequences]

        try:
            with Pool(processes=min(self.n_workers, len(sequences))) as pool:
                results = pool.map(_worker_compute_frequencies, args_list)
        except Exception as e:
            logger.warning(f"Parallel execution failed ({e}), falling back to sequential.")
            return self._aggregate_sequential(sequences, alphabet)

        return self._reduce(results, alphabet)

    def _aggregate_sequential(self, sequences: List[str], alphabet: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Sequential fallback for small datasets."""
        results = []
        for seq in sequences:
            preprocessor = SequencePreprocessor(alphabet=alphabet)
            aac, dpc = preprocessor.preprocess(seq)
            clean_seq = ''.join(c for c in seq.upper().strip() if c.isalpha())
            results.append((aac, dpc, len(clean_seq)))
        return self._reduce(results, alphabet)

    def _reduce(self, results, alphabet) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Reduce step: compute length-weighted global frequencies.

        Eq. (2): f_a^global = sum_S [f_S(a) * L_S] / sum_S L_S
        Eq. (4): f_d^global = sum_S [f_S(d) * (L_S - 1)] / sum_S (L_S - 1)
        """
        total_length = sum(r[2] for r in results)
        total_dipeptide_length = sum(max(0, r[2] - 1) for r in results)

        global_aac = {a: 0.0 for a in alphabet}
        # Collect all dipeptide keys
        all_dp_keys = set()
        for _, dpc, _ in results:
            all_dp_keys.update(dpc.keys())
        global_dpc = {dp: 0.0 for dp in all_dp_keys}

        for aac, dpc, L in results:
            # Eq. (2): weighted by sequence length
            for a, freq in aac.items():
                if a in global_aac:
                    global_aac[a] += freq * L

            # Eq. (4): weighted by (L - 1)
            for dp, freq in dpc.items():
                if dp in global_dpc:
                    global_dpc[dp] += freq * max(0, L - 1)

        # Normalize
        if total_length > 0:
            for a in global_aac:
                global_aac[a] /= total_length

        if total_dipeptide_length > 0:
            for dp in global_dpc:
                global_dpc[dp] /= total_dipeptide_length

        return global_aac, global_dpc


# =============================================================================
# Stage (iii): Compositional Scoring Matrix Construction
# =============================================================================

class CompositionalScoringMatrix:
    """
    Constructs the compositional scoring matrix M(a, b).

    Reference: Paper Section V, step (iii) and Eq. (6)

        M(a, b) = 1 / (f_a^global + epsilon)            if a == b  (match)
                = -1 / (f_{ab}^global + epsilon)          if a != b and dipeptide ab observed
                = -gamma                                   otherwise (mismatch)

    where epsilon = 1e-6 and gamma = 1 (default penalty).
    """

    def __init__(self, gamma: float = 1.0, epsilon: float = EPSILON):
        """
        Args:
            gamma: Default mismatch penalty for unobserved dipeptides.
            epsilon: Small constant to prevent division by zero.
        """
        self.gamma = gamma
        self.epsilon = epsilon

    def build(self, global_aac: Dict[str, float], global_dpc: Dict[str, float],
              alphabet: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Build the |Sigma| x |Sigma| scoring matrix.

        NOTE ON FORMULA: The paper's Eq. (6) states M(a,b) = 1/(freq(a)+epsilon)
        for matches. However, cross-checking against the paper's Table 4 reveals
        the actual implemented formula is:

            Match:    M(a,a) =  1 / (1 + global_freq(a))
            Mismatch: M(a,b) = -1 / (1 + global_freq(ab))   if dipeptide ab observed
            Default:  M(a,b) = -gamma                         otherwise

        Verification against Table 4:
            A match: 1/(1 + 3/13) = 1/1.2308 = 0.8125 ≈ 0.813  ✓
            C match: 1/(1 + 4/13) = 1/1.3077 = 0.7647 ≈ 0.765  ✓
            T match: 1/(1 + 2/13) = 1/1.1538 = 0.8667 ≈ 0.867  ✓
            A-C mismatch: -1/(1 + 1/11) = -1/1.0909 = -0.9167 ≈ -0.917  ✓
            G-T mismatch: -1/(1 + 2/11) = -1/1.1818 = -0.8462 ≈ -0.846  ✓

        We implement the formula that matches the paper's numerical results (Table 4).

        Args:
            global_aac: Global amino acid frequencies.
            global_dpc: Global dipeptide frequencies.
            alphabet: Character alphabet.

        Returns:
            Tuple of (scoring_matrix as np.ndarray, char_to_index mapping).
        """
        n = len(alphabet)
        char_to_idx = {c: i for i, c in enumerate(alphabet)}
        matrix = np.full((n, n), -self.gamma, dtype=np.float64)

        for i, a in enumerate(alphabet):
            for j, b in enumerate(alphabet):
                if a == b:
                    # Match: 1 / (1 + global_freq(a))
                    freq_a = global_aac.get(a, 0.0)
                    matrix[i][j] = 1.0 / (1.0 + freq_a)
                else:
                    # Mismatch: check if dipeptide ab is observed
                    dipeptide = a + b
                    freq_ab = global_dpc.get(dipeptide, 0.0)
                    if freq_ab > 0:
                        # Observed dipeptide: -1 / (1 + global_freq(ab))
                        matrix[i][j] = -1.0 / (1.0 + freq_ab)
                    else:
                        # Unobserved: default penalty -gamma
                        matrix[i][j] = -self.gamma

        return matrix, char_to_idx

    def build_from_sequences(self, seq1: str, seq2: str, alphabet: List[str] = None,
                              dataset: List[str] = None) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Convenience method: build scoring matrix from two sequences
        (or a larger dataset if provided).

        When no external dataset is provided, the global frequencies
        are computed from just the two input sequences, as shown in
        the paper's worked example (Section V-B).

        Args:
            seq1: First sequence.
            seq2: Second sequence.
            alphabet: Character alphabet. Auto-detected if None.
            dataset: Optional larger dataset for frequency computation.

        Returns:
            Tuple of (scoring_matrix, char_to_index).
        """
        if alphabet is None:
            preprocessor = SequencePreprocessor()
            alphabet = preprocessor.detect_alphabet(seq1 + seq2)

        sequences = dataset if dataset else [seq1, seq2]
        aggregator = ParallelFrequencyAggregator()
        global_aac, global_dpc = aggregator.aggregate(sequences, alphabet)

        logger.info(f"Global AAC: { {k: round(v, 4) for k, v in global_aac.items() if v > 0} }")
        logger.info(f"Observed dipeptides (freq > 0): {sum(1 for v in global_dpc.values() if v > 0)}")

        return self.build(global_aac, global_dpc, alphabet)


# =============================================================================
# Stage (iv) & (v): Banded Needleman-Wunsch Alignment + Traceback
# =============================================================================

class BandedNeedlemanWunsch:
    """
    Banded Needleman-Wunsch alignment using the compositional scoring matrix.

    Reference: Paper Section V, steps (iv) and (v), Eq. (7) and Eq. (8)

    Initialization:
        S(i, 0) = i * delta_g
        S(0, j) = j * delta_g
        where delta_g = -2 (gap penalty)

    Recurrence (within band |i - j| <= k):
        S(i, j) = max(
            S(i-1, j-1) + M(S1[i], S2[j]),   # match/mismatch
            S(i-1, j) + delta_g,                # deletion
            S(i, j-1) + delta_g                  # insertion
        )
        for max(1, i - k) <= j <= min(N, i + k)

    Traceback from S(M, N) to S(0, 0).
    """

    # Traceback pointer constants
    NONE = 0
    DIAGONAL = 1
    UP = 2
    LEFT = 3

    def __init__(self, gap_penalty: float = -2.0, bandwidth: int = 5):
        """
        Args:
            gap_penalty: Gap penalty delta_g (default: -2 as per paper).
            bandwidth: Band width k (default: 5 as per paper).
        """
        self.gap_penalty = gap_penalty
        self.bandwidth = bandwidth

    def align(self, seq1: str, seq2: str, scoring_matrix: np.ndarray,
              char_to_idx: Dict[str, int]) -> Dict:
        """
        Perform banded NW alignment.

        Args:
            seq1: First sequence (length M).
            seq2: Second sequence (length N).
            scoring_matrix: The compositional scoring matrix M(a,b).
            char_to_idx: Character to index mapping.

        Returns:
            Dictionary with alignment results:
                - aligned_seq1, aligned_seq2: Aligned sequences with gaps
                - score: Optimal alignment score
                - matches, mismatches, gaps: Counts
                - score_matrix: The full DP score matrix (for inspection)
                - identity: Sequence identity percentage
        """
        seq1 = seq1.upper().strip()
        seq2 = seq2.upper().strip()
        seq1 = ''.join(c for c in seq1 if c.isalpha())
        seq2 = ''.join(c for c in seq2 if c.isalpha())

        M = len(seq1)
        N = len(seq2)
        k = self.bandwidth
        delta_g = self.gap_penalty

        # Use -infinity for cells outside the band
        NEG_INF = float('-inf')

        # Initialize score matrix and traceback matrix
        # Using (M+1) x (N+1) matrices
        score = np.full((M + 1, N + 1), NEG_INF, dtype=np.float64)
        traceback = np.zeros((M + 1, N + 1), dtype=np.int8)

        # Initialization -- Eq. (7)
        score[0][0] = 0.0
        for i in range(1, M + 1):
            if i <= k:  # Only within band from (0,0)
                score[i][0] = i * delta_g
                traceback[i][0] = self.UP
        for j in range(1, N + 1):
            if j <= k:  # Only within band from (0,0)
                score[0][j] = j * delta_g
                traceback[0][j] = self.LEFT

        # Fill matrix -- Eq. (8)
        for i in range(1, M + 1):
            j_start = max(1, i - k)
            j_end = min(N, i + k)

            for j in range(j_start, j_end + 1):
                # Get characters (1-indexed in paper, 0-indexed in Python)
                c1 = seq1[i - 1]
                c2 = seq2[j - 1]

                # Look up substitution score from compositional matrix
                idx1 = char_to_idx.get(c1)
                idx2 = char_to_idx.get(c2)

                if idx1 is not None and idx2 is not None:
                    sub_score = scoring_matrix[idx1][idx2]
                else:
                    # Unknown character: use gap penalty as fallback
                    sub_score = delta_g

                # Three possible moves
                candidates = []

                # Diagonal: match/mismatch
                if score[i-1][j-1] != NEG_INF:
                    candidates.append((score[i-1][j-1] + sub_score, self.DIAGONAL))

                # Up: deletion (gap in seq2)
                if score[i-1][j] != NEG_INF:
                    candidates.append((score[i-1][j] + delta_g, self.UP))

                # Left: insertion (gap in seq1)
                if score[i][j-1] != NEG_INF:
                    candidates.append((score[i][j-1] + delta_g, self.LEFT))

                if candidates:
                    best_score, best_ptr = max(candidates, key=lambda x: x[0])
                    score[i][j] = best_score
                    traceback[i][j] = best_ptr

        # Traceback -- Stage (v)
        aligned_seq1, aligned_seq2 = self._traceback(
            seq1, seq2, score, traceback, scoring_matrix, char_to_idx
        )

        # Compute statistics
        matches, mismatches, gaps = self._compute_stats(aligned_seq1, aligned_seq2)
        alignment_length = len(aligned_seq1)
        identity = (matches / alignment_length * 100) if alignment_length > 0 else 0.0

        return {
            "aligned_seq1": aligned_seq1,
            "aligned_seq2": aligned_seq2,
            "score": score[M][N] if score[M][N] != NEG_INF else None,
            "matches": matches,
            "mismatches": mismatches,
            "gaps": gaps,
            "alignment_length": alignment_length,
            "identity": identity,
            "score_matrix": score,
        }

    def _traceback(self, seq1: str, seq2: str, score: np.ndarray,
                   traceback: np.ndarray, scoring_matrix: np.ndarray,
                   char_to_idx: Dict[str, int]) -> Tuple[str, str]:
        """
        Reconstruct optimal alignment from traceback matrix.

        Reference: Paper Section V, step (v)
            - Diagonal: match/mismatch
            - Up: deletion (gap in seq2)
            - Left: insertion (gap in seq1)
        """
        aligned1 = []
        aligned2 = []
        i, j = len(seq1), len(seq2)

        # If the final cell is unreachable (outside band), find the nearest
        # reachable cell on the boundary. This handles edge cases for very
        # different-length sequences with small bandwidth.
        if score[i][j] == float('-inf'):
            logger.warning(
                f"Final cell ({i},{j}) unreachable with bandwidth {self.bandwidth}. "
                f"Consider increasing bandwidth. Attempting boundary traceback."
            )
            # Find the best reachable cell on the last row or column
            best_val = float('-inf')
            best_i, best_j = i, j
            for jj in range(max(0, i - self.bandwidth), min(len(seq2) + 1, i + self.bandwidth + 1)):
                if jj <= len(seq2) and score[i][jj] > best_val:
                    best_val = score[i][jj]
                    best_i, best_j = i, jj
            for ii in range(max(0, j - self.bandwidth), min(len(seq1) + 1, j + self.bandwidth + 1)):
                if ii <= len(seq1) and score[ii][j] > best_val:
                    best_val = score[ii][j]
                    best_i, best_j = ii, j

            # Add trailing gaps
            while i > best_i:
                aligned1.append(seq1[i - 1])
                aligned2.append('-')
                i -= 1
            while j > best_j:
                aligned1.append('-')
                aligned2.append(seq2[j - 1])
                j -= 1

        while i > 0 or j > 0:
            if i > 0 and j > 0 and traceback[i][j] == self.DIAGONAL:
                aligned1.append(seq1[i - 1])
                aligned2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif i > 0 and traceback[i][j] == self.UP:
                aligned1.append(seq1[i - 1])
                aligned2.append('-')
                i -= 1
            elif j > 0 and traceback[i][j] == self.LEFT:
                aligned1.append('-')
                aligned2.append(seq2[j - 1])
                j -= 1
            else:
                # Safety: should not happen in a correct traceback,
                # but handle gracefully
                if i > 0:
                    aligned1.append(seq1[i - 1])
                    aligned2.append('-')
                    i -= 1
                elif j > 0:
                    aligned1.append('-')
                    aligned2.append(seq2[j - 1])
                    j -= 1
                else:
                    break

        return ''.join(reversed(aligned1)), ''.join(reversed(aligned2))

    def _compute_stats(self, aligned1: str, aligned2: str) -> Tuple[int, int, int]:
        """Compute match, mismatch, and gap counts."""
        matches = mismatches = gaps = 0
        for c1, c2 in zip(aligned1, aligned2):
            if c1 == '-' or c2 == '-':
                gaps += 1
            elif c1 == c2:
                matches += 1
            else:
                mismatches += 1
        return matches, mismatches, gaps


# =============================================================================
# Stage (vi): CS-NW Complete Pipeline
# =============================================================================

class CSNW:
    """
    Complete Compositional Scoring Needleman-Wunsch pipeline.

    Integrates all six stages:
        (i)   Sequence Preprocessing
        (ii)  Parallel Frequency Aggregation
        (iii) Scoring Matrix Construction
        (iv)  Banded NW Alignment
        (v)   Traceback
        (vi)  Result Generation

    Parameters from paper:
        - bandwidth k = 5 (default)
        - gap penalty delta = -2 (default)
        - mismatch penalty gamma = 1 (default)
        - epsilon = 1e-6
    """

    def __init__(self, bandwidth: int = 5, gap_penalty: float = -2.0,
                 gamma: float = 1.0, epsilon: float = EPSILON,
                 alphabet: List[str] = None, n_workers: int = None):
        """
        Args:
            bandwidth: Band width k for banded alignment.
            gap_penalty: Gap penalty delta_g.
            gamma: Default mismatch penalty for unobserved dipeptides.
            epsilon: Small constant to prevent division by zero.
            alphabet: Character alphabet. Auto-detected if None.
            n_workers: Number of parallel workers for frequency aggregation.
        """
        self.bandwidth = bandwidth
        self.gap_penalty = gap_penalty
        self.gamma = gamma
        self.epsilon = epsilon
        self.alphabet = alphabet
        self.n_workers = n_workers

        self.scoring_matrix_builder = CompositionalScoringMatrix(gamma=gamma, epsilon=epsilon)
        self.aligner = BandedNeedlemanWunsch(gap_penalty=gap_penalty, bandwidth=bandwidth)

    def align_pair(self, seq1: str, seq2: str, dataset: List[str] = None,
                   verbose: bool = True) -> Dict:
        """
        Align two sequences using the full CS-NW pipeline.

        Args:
            seq1: First sequence.
            seq2: Second sequence.
            dataset: Optional larger dataset for computing global frequencies.
                     If None, frequencies are computed from seq1 and seq2 only
                     (as in the paper's worked example).
            verbose: Print detailed alignment information.

        Returns:
            Dictionary with full alignment results and metrics.
        """
        # Start tracking time and memory
        tracemalloc.start()
        start_time = time.time()

        # Clean sequences
        seq1_clean = ''.join(c for c in seq1.upper().strip() if c.isalpha())
        seq2_clean = ''.join(c for c in seq2.upper().strip() if c.isalpha())

        # Auto-detect alphabet
        if self.alphabet is None:
            preprocessor = SequencePreprocessor()
            alphabet = preprocessor.detect_alphabet(seq1_clean + seq2_clean)
        else:
            alphabet = self.alphabet

        # Stage (i) + (ii) + (iii): Build compositional scoring matrix
        scoring_matrix, char_to_idx = self.scoring_matrix_builder.build_from_sequences(
            seq1_clean, seq2_clean, alphabet=alphabet, dataset=dataset
        )

        # Stage (iv) + (v): Banded alignment + traceback
        result = self.aligner.align(seq1_clean, seq2_clean, scoring_matrix, char_to_idx)

        # Timing and memory
        elapsed = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result["runtime_seconds"] = elapsed
        result["memory_current_bytes"] = current_mem
        result["memory_peak_bytes"] = peak_mem
        result["memory_peak_mb"] = peak_mem / (1024 * 1024)
        result["bandwidth"] = self.bandwidth
        result["gap_penalty"] = self.gap_penalty
        result["gamma"] = self.gamma
        result["alphabet"] = alphabet
        result["scoring_matrix"] = scoring_matrix
        result["char_to_idx"] = char_to_idx

        if verbose:
            self._print_results(result, seq1_clean, seq2_clean)

        return result

    def align_batch(self, sequence_pairs: List[Tuple[str, str]],
                    dataset: List[str] = None) -> List[Dict]:
        """
        Align multiple sequence pairs.

        When a dataset is provided, the global frequencies are computed
        once from the full dataset and reused for all pairs. This is
        the intended use case for large-scale analyses (Section VI-B).

        Args:
            sequence_pairs: List of (seq1, seq2) tuples.
            dataset: Full dataset for global frequency computation.

        Returns:
            List of alignment result dictionaries.
        """
        # Pre-compute scoring matrix from dataset
        if dataset and len(dataset) > 2:
            all_seqs = dataset
        else:
            all_seqs = []
            for s1, s2 in sequence_pairs:
                all_seqs.extend([s1, s2])

        if self.alphabet is None:
            preprocessor = SequencePreprocessor()
            combined = ''.join(s.upper() for s in all_seqs[:10])  # Sample for detection
            alphabet = preprocessor.detect_alphabet(combined)
        else:
            alphabet = self.alphabet

        # Compute global frequencies once
        aggregator = ParallelFrequencyAggregator(n_workers=self.n_workers)
        global_aac, global_dpc = aggregator.aggregate(all_seqs, alphabet)
        scoring_matrix, char_to_idx = self.scoring_matrix_builder.build(
            global_aac, global_dpc, alphabet
        )

        results = []
        for i, (s1, s2) in enumerate(sequence_pairs):
            logger.info(f"Aligning pair {i+1}/{len(sequence_pairs)}")
            result = self.aligner.align(s1, s2, scoring_matrix, char_to_idx)
            results.append(result)

        return results

    def _print_results(self, result: Dict, seq1: str, seq2: str):
        """Print formatted alignment results."""
        print("\n" + "=" * 70)
        print("CS-NW ALIGNMENT RESULTS")
        print("=" * 70)
        print(f"Sequence 1: {seq1[:50]}{'...' if len(seq1) > 50 else ''} (length: {len(seq1)})")
        print(f"Sequence 2: {seq2[:50]}{'...' if len(seq2) > 50 else ''} (length: {len(seq2)})")
        print(f"Bandwidth (k): {result['bandwidth']}")
        print(f"Gap penalty (δ): {result['gap_penalty']}")
        print("-" * 70)

        # Show alignment (truncated for long sequences)
        a1 = result["aligned_seq1"]
        a2 = result["aligned_seq2"]
        mid = ""
        for c1, c2 in zip(a1, a2):
            if c1 == c2 and c1 != '-':
                mid += "|"
            elif c1 == '-' or c2 == '-':
                mid += " "
            else:
                mid += "."

        line_width = 60
        for start in range(0, len(a1), line_width):
            end = start + line_width
            print(f"  Seq1: {a1[start:end]}")
            print(f"        {mid[start:end]}")
            print(f"  Seq2: {a2[start:end]}")
            if end < len(a1):
                print()

        print("-" * 70)
        print(f"Alignment Score : {result['score']:.4f}" if result['score'] is not None else "Score: N/A (unreachable)")
        print(f"Matches         : {result['matches']}")
        print(f"Mismatches      : {result['mismatches']}")
        print(f"Gaps            : {result['gaps']}")
        print(f"Alignment Length: {result['alignment_length']}")
        print(f"Sequence Identity: {result['identity']:.2f}%")
        print(f"Runtime         : {result['runtime_seconds']:.4f} seconds")
        print(f"Peak Memory     : {result['memory_peak_mb']:.2f} MB")
        print("=" * 70)


# =============================================================================
# FASTA I/O Utilities
# =============================================================================

def read_fasta(filepath: str) -> List[Tuple[str, str]]:
    """
    Read sequences from a FASTA file.

    Args:
        filepath: Path to FASTA file.

    Returns:
        List of (header, sequence) tuples.
    """
    sequences = []
    current_header = None
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)

    if current_header is not None:
        sequences.append((current_header, ''.join(current_seq)))

    return sequences


def write_alignment(result: Dict, filepath: str, header1: str = "Seq1", header2: str = "Seq2"):
    """
    Write alignment result to a file in FASTA-like format.

    Args:
        result: Alignment result dictionary.
        filepath: Output file path.
        header1: Header for first sequence.
        header2: Header for second sequence.
    """
    with open(filepath, 'w') as f:
        f.write(f">{header1}\n{result['aligned_seq1']}\n")
        f.write(f">{header2}\n{result['aligned_seq2']}\n")
        f.write(f"\n# Score: {result['score']}\n")
        f.write(f"# Identity: {result['identity']:.2f}%\n")
        f.write(f"# Matches: {result['matches']}, Mismatches: {result['mismatches']}, Gaps: {result['gaps']}\n")


# =============================================================================
# Evaluation Metrics
# =============================================================================

class AlignmentMetrics:
    """
    Evaluation metrics as described in Section VI-D of the paper.

    Includes:
        - SP (Sum-of-Pairs) Score
        - Alignment Accuracy
        - Sequence Identity
        - Precision, Recall, F1 Score
    """

    @staticmethod
    def sequence_identity(aligned_seq1: str, aligned_seq2: str) -> float:
        """
        Compute sequence identity: fraction of positions that are identical
        (excluding gap-gap columns).
        """
        matches = 0
        total = 0
        for c1, c2 in zip(aligned_seq1, aligned_seq2):
            if c1 == '-' and c2 == '-':
                continue
            total += 1
            if c1 == c2:
                matches += 1
        return (matches / total * 100) if total > 0 else 0.0

    @staticmethod
    def sp_score(aligned_seq1: str, aligned_seq2: str,
                 match_score: float = 1.0, mismatch_score: float = -1.0,
                 gap_score: float = -2.0) -> float:
        """
        Compute Sum-of-Pairs score for the alignment.

        For pairwise alignment, this is the sum of column scores.
        """
        total = 0.0
        for c1, c2 in zip(aligned_seq1, aligned_seq2):
            if c1 == '-' or c2 == '-':
                total += gap_score
            elif c1 == c2:
                total += match_score
            else:
                total += mismatch_score
        return total

    @staticmethod
    def alignment_accuracy(test_aligned1: str, test_aligned2: str,
                           ref_aligned1: str, ref_aligned2: str) -> float:
        """
        Compute alignment accuracy against a reference alignment.

        Accuracy = (number of correctly aligned columns) / (total columns in reference)
        """
        # Align test and reference by finding matching columns
        correct = 0
        total = max(len(ref_aligned1), 1)

        # Simple column-by-column comparison
        min_len = min(len(test_aligned1), len(ref_aligned1))
        for i in range(min_len):
            if (test_aligned1[i] == ref_aligned1[i] and
                test_aligned2[i] == ref_aligned2[i]):
                correct += 1

        return (correct / total) * 100

    @staticmethod
    def precision_recall_f1(test_aligned1: str, test_aligned2: str,
                            ref_aligned1: str, ref_aligned2: str) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 score of aligned residue pairs.

        A "correctly aligned pair" means residue i in seq1 is aligned with
        residue j in seq2 in both the test and reference alignments.
        """
        def get_aligned_pairs(a1, a2):
            """Extract (pos_in_seq1, pos_in_seq2) for all non-gap columns."""
            pairs = set()
            pos1 = pos2 = 0
            for c1, c2 in zip(a1, a2):
                if c1 != '-' and c2 != '-':
                    pairs.add((pos1, pos2))
                if c1 != '-':
                    pos1 += 1
                if c2 != '-':
                    pos2 += 1
            return pairs

        test_pairs = get_aligned_pairs(test_aligned1, test_aligned2)
        ref_pairs = get_aligned_pairs(ref_aligned1, ref_aligned2)

        if len(test_pairs) == 0:
            return 0.0, 0.0, 0.0

        true_positives = len(test_pairs & ref_pairs)
        precision = true_positives / len(test_pairs) if len(test_pairs) > 0 else 0.0
        recall = true_positives / len(ref_pairs) if len(ref_pairs) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return precision, recall, f1


# =============================================================================
# Standard Needleman-Wunsch (Baseline for Comparison)
# =============================================================================

class StandardNeedlemanWunsch:
    """
    Standard (non-banded, non-compositional) Needleman-Wunsch for baseline comparison.
    Uses a simple match/mismatch/gap scoring scheme.
    """

    def __init__(self, match_score: float = 1.0, mismatch_score: float = -1.0,
                 gap_penalty: float = -2.0):
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty

    def align(self, seq1: str, seq2: str) -> Dict:
        """Perform standard NW alignment (O(MN) time and space)."""
        tracemalloc.start()
        start_time = time.time()

        seq1 = ''.join(c for c in seq1.upper().strip() if c.isalpha())
        seq2 = ''.join(c for c in seq2.upper().strip() if c.isalpha())

        M, N = len(seq1), len(seq2)
        score = np.zeros((M + 1, N + 1), dtype=np.float64)
        traceback_mat = np.zeros((M + 1, N + 1), dtype=np.int8)

        for i in range(1, M + 1):
            score[i][0] = i * self.gap_penalty
            traceback_mat[i][0] = 2  # UP
        for j in range(1, N + 1):
            score[0][j] = j * self.gap_penalty
            traceback_mat[0][j] = 3  # LEFT

        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if seq1[i-1] == seq2[j-1]:
                    diag = score[i-1][j-1] + self.match_score
                else:
                    diag = score[i-1][j-1] + self.mismatch_score
                up = score[i-1][j] + self.gap_penalty
                left = score[i][j-1] + self.gap_penalty

                best = max(diag, up, left)
                score[i][j] = best
                if best == diag:
                    traceback_mat[i][j] = 1
                elif best == up:
                    traceback_mat[i][j] = 2
                else:
                    traceback_mat[i][j] = 3

        # Traceback
        aligned1, aligned2 = [], []
        i, j = M, N
        while i > 0 or j > 0:
            if i > 0 and j > 0 and traceback_mat[i][j] == 1:
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                i -= 1; j -= 1
            elif i > 0 and traceback_mat[i][j] == 2:
                aligned1.append(seq1[i-1])
                aligned2.append('-')
                i -= 1
            elif j > 0:
                aligned1.append('-')
                aligned2.append(seq2[j-1])
                j -= 1
            else:
                break

        aligned1 = ''.join(reversed(aligned1))
        aligned2 = ''.join(reversed(aligned2))

        elapsed = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != '-')
        mismatches = sum(1 for a, b in zip(aligned1, aligned2) if a != b and a != '-' and b != '-')
        gaps = sum(1 for a, b in zip(aligned1, aligned2) if a == '-' or b == '-')

        return {
            "aligned_seq1": aligned1,
            "aligned_seq2": aligned2,
            "score": score[M][N],
            "matches": matches,
            "mismatches": mismatches,
            "gaps": gaps,
            "alignment_length": len(aligned1),
            "identity": matches / len(aligned1) * 100 if len(aligned1) > 0 else 0,
            "runtime_seconds": elapsed,
            "memory_peak_mb": peak_mem / (1024 * 1024),
        }


# =============================================================================
# Verification: Paper's Worked Example (Section V-B)
# =============================================================================

def verify_paper_example():
    """
    Reproduce the paper's worked example from Section V-B.

    Sequences: Seq1 = ACGTGCA, Seq2 = AGTGCC
    Bandwidth: k = 2
    Gap penalty: delta = -2

    Expected results (from paper Tables 2-6):
        Amino Acid Frequencies: A=3/13, C=4/13, G=4/13, T=2/13
        Scoring Matrix: Table 4
        Optimal Alignment:
            Seq1: A C G T G C A
            Seq2: A - G T G C C
        Matches: 5, Mismatches: 1, Gaps: 1
    """
    print("\n" + "#" * 70)
    print("# VERIFICATION: Paper's Worked Example (Section V-B)")
    print("#" * 70)

    seq1 = "ACGTGCA"
    seq2 = "AGTGCC"
    alphabet = list("ACGT")

    # --- Verify Stage (i): Amino Acid Frequencies ---
    print("\n--- Stage (i): Amino Acid Composition ---")

    # The paper computes global frequencies by pooling both sequences
    # into one combined string: ACGTGCA + AGTGCC = ACGTGCAAGTGCC (length 13)
    combined = seq1 + seq2  # "ACGTGCAAGTGCC", length 13
    total_len = len(combined)
    print(f"Combined sequence: {combined} (length: {total_len})")

    expected_aac = {'A': 3/13, 'C': 4/13, 'G': 4/13, 'T': 2/13}
    counts = Counter(combined)
    actual_aac = {c: counts[c]/total_len for c in alphabet}
    print(f"Expected AAC: { {k: round(v, 3) for k, v in expected_aac.items()} }")
    print(f"Computed AAC: { {k: round(v, 3) for k, v in actual_aac.items()} }")

    # --- Verify dipeptide frequencies ---
    print("\n--- Stage (i): Dipeptide Composition ---")
    # Dipeptides from seq1: AC, CG, GT, TG, GC, CA  (6 dipeptides)
    # Dipeptides from seq2: AG, GT, TG, GC, CC       (5 dipeptides)
    # Total dipeptides: 11
    dp_seq1 = [seq1[i:i+2] for i in range(len(seq1)-1)]
    dp_seq2 = [seq2[i:i+2] for i in range(len(seq2)-1)]
    all_dp = dp_seq1 + dp_seq2
    dp_counts = Counter(all_dp)
    total_dp = len(all_dp)
    print(f"Dipeptides from Seq1: {dp_seq1}")
    print(f"Dipeptides from Seq2: {dp_seq2}")
    print(f"Total dipeptides: {total_dp}")
    print(f"Dipeptide counts: {dict(dp_counts)}")
    for dp, cnt in sorted(dp_counts.items()):
        print(f"  {dp}: {cnt}/{total_dp} = {cnt/total_dp:.3f}")

    # --- Run CS-NW with k=2 (paper's example uses k=2) ---
    print("\n--- Running CS-NW (k=2, δ=-2) ---")
    csnw = CSNW(bandwidth=2, gap_penalty=-2.0, gamma=1.0, alphabet=alphabet)
    result = csnw.align_pair(seq1, seq2, verbose=True)

    # --- Verify scoring matrix against Table 4 ---
    print("\n--- Scoring Matrix Verification (Table 4) ---")
    idx = result["char_to_idx"]
    sm = result["scoring_matrix"]
    print("Computed Scoring Matrix:")
    header = "     " + "     ".join(f"{c:>6}" for c in alphabet)
    print(header)
    for a in alphabet:
        row = f"  {a}  "
        for b in alphabet:
            row += f"{sm[idx[a]][idx[b]]:>7.3f}"
        print(row)

    print("\nExpected (from Table 4):")
    print("     A: [ 0.813, -0.917, -0.917, -1.000]")
    print("     C: [-0.917,  0.765, -0.917, -1.000]")
    print("     G: [-0.917, -0.917,  0.765, -0.846]")
    print("     T: [-1.000, -1.000, -0.846,  0.867]")

    # Note on Table 4 discrepancy
    print("\n  NOTE: Paper's Table 4 shows G->A = -0.917, but the dipeptide 'GA'")
    print("  is NOT observed in the combined sequences (Table 3 lists AG, not GA).")
    print("  Correct value for G->A is -1.000 (default gamma). This is a minor")
    print("  error in the paper's Table 4. Our implementation is correct.")

    # --- Verify banded score matrix against Table 5 ---
    print("\n--- Banded Score Matrix Verification (Table 5) ---")
    score_mat = result['score_matrix']
    M_len, N_len = 7, 6
    print(f"{'':>4}", end="")
    for hdr in ['-', 'A', 'G', 'T', 'G', 'C', 'C']:
        print(f"{hdr:>8}", end="")
    print()
    row_labels = ['-'] + list(seq1)
    for i in range(M_len + 1):
        print(f"{row_labels[i]:>4}", end="")
        for j in range(N_len + 1):
            val = score_mat[i][j]
            if val == float('-inf'):
                print(f"{'---':>8}", end="")
            else:
                print(f"{val:>8.3f}", end="")
        print()

    # --- Verify expected alignment ---
    print("\n--- Expected Alignment (Table 6) ---")
    print("  Expected Seq1: A C G T G C A")
    print("  Expected Seq2: A - G T G C C")
    print(f"  Got      Seq1: {' '.join(result['aligned_seq1'])}")
    print(f"  Got      Seq2: {' '.join(result['aligned_seq2'])}")

    matches_ok = result['matches'] == 5
    mismatches_ok = result['mismatches'] == 1
    gaps_ok = result['gaps'] == 1
    alignment_ok = (result['aligned_seq1'] == 'ACGTGCA' and result['aligned_seq2'] == 'A-GTGCC')

    print(f"\n  Matches=5: {'PASS' if matches_ok else 'FAIL'} (got {result['matches']})")
    print(f"  Mismatches=1: {'PASS' if mismatches_ok else 'FAIL'} (got {result['mismatches']})")
    print(f"  Gaps=1: {'PASS' if gaps_ok else 'FAIL'} (got {result['gaps']})")
    print(f"  Alignment match: {'PASS' if alignment_ok else 'FAIL'}")

    if all([matches_ok, mismatches_ok, gaps_ok, alignment_ok]):
        print("\n  >>> ALL VERIFICATION CHECKS PASSED <<<")
    else:
        print("\n  >>> SOME CHECKS FAILED - INVESTIGATE <<<")

    return result


# =============================================================================
# Comparison: CS-NW vs Standard NW
# =============================================================================

def compare_csnw_vs_standard_nw():
    """
    Compare CS-NW against standard NW on the paper's example
    and report the differences in score, identity, runtime, and memory.
    """
    print("\n" + "#" * 70)
    print("# COMPARISON: CS-NW vs Standard NW")
    print("#" * 70)

    seq1 = "ACGTGCA"
    seq2 = "AGTGCC"

    # CS-NW (k=5 as default in paper)
    print("\n--- CS-NW (k=5, δ=-2) ---")
    csnw = CSNW(bandwidth=5, gap_penalty=-2.0)
    result_csnw = csnw.align_pair(seq1, seq2, verbose=False)
    print(f"  Alignment: {result_csnw['aligned_seq1']}")
    print(f"             {result_csnw['aligned_seq2']}")
    print(f"  Score: {result_csnw['score']:.4f}")
    print(f"  Identity: {result_csnw['identity']:.2f}%")
    print(f"  Runtime: {result_csnw['runtime_seconds']:.6f}s")
    print(f"  Memory: {result_csnw['memory_peak_mb']:.4f} MB")

    # Standard NW
    print("\n--- Standard NW (match=1, mismatch=-1, gap=-2) ---")
    nw = StandardNeedlemanWunsch(match_score=1.0, mismatch_score=-1.0, gap_penalty=-2.0)
    result_nw = nw.align(seq1, seq2)
    print(f"  Alignment: {result_nw['aligned_seq1']}")
    print(f"             {result_nw['aligned_seq2']}")
    print(f"  Score: {result_nw['score']:.4f}")
    print(f"  Identity: {result_nw['identity']:.2f}%")
    print(f"  Runtime: {result_nw['runtime_seconds']:.6f}s")
    print(f"  Memory: {result_nw['memory_peak_mb']:.4f} MB")


# =============================================================================
# Demo: Protein Sequence Alignment
# =============================================================================

def demo_protein_alignment():
    """
    Demonstrate CS-NW on real protein sequences.
    Uses short segments of hemoglobin alpha and beta chains.
    """
    print("\n" + "#" * 70)
    print("# DEMO: Protein Sequence Alignment")
    print("#" * 70)

    # Human Hemoglobin Alpha Chain (first 50 residues)
    hba = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
    # Human Hemoglobin Beta Chain (first 50 residues)
    hbb = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST"

    print(f"\nHemoglobin Alpha (50 aa): {hba}")
    print(f"Hemoglobin Beta  (50 aa): {hbb}")

    csnw = CSNW(bandwidth=10, gap_penalty=-2.0, gamma=1.0, alphabet=AMINO_ACIDS)
    result = csnw.align_pair(hba, hbb, verbose=True)

    return result


# =============================================================================
# Demo: Bandwidth Sensitivity Analysis (Fig. 3 from paper)
# =============================================================================

def bandwidth_sensitivity_analysis():
    """
    Reproduce the bandwidth sensitivity analysis from Figure 3 of the paper.
    Tests k = 1, 2, 3, 5, 7, 10 and reports accuracy/runtime trade-offs.
    """
    print("\n" + "#" * 70)
    print("# ANALYSIS: Bandwidth Sensitivity (cf. Figure 3)")
    print("#" * 70)

    # Use hemoglobin sequences for a meaningful test
    seq1 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
    seq2 = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST"

    # Get reference alignment from standard NW
    nw = StandardNeedlemanWunsch(match_score=1.0, mismatch_score=-1.0, gap_penalty=-2.0)
    ref_result = nw.align(seq1, seq2)

    bandwidths = [1, 2, 3, 5, 7, 10, 15, 25]
    print(f"\n{'k':>4} | {'Identity%':>10} | {'Accuracy%':>10} | {'Runtime(s)':>11} | {'Memory(MB)':>10} | {'Score':>10}")
    print("-" * 72)

    for k in bandwidths:
        csnw = CSNW(bandwidth=k, gap_penalty=-2.0, gamma=1.0, alphabet=AMINO_ACIDS)
        result = csnw.align_pair(seq1, seq2, verbose=False)

        accuracy = AlignmentMetrics.alignment_accuracy(
            result['aligned_seq1'], result['aligned_seq2'],
            ref_result['aligned_seq1'], ref_result['aligned_seq2']
        )

        print(f"{k:>4} | {result['identity']:>9.2f}% | {accuracy:>9.2f}% | "
              f"{result['runtime_seconds']:>10.6f}s | {result['memory_peak_mb']:>9.4f} | "
              f"{result['score'] if result['score'] is not None else 'N/A':>10}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  CS-NW: Compositional Scoring Needleman-Wunsch Algorithm")
    print("  Implementation for Ph.D. Research Validation")
    print("=" * 70)

    # 1. Verify against the paper's worked example
    verify_paper_example()

    # 2. Compare CS-NW vs Standard NW
    compare_csnw_vs_standard_nw()

    # 3. Protein sequence demo
    demo_protein_alignment()

    # 4. Bandwidth sensitivity analysis
    bandwidth_sensitivity_analysis()

    print("\n\nAll demonstrations complete.")
