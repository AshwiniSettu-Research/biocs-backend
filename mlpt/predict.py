"""
Inference wrapper for the MLPT model.

Loads a trained checkpoint and provides prediction functions for
single sequences or batches.
"""

import os
import re
import numpy as np
import torch

from .config import (
    MODEL_SAVE_DIR, MAX_SEQ_LEN, NUM_CLASSES, CLASS_NAMES,
    AA_TO_IDX, AMINO_ACIDS,
)
from .mlpt_model import MLPTModel
from .features import (
    extract_physicochemical_features,
    compute_kt_feature_vector,
    compute_kt_scores,
)
from .data_loader import encode_sequence


class MLPTPredictor:
    """Inference wrapper for the MLPT model."""

    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or self._get_device()
        self.model = None
        self.kt_weights = None
        self.metrics = None
        self.loaded = False

        if checkpoint_path is None:
            # Try to find best checkpoint
            checkpoint_path = self._find_best_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _find_best_checkpoint(self):
        """Find the best checkpoint file in the save directory."""
        if not os.path.exists(MODEL_SAVE_DIR):
            return None

        # Prefer 80/20 split model, fall back to 70/30
        for tag in ["80_20", "70_30"]:
            path = os.path.join(MODEL_SAVE_DIR, f"mlpt_{tag}.pt")
            if os.path.exists(path):
                return path
        return None

    def load(self, checkpoint_path):
        """Load a trained model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = MLPTModel().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.kt_weights = checkpoint.get("kt_weights")
        self.metrics = checkpoint.get("metrics", {})
        self.loaded = True

        # Also load K-T weights from separate file if available
        if self.kt_weights is None:
            kt_path = os.path.join(MODEL_SAVE_DIR, "kt_weights.npy")
            if os.path.exists(kt_path):
                self.kt_weights = np.load(kt_path)

    @staticmethod
    def clean_sequence(raw_seq):
        """Clean and validate a peptide sequence string."""
        # Strip FASTA header if present
        lines = raw_seq.strip().split("\n")
        seq_parts = []
        for line in lines:
            line = line.strip()
            if line.startswith(">"):
                continue
            seq_parts.append(line)
        seq = "".join(seq_parts)

        # Remove non-alpha characters and uppercase
        seq = re.sub(r"[^A-Za-z]", "", seq).upper()

        # Filter to valid amino acids
        valid_aa = set(AMINO_ACIDS)
        seq = "".join(c for c in seq if c in valid_aa)

        return seq

    @torch.no_grad()
    def predict(self, sequences):
        """
        Predict class for one or more peptide sequences.

        Uses batched inference: all valid sequences are encoded together and
        passed through the model in a single forward pass.

        Args:
            sequences: list of raw sequence strings (plain text or FASTA)

        Returns:
            list of prediction dicts, one per sequence (same order as input)
        """
        if not self.loaded:
            raise RuntimeError("No model loaded. Train first or provide a checkpoint.")

        n = len(sequences)

        # ------------------------------------------------------------------
        # Pass 1: clean all sequences, separate valid from invalid (too short)
        # ------------------------------------------------------------------
        cleaned_seqs = []          # cleaned string per input position
        valid_indices = []         # indices (into `sequences`) that are valid
        error_results = {}         # index -> error dict for invalid sequences

        for i, raw_seq in enumerate(sequences):
            seq = self.clean_sequence(raw_seq)
            cleaned_seqs.append(seq)
            if len(seq) < 3:
                error_results[i] = {
                    "sequence": raw_seq[:50],
                    "error": "Sequence too short (need at least 3 valid amino acids).",
                }
            else:
                valid_indices.append(i)

        # Short-circuit: if nothing is valid, return errors only
        if not valid_indices:
            return [error_results[i] for i in range(n)]

        # ------------------------------------------------------------------
        # Pass 2: batch-extract features for all valid sequences
        # ------------------------------------------------------------------
        valid_seqs = [cleaned_seqs[i] for i in valid_indices]
        batch_size = len(valid_seqs)

        encoded_np = np.stack([encode_sequence(s) for s in valid_seqs])            # (B, MAX_SEQ_LEN)
        phys_np = np.stack([extract_physicochemical_features(s) for s in valid_seqs])  # (B, 39, MAX_SEQ_LEN)
        kt_np = np.stack([compute_kt_feature_vector(s, self.kt_weights) for s in valid_seqs])  # (B, MAX_SEQ_LEN)

        # ------------------------------------------------------------------
        # Pass 3: single batched forward pass
        # ------------------------------------------------------------------
        encoded_t = torch.tensor(encoded_np, dtype=torch.long, device=self.device)
        phys_t = torch.tensor(phys_np, dtype=torch.float32, device=self.device)
        kt_t = torch.tensor(kt_np, dtype=torch.float32, device=self.device)

        logits = self.model(encoded_t, phys_t, kt_t)              # (B, NUM_CLASSES)
        all_probs = torch.softmax(logits, dim=-1).cpu().numpy()    # (B, NUM_CLASSES)

        # ------------------------------------------------------------------
        # Pass 4: unpack results and compute per-sequence K-T analysis
        # ------------------------------------------------------------------
        valid_results = {}
        for batch_idx, orig_idx in enumerate(valid_indices):
            seq = valid_seqs[batch_idx]
            probs = all_probs[batch_idx]

            predicted_idx = int(np.argmax(probs))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(probs[predicted_idx])

            # K-T analysis (inherently per-sequence)
            kt_per_residue, avg_antigenicity, antigenic_regions = compute_kt_scores(seq)

            # Build probability dict
            prob_dict = {
                CLASS_NAMES[i]: round(float(probs[i]), 4)
                for i in range(NUM_CLASSES)
            }

            valid_results[orig_idx] = {
                "sequence": seq,
                "sequence_length": len(seq),
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
                "kt_scores": kt_per_residue[:len(seq)].tolist(),
                "antigenicity_score": round(avg_antigenicity, 4),
                "antigenic_regions": antigenic_regions,
            }

        # ------------------------------------------------------------------
        # Pass 5: merge valid + error results in original input order
        # ------------------------------------------------------------------
        results = []
        for i in range(n):
            if i in error_results:
                results.append(error_results[i])
            else:
                results.append(valid_results[i])

        return results

    def get_model_info(self):
        """Return model metadata."""
        info = {
            "loaded": self.loaded,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "device": str(self.device),
        }
        if self.metrics:
            info["accuracy"] = self.metrics.get("accuracy")
            info["macro_f1"] = self.metrics.get("macro_f1")
        return info
