"""
Data loading and preprocessing for the MLPT model.
Loads the IEDB dataset, cleans it, and creates PyTorch DataLoaders.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from .config import (
    DATASET_PATH, MAX_SEQ_LEN, AA_TO_IDX, CLASS_NAMES, NUM_CLASSES,
)
from .features import extract_physicochemical_features, compute_kt_feature_vector


def load_dataset(path=None):
    """
    Load and clean the IEDB dataset from the Excel file.

    Returns:
        sequences: list of peptide sequence strings
        labels: list of integer class labels
        class_to_idx: dict mapping class name to index
        idx_to_class: dict mapping index to class name
    """
    import openpyxl

    if path is None:
        path = DATASET_PATH

    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active

    sequences = []
    labels = []
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    idx_to_class = {idx: name for idx, name in enumerate(CLASS_NAMES)}

    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        seq = row[0]
        cls = row[1]

        # Skip dirty rows
        if seq is None or cls is None:
            continue
        seq = str(seq).strip().replace("\xa0", "")
        cls = str(cls).strip()
        if cls == "class" or cls == "":
            continue

        # Validate class
        if cls not in class_to_idx:
            continue

        # Clean sequence: uppercase, only valid AA characters
        seq_clean = "".join(c for c in seq.upper() if c in AA_TO_IDX)
        if len(seq_clean) < 3:
            continue

        sequences.append(seq_clean)
        labels.append(class_to_idx[cls])

    wb.close()
    return sequences, labels, class_to_idx, idx_to_class


def encode_sequence(sequence, max_len=MAX_SEQ_LEN):
    """
    Integer-encode a peptide sequence with padding.

    Returns:
        numpy array of shape (max_len,) with dtype int64
    """
    encoded = np.zeros(max_len, dtype=np.int64)
    for i, aa in enumerate(sequence[:max_len]):
        encoded[i] = AA_TO_IDX.get(aa, 0)
    return encoded


class PeptideDataset(Dataset):
    """PyTorch Dataset for peptide sequences with precomputed features."""

    def __init__(self, sequences, labels, kt_weights=None):
        self.sequences = sequences
        self.labels = labels
        self.kt_weights = kt_weights

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Integer-encoded sequence
        encoded = encode_sequence(seq)

        # Physicochemical features (39, MAX_SEQ_LEN)
        phys_features = extract_physicochemical_features(seq)

        # K-T antigenicity scores (MAX_SEQ_LEN,)
        kt_scores = compute_kt_feature_vector(seq, self.kt_weights)

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(phys_features, dtype=torch.float32),
            torch.tensor(kt_scores, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def compute_class_weights(labels):
    """
    Compute class weights inversely proportional to class frequency.

    Returns:
        torch.FloatTensor of shape (NUM_CLASSES,)
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid division by zero
    total = len(labels)
    weights = total / (NUM_CLASSES * counts)
    # Normalize so min weight = 1.0
    weights = weights / weights.min()
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(sequences, labels, train_ratio=0.7, batch_size=32,
                       kt_weights=None, seed=42):
    """
    Create train and test DataLoaders with stratified split and class-balanced sampling.

    Returns:
        train_loader, test_loader, class_weights
    """
    # Stratified split
    train_seqs, test_seqs, train_labels, test_labels = train_test_split(
        sequences, labels, train_size=train_ratio,
        stratify=labels, random_state=seed,
    )

    # Datasets
    train_dataset = PeptideDataset(train_seqs, train_labels, kt_weights)
    test_dataset = PeptideDataset(test_seqs, test_labels, kt_weights)

    # Class weights for loss function
    class_weights = compute_class_weights(train_labels)

    # Weighted random sampler for oversampling minority classes
    sample_weights = class_weights[train_labels].numpy()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    return train_loader, test_loader, class_weights
