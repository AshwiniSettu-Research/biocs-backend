"""
Training script for the MLPT model.

Performs:
1. SA-BWK optimization of K-T feature weights
2. Model training with class-weighted loss and early stopping
3. Evaluation with full metrics (accuracy, precision, sensitivity, etc.)
4. Saves best checkpoint with metrics and K-T parameters
5. Runs both 70/30 and 80/20 train/test splits as described in the paper
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, classification_report,
)

from .config import (
    DATASET_PATH, MODEL_SAVE_DIR, NUM_CLASSES, CLASS_NAMES,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE,
    TRAIN_SPLIT_1, TRAIN_SPLIT_2,
)
from .data_loader import load_dataset, create_dataloaders
from .mlpt_model import MLPTModel
from .sabwk_optimizer import sa_bwk_optimize, backward_feature_selection


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    """
    Compute comprehensive evaluation metrics.

    Returns dict with: accuracy, macro_precision, macro_sensitivity (recall),
    macro_specificity, macro_f1, macro_fnr, macro_fpr, macro_npv, mcc
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    sensitivity = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Per-class specificity, FNR, FPR, NPV from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    fnrs = []
    fprs = []
    npvs = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        specificities.append(specificity)
        fnrs.append(fnr)
        fprs.append(fpr)
        npvs.append(npv)

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_sensitivity": sensitivity,
        "macro_specificity": np.mean(specificities),
        "macro_f1": f1,
        "macro_fnr": np.mean(fnrs),
        "macro_fpr": np.mean(fprs),
        "macro_npv": np.mean(npvs),
        "mcc": mcc,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for encoded_seq, phys_features, kt_scores, labels in loader:
        encoded_seq = encoded_seq.to(device)
        phys_features = phys_features.to(device)
        kt_scores = kt_scores.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(encoded_seq, phys_features, kt_scores)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a data loader. Returns loss, predictions, true labels."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    for encoded_seq, phys_features, kt_scores, labels in loader:
        encoded_seq = encoded_seq.to(device)
        phys_features = phys_features.to(device)
        kt_scores = kt_scores.to(device)
        labels = labels.to(device)

        logits = model(encoded_seq, phys_features, kt_scores)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / max(num_batches, 1)

    return avg_loss, all_preds, all_labels


def train_model(train_ratio, kt_weights, sequences, labels, device, tag=""):
    """
    Full training pipeline for a given train/test split.

    Args:
        train_ratio: fraction for training (e.g. 0.7 or 0.8)
        kt_weights: optimized K-T weights from SA-BWK
        sequences: list of peptide strings
        labels: list of integer labels
        device: torch device
        tag: string tag for saving (e.g. "70_30" or "80_20")

    Returns:
        metrics: dict of evaluation metrics on test set
    """
    split_name = f"{int(train_ratio*100)}/{int((1-train_ratio)*100)}"
    print(f"\n{'='*60}")
    print(f"Training with {split_name} split ({tag})")
    print(f"{'='*60}")

    # Create data loaders
    train_loader, test_loader, class_weights = create_dataloaders(
        sequences, labels, train_ratio=train_ratio,
        batch_size=BATCH_SIZE, kt_weights=kt_weights,
    )
    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Model
    model = MLPTModel().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    # Early stopping
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)

        scheduler.step(val_f1)
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"{elapsed:.1f}s")

        # Early stopping check
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    # Load best model and final evaluation
    model.load_state_dict(best_state)
    model.to(device)
    _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(test_labels, test_preds)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Test Results ({split_name}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=CLASS_NAMES, zero_division=0,
    ))

    # Save checkpoint
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, f"mlpt_{tag}.pt")
    torch.save({
        "model_state_dict": best_state,
        "kt_weights": kt_weights,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "train_ratio": train_ratio,
        "class_names": CLASS_NAMES,
    }, save_path)
    print(f"Checkpoint saved: {save_path}")

    # Save metrics as JSON
    metrics_path = os.path.join(MODEL_SAVE_DIR, f"metrics_{tag}.json")
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    return metrics


def main():
    """Main training entry point."""
    print("MLPT Training Pipeline")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Step 1: Load dataset
    print("\nLoading dataset...")
    sequences, labels, class_to_idx, idx_to_class = load_dataset()
    labels = np.array(labels)
    print(f"Loaded {len(sequences)} sequences, {NUM_CLASSES} classes")
    for name, idx in class_to_idx.items():
        count = (labels == idx).sum()
        print(f"  {name}: {count}")

    # Step 2: SA-BWK optimization for K-T weights
    print("\nRunning SA-BWK optimization...")
    # Use a subset for faster optimization
    subset_size = min(1000, len(sequences))
    rng = np.random.RandomState(42)
    subset_idx = rng.choice(len(sequences), subset_size, replace=False)
    subset_seqs = [sequences[i] for i in subset_idx]
    subset_labels = labels[subset_idx]

    kt_weights, kt_fitness, kt_history = sa_bwk_optimize(
        subset_seqs, subset_labels, verbose=True,
    )

    # Feature importance
    selected_idx, selected_names, ranking = backward_feature_selection(kt_weights)
    print(f"\nTop 10 features by SA-BWK weight:")
    for name, weight in ranking[:10]:
        print(f"  {name}: {weight:.4f}")
    print(f"Selected features (>{0.05}): {len(selected_idx)}/{len(kt_weights)}")

    # Save K-T weights
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    kt_path = os.path.join(MODEL_SAVE_DIR, "kt_weights.npy")
    np.save(kt_path, kt_weights)
    print(f"K-T weights saved: {kt_path}")

    # Step 3: Train with 70/30 split
    metrics_70 = train_model(
        TRAIN_SPLIT_1, kt_weights, sequences, labels.tolist(),
        device, tag="70_30",
    )

    # Step 4: Train with 80/20 split
    metrics_80 = train_model(
        TRAIN_SPLIT_2, kt_weights, sequences, labels.tolist(),
        device, tag="80_20",
    )

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"\n70/30 Split:")
    print(f"  Accuracy:    {metrics_70['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics_70['macro_f1']:.4f}")
    print(f"  MCC:         {metrics_70['mcc']:.4f}")
    print(f"\n80/20 Split:")
    print(f"  Accuracy:    {metrics_80['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics_80['macro_f1']:.4f}")
    print(f"  MCC:         {metrics_80['mcc']:.4f}")


if __name__ == "__main__":
    main()
