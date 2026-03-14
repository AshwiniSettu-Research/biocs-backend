"""
Self-Improved Black-Winged Kite (SA-BWK) Optimizer.

Bio-inspired meta-heuristic optimization algorithm for optimizing the
Kolaskar-Tongaonkar scoring matrix weights. Based on the hunting behavior
of the black-winged kite (Elanus caeruleus):

1. Initialization: Random population within search bounds
2. Attacking behavior: Nonlinear convergence factor drives exploitation
3. Migration behavior: Cauchy mutation for exploration/escaping local optima

Objective: Find optimal weights for the 39 physicochemical features that
maximize classification performance when used in the K-T scoring function.
"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from .features import extract_physicochemical_features, PROPERTY_NAMES
from .config import (
    SABWK_POP_SIZE, SABWK_MAX_ITER, MAX_SEQ_LEN,
    NUM_PHYSICOCHEMICAL_FEATURES,
)


def _evaluate_weights(weights, sequences, labels, max_len=MAX_SEQ_LEN):
    """
    Evaluate a weight vector by computing weighted K-T scores and
    measuring classification performance via KNN + macro F1.

    Args:
        weights: (39,) feature weight vector
        sequences: list of peptide strings
        labels: numpy array of integer labels
        max_len: sequence padding length

    Returns:
        fitness: macro F1 score (higher is better)
    """
    n = len(sequences)
    # Compute weighted scores per sequence
    X = np.zeros((n, max_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        features = extract_physicochemical_features(seq, max_len)
        # Weighted sum across 39 features at each position
        X[i] = np.dot(weights, features)

    # Quick KNN evaluation (k=5) for fitness
    knn = KNeighborsClassifier(n_neighbors=min(5, n - 1), metric="euclidean")
    try:
        knn.fit(X, labels)
        preds = knn.predict(X)
        return f1_score(labels, preds, average="macro", zero_division=0)
    except Exception:
        return 0.0


def sa_bwk_optimize(sequences, labels, pop_size=SABWK_POP_SIZE,
                    max_iter=SABWK_MAX_ITER, seed=42, verbose=True):
    """
    Run the SA-BWK optimizer to find optimal K-T feature weights.

    The algorithm simulates the black-winged kite's hunting strategy:
    - Population of candidate weight vectors
    - Attacking phase: agents converge toward the best solution using a
      nonlinear convergence factor
    - Migration phase: Cauchy mutation injects diversity to escape local optima

    Args:
        sequences: list of peptide strings
        labels: numpy array of integer labels
        pop_size: number of kites in the population
        max_iter: maximum optimization iterations
        seed: random seed for reproducibility
        verbose: print progress

    Returns:
        best_weights: (39,) optimal weight vector
        best_fitness: best macro F1 score achieved
        history: list of best fitness per iteration
    """
    rng = np.random.RandomState(seed)
    dim = NUM_PHYSICOCHEMICAL_FEATURES  # 39

    # Initialize population in [0, 1] range
    population = rng.uniform(0, 1, size=(pop_size, dim)).astype(np.float32)

    # Evaluate initial population
    fitness = np.array([_evaluate_weights(w, sequences, labels)
                        for w in population])

    best_idx = np.argmax(fitness)
    best_weights = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    history = [best_fitness]

    if verbose:
        print(f"SA-BWK Initialization: best F1 = {best_fitness:.4f}")

    for t in range(1, max_iter + 1):
        # Nonlinear convergence factor (decreases from 2 to 0)
        a = 2.0 * (1.0 - t / max_iter)

        for i in range(pop_size):
            # --- Attacking behavior ---
            r1 = rng.random()
            r2 = rng.random()

            # Convergence coefficient
            A = 2.0 * a * r1 - a
            C = 2.0 * r2

            # Distance to best solution
            D = np.abs(C * best_weights - population[i])

            # Position update (exploitation)
            new_pos = best_weights - A * D

            # --- Migration behavior (Cauchy mutation) ---
            # Probability of migration increases as convergence slows
            p_migrate = 0.5 * (1.0 - t / max_iter)
            if rng.random() < p_migrate:
                # Cauchy mutation for exploration
                cauchy_step = rng.standard_cauchy(dim).astype(np.float32)
                cauchy_step = np.clip(cauchy_step, -5, 5)  # bound extreme values
                scale = 0.1 * (1.0 - t / max_iter)
                new_pos = new_pos + scale * cauchy_step

            # Clip to valid range [0, 1]
            new_pos = np.clip(new_pos, 0, 1).astype(np.float32)

            # Evaluate new position
            new_fitness = _evaluate_weights(new_pos, sequences, labels)

            # Greedy selection
            if new_fitness >= fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fitness

                if new_fitness > best_fitness:
                    best_weights = new_pos.copy()
                    best_fitness = new_fitness

        history.append(best_fitness)
        if verbose and (t % 10 == 0 or t == 1):
            print(f"SA-BWK Iter {t}/{max_iter}: best F1 = {best_fitness:.4f}")

    if verbose:
        print(f"SA-BWK Complete: best F1 = {best_fitness:.4f}")

    return best_weights, best_fitness, history


def backward_feature_selection(weights, threshold=0.05):
    """
    Perform backward feature selection based on optimized weights.
    Features with weights below the threshold are considered unimportant.

    Args:
        weights: (39,) weight vector from SA-BWK
        threshold: minimum weight to keep a feature

    Returns:
        selected_indices: list of feature indices to keep
        selected_names: list of corresponding feature names
        importance_ranking: list of (name, weight) sorted by weight descending
    """
    importance_ranking = sorted(
        zip(PROPERTY_NAMES, weights),
        key=lambda x: x[1],
        reverse=True,
    )

    selected_indices = []
    selected_names = []
    for idx, (name, w) in enumerate(zip(PROPERTY_NAMES, weights)):
        if w >= threshold:
            selected_indices.append(idx)
            selected_names.append(name)

    return selected_indices, selected_names, importance_ranking
