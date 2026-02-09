import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def knn_predict_with_event_mask(
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    event_test: np.ndarray,
    event_train: np.ndarray,
    k: int,
    ) -> np.ndarray:
    """
    Perform k-NN prediction with cosine similarity and
    exclusion of same-event neighbors.
    """

    sim = cosine_similarity(X_test, X_train)

    # Mask same-event neighbors
    for i in range(len(X_test)):
        same_event = event_train == event_test[i]
        sim[i, same_event] = -np.inf

    preds = np.empty(len(X_test), dtype=y_train.dtype)

    for i in range(len(X_test)):
        knn_idx = np.argsort(sim[i])[-k:]
        knn_labels = y_train[knn_idx]
        preds[i] = Counter(knn_labels).most_common(1)[0][0]

    return preds


def balance_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    event_train: np.ndarray,
    rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample training data to equal class sizes.

    Balancing is done without replacement, using the smallest
    class size in the training split.
    """

    balanced_indices = []

    classes, counts = np.unique(y_train, return_counts=True)
    n_per_class = counts.min()

    for cls in classes:
        cls_idx = np.where(y_train == cls)[0]
        sampled = rng.choice(cls_idx, size=n_per_class, replace=False)
        balanced_indices.append(sampled)

    balanced_indices = np.concatenate(balanced_indices)

    return (
        X_train[balanced_indices],
        y_train[balanced_indices],
        event_train[balanced_indices],
        n_per_class,
    )


def evaluate_embeddings_knn_cv(
    df: pd.DataFrame,
    y_col: str = "species",
    k: int = 5,
    n_splits: int = 5,
    random_state: int = 42,
    ) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[float],
    ]:
    """
    Balanced k-fold k-NN embedding evaluation with per-fold predictions.
    Training sets are class-balanced per fold.
    """

    rng = np.random.default_rng(random_state)

    X = np.vstack(df["emb"].values)
    y = np.asarray(df[y_col].values)
    event_ids = np.asarray(df["event_id"].values)

    # Normalize embeddings
    X = normalize(X, axis=1, norm="l2")

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_predictions = []
    fold_true_labels = []
    fold_test_indices = []
    fold_accuracies = []

    for train_idx, test_idx in skf.split(X, y):

        # Split
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        event_train, event_test = event_ids[train_idx], event_ids[test_idx]

        # Balance training set
        X_train, y_train, event_train, min_n = balance_training_set(
            X_train, y_train, event_train, rng
        )

        print(f"Set test set to {min_n}")

        # Predict
        preds = knn_predict_with_event_mask(
            X_test,
            X_train,
            y_train,
            event_test,
            event_train,
            k,
        )

        acc = np.mean(preds == y_test)

        fold_predictions.append(preds)
        fold_true_labels.append(y_test)
        fold_test_indices.append(test_idx)
        fold_accuracies.append(acc)

    return (
        fold_predictions,
        fold_true_labels,
        fold_test_indices,
        fold_accuracies,
    )