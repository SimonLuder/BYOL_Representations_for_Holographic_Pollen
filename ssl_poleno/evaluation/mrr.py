import numpy as np

# def mean_reciprocal_rank(emb, labels, reduction="mean"):

#     """
#     emb: (N, D) numpy array of embeddings
#     labels: (N,) numpy array of integer labels
#     reduction: (str) "None" or "mean" 
#     """

#     # Normalize embeddings (L2 normalization)
#     norm = np.linalg.norm(emb, axis=1, keepdims=True)
#     emb = emb / (norm + 1e-12)

#     # Cosine similarity matrix
#     sim = emb @ emb.T

#     # Remove self-similarity
#     np.fill_diagonal(sim, -np.inf)

#     # Sort indices by similarity (descending)
#     ranked_indices = np.argsort(-sim, axis=1)

#     # Gather labels according to ranking
#     ranked_labels = labels[ranked_indices]  # (N, N)

#     # Find matches
#     correct = ranked_labels == labels[:, None]

#     # Index of first True (rank is 1-based)
#     ranks = np.argmax(correct, axis=1) + 1

#     # Handle cases with no correct match
#     valid = np.any(correct, axis=1)

#     reciprocal_ranks = np.zeros_like(ranks, dtype=float)
#     reciprocal_ranks[valid] = 1.0 / ranks[valid].astype(float)

#     if reduction == "mean":
#         return reciprocal_ranks.mean()
    
#     return reciprocal_ranks


def mean_reciprocal_rank(
    emb,
    labels,
    reduction="mean",
    batch_size=512,
):
    """
    Memory-efficient MRR computation.

    emb: (N, D) numpy array
    labels: (N,)
    reduction: "mean" or "None"
    batch_size: number of queries processed at once
    """

    N = emb.shape[0]

    # Normalize embeddings
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norm + 1e-12)

    reciprocal_ranks = np.zeros(N, dtype=float)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        query_emb = emb[start:end]  # (B, D)

        # Compute similarity to all embeddings
        sim = query_emb @ emb.T  # (B, N)

        # Remove self similarity
        for i in range(start, end):
            sim[i - start, i] = -np.inf

        # Rank
        ranked_indices = np.argsort(-sim, axis=1)

        ranked_labels = labels[ranked_indices]

        correct = ranked_labels == labels[start:end, None]

        ranks = np.argmax(correct, axis=1) + 1
        valid = np.any(correct, axis=1)

        reciprocal_ranks[start:end][valid] = 1.0 / ranks[valid]

    if reduction == "mean":
        return reciprocal_ranks.mean()

    return reciprocal_ranks


def calc_mrr_pd(df, emb_col="emb", lbl_col="event_id"):
    """Calculate mean reciprocal rank (MRR) for a pandas DataFrame with embeddings and labels.

    Args:
        df (pd.DataFrame): Metrics DataFrame containing embeddings and labels.
        emb_col (str, optional): Embedding column name. Defaults to "emb".
        lbl_col (str, optional): Label column name. Defaults to "event_id".

    Returns:
        Mean reciprocal rank (MRR) value.
    """
    emb = np.vstack(df[emb_col].values)
    labels = df[lbl_col].values
    return mean_reciprocal_rank(emb, labels, reduction="mean")
