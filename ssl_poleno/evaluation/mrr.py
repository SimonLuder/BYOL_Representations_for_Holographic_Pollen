import numpy as np

def mean_reciprocal_rank(emb, labels, reduction="mean"):

    """
    emb: (N, D) numpy array of embeddings
    labels: (N,) numpy array of integer labels
    reduction: (str) "None" or "mean" 
    """

    # Normalize embeddings (L2 normalization)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norm + 1e-12)

    # Cosine similarity matrix
    sim = emb @ emb.T

    # Remove self-similarity
    np.fill_diagonal(sim, -np.inf)

    # Sort indices by similarity (descending)
    ranked_indices = np.argsort(-sim, axis=1)

    # Gather labels according to ranking
    ranked_labels = labels[ranked_indices]  # (N, N)

    # Find matches
    correct = ranked_labels == labels[:, None]

    # Index of first True (rank is 1-based)
    ranks = np.argmax(correct, axis=1) + 1

    # Handle cases with no correct match
    valid = np.any(correct, axis=1)

    reciprocal_ranks = np.zeros_like(ranks, dtype=float)
    reciprocal_ranks[valid] = 1.0 / ranks[valid].astype(float)

    if reduction == "mean":
        return reciprocal_ranks.mean()
    
    return reciprocal_ranks