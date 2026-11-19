"""Matching faces by their identity embeddings.

An embedding is the 128-number "face print" OpenCV's SFace model computes
from an aligned face. Two embeddings of the same person point in nearly the
same direction, so they are compared by cosine similarity: 1.0 means
identical, around 0 means unrelated people.
"""

from __future__ import annotations

import numpy as np

# OpenCV's published cosine threshold for SFace (99.60% accuracy on the LFW
# benchmark). Raise it to reject more look-alikes, at the cost of turning
# away the right person more often.
DEFAULT_THRESHOLD = 0.363


def normalize(vector):
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    length = float(np.linalg.norm(v))
    if not np.isfinite(length) or length == 0:
        raise ValueError("an embedding must be a non-zero vector")
    return v / length


def similarity(a, b):
    """Cosine similarity of two embeddings, from -1 to 1."""
    return float(np.dot(normalize(a), normalize(b)))


def mean_embedding(vectors):
    """One embedding that stands for several samples of the same face."""
    return normalize(np.mean([normalize(v) for v in vectors], axis=0))


def rank(query, gallery):
    """Score every person in the gallery against the query, best first.

    gallery: {name: (n, 128) array of that person's samples}. A person's
    score is their best-matching sample.
    """
    q = normalize(query)
    scores = []
    for name, samples in gallery.items():
        stored = np.asarray(samples, dtype=np.float32).reshape(-1, q.size)
        stored = stored / np.linalg.norm(stored, axis=1, keepdims=True)
        scores.append((name, float(np.max(stored @ q))))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def decide(query, gallery, threshold=DEFAULT_THRESHOLD, claimed=None):
    """Who is this? Or, with `claimed`, is this that person?

    Returns a JSON-ready dict: mode ("identify" or "verify"), matched, name
    (the matched person, or None), similarity, threshold and the top
    candidates.
    """
    if not gallery:
        raise ValueError("nobody is registered")
    ranking = rank(query, gallery)
    if claimed is None:
        best, score = ranking[0]
    else:
        names = {name.casefold(): name for name in gallery}
        if claimed.casefold() not in names:
            raise KeyError(claimed)
        best = names[claimed.casefold()]
        score = dict(ranking)[best]
    matched = score >= threshold
    return {
        "mode": "identify" if claimed is None else "verify",
        "claimed_name": claimed,
        "matched": matched,
        "name": best if matched else None,
        "similarity": round(score, 4),
        "threshold": threshold,
        "candidates": [{"name": n, "similarity": round(s, 4)} for n, s in ranking[:3]],
    }
