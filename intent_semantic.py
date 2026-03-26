"""Semantic intent classification with SentenceTransformer (cosine similarity to example phrases)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

INTENT_ASK_BOTTLE_POSITION = "ASK_BOTTLE_POSITION"

INTENT_EXAMPLES: dict[str, list[str]] = {
    INTENT_ASK_BOTTLE_POSITION: [
        "where is the bottle",
        "can you tell me where the bottle is",
    ],
    "GREETING": ["hello", "hi there"],
}

SIMILARITY_THRESHOLD = 0.6

_model: SentenceTransformer | None = None
_intent_embs: dict[str, np.ndarray] | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_intent_embeddings() -> dict[str, np.ndarray]:
    global _intent_embs
    if _intent_embs is None:
        m = _get_model()
        _intent_embs = {intent: m.encode(sents) for intent, sents in INTENT_EXAMPLES.items()}
    return _intent_embs


def classify_intent_semantic(text: str) -> str:
    if not text or not text.strip():
        return "UNKNOWN"
    emb = _get_model().encode(text.strip())
    intent_embs = _get_intent_embeddings()
    best_intent, best_score = "UNKNOWN", -1.0
    emb_norm = np.linalg.norm(emb) + 1e-8
    for intent, ex_embs in intent_embs.items():
        sims = np.dot(ex_embs, emb) / (np.linalg.norm(ex_embs, axis=1) * emb_norm)
        score = float(np.max(sims))
        if score > best_score:
            best_intent, best_score = intent, score
    return best_intent if best_score > SIMILARITY_THRESHOLD else "UNKNOWN"
