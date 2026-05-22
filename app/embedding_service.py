from __future__ import annotations

import logging

import requests

from app.config import TEI_URL

logger = logging.getLogger(__name__)

BATCH_SIZE = 8


def embed_text(text: str) -> list[float]:
    vecs = embed_batch([text])
    return vecs[0] if vecs else []


def embed_batch(texts: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i : i + BATCH_SIZE]
        try:
            resp = requests.post(TEI_URL, json={"inputs": chunk}, timeout=60)
            resp.raise_for_status()
            all_embeddings.extend(resp.json())
        except requests.RequestException:
            logger.exception("TEI embedding failed for batch starting at index %d", i)
            raise
    return all_embeddings
