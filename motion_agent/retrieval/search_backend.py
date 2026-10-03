"""Search backend abstraction for retrieval indices."""

from __future__ import annotations

import torch

from motion_agent.retrieval.schemas import SearchHit


class MatrixCosineSearchBackend:
    """Simple in-memory cosine search over normalized embeddings."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        if embeddings:
            matrix = torch.tensor(embeddings, dtype=torch.float32)
        else:
            matrix = torch.empty((0, 0), dtype=torch.float32)
        self.matrix = _normalize_rows(matrix)

    def search(self, query_embedding: list[float], top_n: int) -> list[SearchHit]:
        if self.matrix.numel() == 0:
            return []
        query = torch.tensor(query_embedding, dtype=torch.float32)
        query = query / query.norm().clamp_min(1e-12)
        scores = self.matrix @ query
        k = min(top_n, int(scores.numel()))
        values, indices = torch.topk(scores, k=k)
        return [
            SearchHit(index=int(index.item()), score=float(value.item()))
            for value, index in zip(values, indices)
        ]


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    return matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
