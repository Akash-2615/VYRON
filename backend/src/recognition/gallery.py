"""In-memory embedding gallery with cosine-similarity matching."""
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np


class Gallery:
    def __init__(self, max_per_face: int = 5):
        self.max_per_face = max_per_face
        self._ids: list = []
        self._vecs: list = []
        self._count = defaultdict(int)
        self._mat: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, face_id: int, emb: np.ndarray) -> bool:
        if self._count[face_id] >= self.max_per_face:
            return False
        self._ids.append(face_id)
        self._vecs.append(emb.astype(np.float32))
        self._count[face_id] += 1
        self._mat = None
        return True

    def match(self, emb: np.ndarray) -> Tuple[Optional[int], float]:
        """Best (face_id, cosine similarity). (None, -1.0) when the gallery is empty."""
        if not self._ids:
            return None, -1.0
        if self._mat is None:
            self._mat = np.stack(self._vecs)
        sims = self._mat @ emb
        i = int(np.argmax(sims))
        return self._ids[i], float(sims[i])

    def similarity_to(self, face_id: int, emb: np.ndarray) -> float:
        """Best cosine similarity of ``emb`` against stored vectors for ``face_id``."""
        if not self._ids:
            return -1.0
        if self._mat is None:
            self._mat = np.stack(self._vecs)
        idxs = [i for i, fid in enumerate(self._ids) if int(fid) == int(face_id)]
        if not idxs:
            return -1.0
        sims = self._mat[idxs] @ emb.astype(np.float32)
        return float(np.max(sims))


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / max(float(np.linalg.norm(v)), 1e-9)).astype(np.float32)
