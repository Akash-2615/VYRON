"""Embeddings, gallery matching, and optional appearance tagging."""
from .appearance import AppearanceAnalyzer, body_box, body_crop, person_box_for_face
from .gallery import Gallery, normalize
from .recognizer import FaceEmbedder

__all__ = [
    "AppearanceAnalyzer",
    "FaceEmbedder",
    "Gallery",
    "body_box",
    "body_crop",
    "normalize",
    "person_box_for_face",
]
