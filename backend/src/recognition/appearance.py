"""Person appearance / clothing attributes via OpenCLIP zero-shot.

Given a body crop (face box expanded downward), returns structured attributes
and a short human-readable summary — used at registration / first sighting.
Heavy imports stay inside the class so tests can inject a fake analyzer.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Zero-shot label sets. Each group picks the single best match.
ATTRIBUTE_GROUPS: Dict[str, List[str]] = {
    "upper_color": [
        "a person wearing a white top", "a person wearing a black top",
        "a person wearing a blue top", "a person wearing a red top",
        "a person wearing a green top", "a person wearing a yellow top",
        "a person wearing a orange top", "a person wearing a brown top",
        "a person wearing a grey top", "a person wearing a pink top",
        "a person wearing a purple top",
    ],
    "upper_type": [
        "a person wearing a t-shirt", "a person wearing a shirt",
        "a person wearing a hoodie", "a person wearing a jacket",
        "a person wearing a coat", "a person wearing a sweater",
        "a person wearing a blouse", "a person wearing a dress",
        "a person wearing a sari", "a person wearing traditional clothing",
    ],
    "lower_type": [
        "a person wearing jeans", "a person wearing trousers",
        "a person wearing shorts", "a person wearing a skirt",
        "a person wearing a dress", "a person wearing a sari",
    ],
    "accessories": [
        "a person with no visible bag", "a person carrying a backpack",
        "a person carrying a shopping bag", "a person carrying a handbag",
        "a person wearing a hat", "a person wearing glasses",
    ],
    "hair": [
        "a person with short hair", "a person with long hair",
        "a person with bald head", "a person wearing a head covering",
    ],
}

_SHORT = {
    "a person wearing a white top": "white top",
    "a person wearing a black top": "black top",
    "a person wearing a blue top": "blue top",
    "a person wearing a red top": "red top",
    "a person wearing a green top": "green top",
    "a person wearing a yellow top": "yellow top",
    "a person wearing a orange top": "orange top",
    "a person wearing a brown top": "brown top",
    "a person wearing a grey top": "grey top",
    "a person wearing a pink top": "pink top",
    "a person wearing a purple top": "purple top",
    "a person wearing a t-shirt": "t-shirt",
    "a person wearing a shirt": "shirt",
    "a person wearing a hoodie": "hoodie",
    "a person wearing a jacket": "jacket",
    "a person wearing a coat": "coat",
    "a person wearing a sweater": "sweater",
    "a person wearing a blouse": "blouse",
    "a person wearing a dress": "dress",
    "a person wearing a sari": "sari",
    "a person wearing traditional clothing": "traditional wear",
    "a person wearing jeans": "jeans",
    "a person wearing trousers": "trousers",
    "a person wearing shorts": "shorts",
    "a person wearing a skirt": "skirt",
    "a person with no visible bag": None,
    "a person carrying a backpack": "backpack",
    "a person carrying a shopping bag": "shopping bag",
    "a person carrying a handbag": "handbag",
    "a person wearing a hat": "hat",
    "a person wearing glasses": "glasses",
    "a person with short hair": "short hair",
    "a person with long hair": "long hair",
    "a person with bald head": "bald",
    "a person wearing a head covering": "head covering",
}


def body_box(face_box, frame_shape) -> Optional[Tuple[int, int, int, int]]:
    """Expand a face box into an approximate full-person box (head → feet).

    A face is ~1/7–1/8 of standing height in typical CCTV, so we project a
    tall box downward from the face for clothing / appearance analysis + overlay.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in face_box[:4]]
    fh, fw = max(1.0, y2 - y1), max(1.0, x2 - x1)
    cx = (x1 + x2) / 2.0
    # ~7.5 face-heights tall, slightly above the hairline
    ny1 = max(0, int(y1 - 0.6 * fh))
    ny2 = min(h, int(y1 + 7.5 * fh))
    half = 1.8 * fw
    nx1 = max(0, int(cx - half))
    nx2 = min(w, int(cx + half))
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return None
    return nx1, ny1, nx2, ny2


def body_crop(frame: np.ndarray, face_box) -> Optional[np.ndarray]:
    """Expand a face box downward/sideways to capture torso + clothing."""
    box = body_box(face_box, frame.shape)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2].copy()


def _dominant_colors(bgr: np.ndarray, k: int = 3) -> List[str]:
    """Cheap OpenCV fallback / complement: top clothing colors by name."""
    small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    # Prefer lower 60% of crop (torso) over face/sky
    torso = small[int(64 * 0.35):, :, :]
    pixels = torso.reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        return []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    names = []
    for i in order:
        b, g, r = [int(v) for v in centers[i]]
        names.append(_bgr_to_color_name(b, g, r))
    # unique preserve order
    out = []
    for n in names:
        if n not in out:
            out.append(n)
    return out[:3]


def _bgr_to_color_name(b: int, g: int, r: int) -> str:
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 40:
        return "black"
    if mn > 200:
        return "white"
    if mx - mn < 30:
        return "grey"
    if r > g and r > b:
        return "red" if r - max(g, b) > 40 else "orange" if g > 100 else "red"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        return "blue"
    if r > 150 and g > 150 and b < 100:
        return "yellow"
    if r > 120 and b > 120 and g < 100:
        return "purple"
    if r > 150 and g > 80 and b > 80:
        return "pink"
    return "brown"


class AppearanceAnalyzer:
    """OpenCLIP zero-shot clothing / appearance tags + dominant-color fallback."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.model_name = cfg.get("model", "ViT-B-32")
        self.pretrained = cfg.get("pretrained", "openai")
        self.min_side = int(cfg.get("min_crop_px", 48))
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cpu"
        self._text_features: Dict[str, Any] = {}
        if self.enabled:
            self._load(cfg.get("device", "auto"))

    def _load(self, device: str) -> None:
        import torch
        import open_clip

        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained)
        model.eval().to(self._device)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        # Precompute text features per group
        with torch.no_grad():
            for group, prompts in ATTRIBUTE_GROUPS.items():
                tokens = self._tokenizer(prompts).to(self._device)
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                self._text_features[group] = (prompts, feats)
        log.info("OpenCLIP appearance model ready (%s/%s on %s)",
                 self.model_name, self.pretrained, self._device)

    def facing_away(self, crop: np.ndarray) -> Tuple[bool, float]:
        """CLIP zero-shot: True if the person is seen from behind / facing away."""
        if self._model is None or crop is None or crop.size == 0:
            return True, 0.5  # no CLIP → trust the "no face" heuristic
        import torch
        from PIL import Image
        prompts = [
            "a person facing the camera",
            "a person seen from the front",
            "a person seen from behind",
            "a person facing away from the camera",
            "the back of a person walking away",
        ]
        tokens = self._tokenizer(prompts).to(self._device)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = self._preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_features = self._model.encode_image(img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_feats = self._model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            sims = (100.0 * image_features @ text_feats.T).softmax(dim=-1)[0]
        away_score = float((sims[2] + sims[3] + sims[4]).item())
        toward_score = float((sims[0] + sims[1]).item())
        return away_score >= toward_score, away_score

    def describe(self, frame: np.ndarray, face_box=None, person_box=None) -> Optional[Dict[str, Any]]:
        """Describe clothing. Prefer an explicit person_box (full body / back view)."""
        if person_box is not None:
            x1, y1, x2, y2 = [int(v) for v in person_box[:4]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None
        elif face_box is not None:
            crop = body_crop(frame, face_box)
        else:
            return None
        return self.describe_crop(crop)

    def describe_crop(self, crop: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if min(h, w) < self.min_side:
            return None
        colors = _dominant_colors(crop)
        attrs: Dict[str, Any] = {"dominant_colors": colors, "model": f"open_clip:{self.model_name}"}
        scores: Dict[str, float] = {}

        if self._model is not None:
            import torch
            from PIL import Image
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = self._preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self._device)
            with torch.no_grad():
                image_features = self._model.encode_image(img)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                for group, (prompts, text_feats) in self._text_features.items():
                    sims = (100.0 * image_features @ text_feats.T).softmax(dim=-1)[0]
                    best = int(sims.argmax().item())
                    conf = float(sims[best].item())
                    label = prompts[best]
                    short = _SHORT.get(label)
                    if short is not None:
                        attrs[group] = short
                        scores[group] = round(conf, 3)
        elif colors:
            attrs["upper_color"] = f"{colors[0]} top"

        parts = []
        for key in ("upper_color", "upper_type", "lower_type", "accessories", "hair"):
            if key in attrs and attrs[key]:
                parts.append(str(attrs[key]))
        if not parts and colors:
            parts.append(f"{'/'.join(colors)} clothing")
        if not parts:
            return None
        summary = ", ".join(parts)
        attrs["summary"] = summary
        attrs["scores"] = scores
        return attrs
