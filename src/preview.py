"""Publish the latest annotated frame for the dashboard live view."""
from pathlib import Path

import cv2


def publish_preview(frame, path: str, quality: int = 55, max_width: int = 960) -> None:
    """Atomically write a downscaled JPEG preview (keeps the live dashboard snappy)."""
    if not path:
        return
    vis = frame
    if max_width and vis.shape[1] > max_width:
        scale = max_width / float(vis.shape[1])
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV picks the encoder from the suffix — keep a real image extension on the temp file.
    tmp = p.parent / f".{p.stem}.tmp{p.suffix or '.jpg'}"
    if not cv2.imwrite(str(tmp), vis, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]):
        tmp.unlink(missing_ok=True)
        return
    tmp.replace(p)
