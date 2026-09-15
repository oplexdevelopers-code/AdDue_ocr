"""PaddleOCR wrapper: file path in, list of {text, confidence, bbox, page} out.

Adapted from the extract.py CLI script — same PaddleOCR call, same result
parsing (PaddleOCR 3.x's predict() returns dict-like pages with rec_texts /
rec_scores / rec_polys, not the old ocr()-tuple format).
"""
from functools import lru_cache
from typing import Any, Dict, List


@lru_cache(maxsize=1)
def _get_engine():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_textline_orientation=True, lang="en")


def extract_text(file_path: str) -> List[Dict[str, Any]]:
    engine = _get_engine()
    results = engine.predict(file_path)

    lines: List[Dict[str, Any]] = []
    for page_idx, page in enumerate(results):
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polys = page.get("rec_polys", page.get("dt_polys", []))

        for i, text in enumerate(texts):
            confidence = float(scores[i]) if i < len(scores) else 0.0
            bbox = None
            if i < len(polys) and polys[i] is not None and len(polys[i]) > 0:
                poly = polys[i]
                x_coords = [p[0] for p in poly]
                y_coords = [p[1] for p in poly]
                bbox = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]

            lines.append({
                "text": text,
                "confidence": confidence,
                "bbox": bbox,
                "page": page_idx + 1,
            })

    return lines
