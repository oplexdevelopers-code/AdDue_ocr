"""PaddleOCR wrapper with digital PDF fast-path and image pre-resizing.

File path in, list of {text, confidence, bbox, page} out.
"""
from functools import lru_cache
import os
from typing import Any, Dict, List


@lru_cache(maxsize=1)
def _get_engine():
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_textline_orientation=False,
        enable_mkldnn=True,
        lang="en",
    )


def preprocess_image(file_path: str, max_size: int = 1600) -> str:
    """Resizes high-res images to max_size before OCR to drastically improve speed."""
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                resized_path = file_path + "_resized.jpg"
                img.save(resized_path, "JPEG", quality=85)
                return resized_path
    except Exception:
        pass
    return file_path


def extract_text(file_path: str) -> List[Dict[str, Any]]:
    # 1. Fast-path: Check if digital PDF with embedded text
    if file_path.lower().endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            digital_lines: List[Dict[str, Any]] = []
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 30:
                    for line in text.splitlines():
                        clean_line = line.strip()
                        if clean_line:
                            digital_lines.append({
                                "text": clean_line,
                                "confidence": 1.0,
                                "bbox": None,
                                "page": page_idx + 1,
                            })
            if len(digital_lines) >= 3:
                return digital_lines
        except Exception:
            pass

    # 2. Image pre-resizing for fast OCR
    ocr_input_path = preprocess_image(file_path)
    lines: List[Dict[str, Any]] = []

    try:
        engine = _get_engine()
        results = engine.predict(ocr_input_path)

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
    finally:
        if ocr_input_path != file_path and os.path.exists(ocr_input_path):
            os.remove(ocr_input_path)

    return lines
