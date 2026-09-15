"""Standalone OCR microservice for AdDue.

Single responsibility: given an uploaded image or PDF, run PaddleOCR and
return the extracted text. No classification, no field extraction, no AI
reasoning — that happens downstream (see backend/internal/ai/ocr_provider.go,
which sends the returned text to a local Ollama model).
"""
import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from ocr import extract_text

app = FastAPI(title="AdDue OCR Service", version="1.0.0")


class OCRLine(BaseModel):
    text: str
    confidence: float
    bbox: Optional[List[int]] = None
    page: int = 1


class OCRResponse(BaseModel):
    text: str
    lines: List[OCRLine]
    engine: str = "paddleocr"


@app.get("/healthz")
async def health_check():
    return {"status": "ok", "app": "AdDue OCR Service"}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        lines = extract_text(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
    finally:
        os.remove(temp_path)

    full_text = "\n".join(l["text"] for l in lines)
    return OCRResponse(text=full_text, lines=lines)
