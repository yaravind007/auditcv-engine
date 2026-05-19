"""
core/extractor.py
Layered PDF extraction: PyMuPDF → pdfplumber. OCR excluded (HF Spaces incompatible).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from io import BytesIO

@dataclass
class ExtractionResult:
    text:       str
    method:     str
    confidence: float
    pages:      int
    char_count: int
    warnings:   list[str] = field(default_factory=list)

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.75: return "High"
        if self.confidence >= 0.45: return "Medium"
        return "Low"

    @property
    def ok(self) -> bool:
        return self.method != "failed" and self.char_count >= 80

_SECTION_KEYWORDS = {
    "education","skills","experience","projects","certifications",
    "achievements","summary","objective","internship","work","awards"
}
_GARBLE_RE = re.compile(r"[^\x00-\x7F]{4,}")

def _score_extraction(text: str) -> tuple[float, list[str]]:
    warnings: list[str] = []
    score = 0.0
    if not text or len(text) < 80:
        return 0.0, ["Extracted text too short — PDF may be image-only or encrypted."]
    chars = len(text)
    if chars >= 800:   score += 0.30
    elif chars >= 400: score += 0.20
    else:
        score += 0.10
        warnings.append("Very short resume text — some content may not have been extracted.")
    text_lower = text.lower()
    found = sum(1 for kw in _SECTION_KEYWORDS if kw in text_lower)
    score += min(found / 4, 1.0) * 0.30
    if found < 2:
        warnings.append("Fewer than 2 section headers found — check PDF formatting.")
    words = text.split()
    real_words = [w for w in words if re.match(r"^[a-zA-Z]{2,}$", w)]
    if words:
        word_ratio = len(real_words) / len(words)
        score += word_ratio * 0.25
        if word_ratio < 0.4:
            warnings.append("High proportion of non-word tokens — encoding issues possible.")
    garble_matches = len(_GARBLE_RE.findall(text))
    if garble_matches == 0:   score += 0.15
    elif garble_matches < 5:
        score += 0.08
        warnings.append(f"{garble_matches} garbled character sequence(s) detected.")
    else:
        warnings.append("Heavy garbling — try exporting PDF from a word processor.")
    return round(min(score, 1.0), 3), warnings

def _extract_pymupdf(raw: bytes) -> tuple[str, int]:
    import fitz
    doc = fitz.open(stream=raw, filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages).strip(), len(pages)

def _extract_pdfplumber(raw: bytes) -> tuple[str, int]:
    import pdfplumber
    with pdfplumber.open(BytesIO(raw)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
        n = len(pdf.pages)
    return "\n".join(pages).strip(), n

def extract(file_bytes: bytes, filename: str = "resume.pdf") -> ExtractionResult:
    text = ""; method = "failed"; pages = 0; warns: list[str] = []
    try:
        text, pages = _extract_pymupdf(file_bytes)
        method = "pymupdf"
    except Exception as exc:
        warns.append(f"PyMuPDF failed ({exc}), trying pdfplumber.")
    if method == "failed" or len(text) < 80:
        try:
            text2, pages2 = _extract_pdfplumber(file_bytes)
            if len(text2) > len(text):
                text, pages, method = text2, pages2, "pdfplumber"
        except Exception as exc:
            warns.append(f"pdfplumber also failed ({exc}).")
    if not text or len(text) < 80:
        return ExtractionResult(
            text="", method="failed", confidence=0.0, pages=0, char_count=0,
            warnings=warns + ["Could not extract text. Upload a text-based PDF."],
        )
    conf, conf_warns = _score_extraction(text)
    return ExtractionResult(
        text=text, method=method, confidence=conf,
        pages=pages, char_count=len(text), warnings=warns + conf_warns,
    )