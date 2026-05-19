"""
core/anti_gaming.py
Keyword stuffing · unsupported claims · buzzwords · hidden text detection.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from core.parser import ParsedResume

_DATA = Path(__file__).parent.parent / "data"

def _load_lines(f: str) -> set[str]:
    p = _DATA / f
    return {l.strip().lower() for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()

_BUZZWORDS    = _load_lines("buzzwords.txt")
_EXPERT_RE    = re.compile(r"\b(expert|advanced|senior|lead|principal|architect|master|proficient)\s+in\s+([\w\s+#.]+)", re.IGNORECASE)
_INFLATED_RE  = re.compile(r"\b(\d{3,})\s*(%|percent|x|times|fold)\b", re.IGNORECASE)

@dataclass
class ClaimWarning:
    claim: str; reason: str; severity: str

@dataclass
class AntiGamingResult:
    keyword_stuffing:     bool               = False
    stuffing_score:       float              = 0.0
    buzzword_density:     float              = 0.0
    top_buzzwords:        list[str]          = field(default_factory=list)
    unsupported_claims:   list[ClaimWarning] = field(default_factory=list)
    suspicious_metrics:   list[str]          = field(default_factory=list)
    hidden_text_detected: bool               = False
    overall_risk:         str               = "Low"
    authenticity_score:   int               = 100

    def has_warnings(self) -> bool:
        return (self.keyword_stuffing or self.top_buzzwords
                or self.unsupported_claims or self.suspicious_metrics
                or self.hidden_text_detected)

def _detect_keyword_stuffing(text: str) -> tuple[bool, float]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    if not words: return False, 0.0
    stopwords = {"the","and","for","with","using","from","that","this","are","was","have"}
    counts = {w: c for w, c in Counter(words).items() if w not in stopwords and c > 2}
    if not counts: return False, 0.0
    score = round(min(max(counts.values()) / max(len(words), 1) * 10, 1.0), 3)
    return score > 0.12, score

def _detect_buzzwords(text: str) -> tuple[float, list[str]]:
    text_lower = text.lower()
    found = [bw for bw in _BUZZWORDS if bw in text_lower]
    return round(len(found) / max(len(text.split()), 1), 4), found[:8]

def _detect_unsupported_claims(parsed: ParsedResume) -> list[ClaimWarning]:
    evidence_blob = " ".join(parsed.projects + parsed.experience + parsed.internship).lower()
    warnings = []
    for match in _EXPERT_RE.finditer(parsed.raw_text):
        level = match.group(1).lower(); skill = match.group(2).strip().lower()
        if skill not in evidence_blob:
            severity = "high" if level in ("expert","senior","lead") else "medium"
            warnings.append(ClaimWarning(
                claim=match.group(0),
                reason=f"Claims '{level}' in '{skill}' but no supporting evidence in Projects/Experience.",
                severity=severity,
            ))
    return warnings[:6]

def _detect_suspicious_metrics(text: str) -> list[str]:
    flags = []
    for m in _INFLATED_RE.finditer(text):
        value = int(m.group(1)); unit = m.group(2)
        if unit in ("%","percent") and value > 300:
            flags.append(f"'{m.group(0)}' — very high percentage; add context.")
        elif unit.lower() in ("x","times","fold") and value > 50:
            flags.append(f"'{m.group(0)}' — extreme multiplier; verify accuracy.")
    return flags[:5]

def _detect_hidden_text(file_bytes: bytes) -> bool:
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("color") == 16777215 and len(span.get("text","").strip()) > 3:
                            doc.close(); return True
        doc.close()
    except Exception: pass
    return False

def _compute_authenticity(r: AntiGamingResult) -> int:
    s = 100
    if r.keyword_stuffing: s -= 25
    s -= int(r.buzzword_density * 200)
    s -= len([c for c in r.unsupported_claims if c.severity == "high"]) * 12
    s -= len([c for c in r.unsupported_claims if c.severity == "medium"]) * 6
    s -= len(r.suspicious_metrics) * 8
    if r.hidden_text_detected: s -= 40
    return max(0, min(100, s))

def analyse(parsed: ParsedResume, file_bytes: bytes | None = None) -> AntiGamingResult:
    r = AntiGamingResult()
    r.keyword_stuffing, r.stuffing_score    = _detect_keyword_stuffing(parsed.raw_text)
    r.buzzword_density, r.top_buzzwords     = _detect_buzzwords(parsed.raw_text)
    r.unsupported_claims                    = _detect_unsupported_claims(parsed)
    r.suspicious_metrics                    = _detect_suspicious_metrics(parsed.raw_text)
    if file_bytes: r.hidden_text_detected   = _detect_hidden_text(file_bytes)
    r.authenticity_score = _compute_authenticity(r)
    r.overall_risk = "High" if r.authenticity_score < 50 or r.hidden_text_detected else \
                     "Medium" if r.authenticity_score < 75 else "Low"
    return r