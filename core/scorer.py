"""
core/scorer.py
────────────────────────
Comprehensive Multi-Parameter Evaluation Engine for AuditCV v2.
Enforces semantic alias mapping, timeline continuity tracking, and multi-factor impact validation.
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from core.parser import ParsedResume
from core.anti_gaming import AntiGamingResult
from core.embeddings import EmbeddingResult

_DATA = Path(__file__).parent.parent / "data"


def _load_verbs() -> set[str]:
    p = _DATA / "action_verbs.txt"
    return {l.strip().lower() for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()


_ACTION_VERBS = _load_verbs()
_NUMBER_RE = re.compile(r"\b\d+[\d,.]*\s*(%|percent|x|k|m|ms|s|gb|mb|hrs?|days?|weeks?|months?)?\b", re.IGNORECASE)
_PASSIVE_RE = re.compile(r"\b(was|were|is|are|been|being)\s+\w+ed\b", re.IGNORECASE)

# Strict Date Parsing Regex to Extract Chronological Lifecycles
_DATE_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december|\d{1,2})?[-/\s.]*(\d{4})\s*[-–—to]+\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december|\d{1,2})?[-/\s.]*(\d{4}|present|current)\b",
    re.IGNORECASE
)

# ── Semantic Alias Mapping Engine ─────────────────────────────────────────────
_CANONICAL_MAP: dict[str, str] = {
    "structured query language": "sql", "mysql": "sql", "postgresql": "sql", "sqlite": "sql", "tsql": "sql",
    "advanced excel": "excel", "ms excel": "excel", "microsoft excel": "excel",
    "js": "javascript", "node": "javascript", "node.js": "javascript", "react.js": "javascript", "react": "javascript",
    "amazon web services": "aws", "google cloud platform": "gcp", "google cloud": "gcp", "microsoft azure": "azure",
    "scikit-learn": "machine learning", "sklearn": "machine learning", "tensorflow": "machine learning",
    "pytorch": "machine learning"
}


def _canonical_skill(skill: str) -> str:
    s = skill.lower().strip()
    return _CANONICAL_MAP.get(s, s)


def _label(s: int) -> str:
    return "Strong" if s >= 80 else "Good" if s >= 60 else "Needs Work" if s >= 40 else "Weak"


def _score_range(s: int) -> str:
    lo = max(0, (s // 10) * 10 - 5);
    hi = min(100, (s // 10) * 10 + 5)
    return f"{lo}–{hi}"


@dataclass
class DimScore:
    name: str
    weight: float
    score: int
    confidence: str
    evidence: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ScoringResult:
    dimensions: list[DimScore]
    overall_score: int
    overall_range: str
    overall_label: str
    classification: str
    is_transition: bool
    transition_score: int | None = None
    missing_critical: list[str] = field(default_factory=list)

    def dim(self, name: str) -> DimScore | None:
        return next((d for d in self.dimensions if d.name == name), None)


# ── Chronological Lifecycles Engine ───────────────────────────────────────────
def _calculate_total_experience(parsed: ParsedResume) -> tuple[float, list[str]]:
    """
    Extracts and merges all historical timeline chunks from text blobs.
    Enforces distinct overlapping bounds tracking.
    """
    blobs = parsed.experience + parsed.internship + parsed.projects
    text_pool = " ".join(blobs)
    matches = _DATE_RE.findall(text_pool)

    intervals: list[tuple[int, int]] = []
    evidence: list[str] = []

    # Simple month conversion utility
    months_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
                  "nov": 11, "dec": 12}

    for start_m, start_y, end_m, end_y in matches:
        try:
            s_year = int(start_y)
            s_month = months_map.get(start_m.lower()[:3], 1) if start_m else 1
            start_total_months = (s_year * 12) + s_month

            if end_y.lower() in ["present", "current"]:
                e_year, e_month = 2026, 5  # Anchored to system simulation frame coordinates
            else:
                e_year = int(end_y)
                e_month = months_map.get(end_m.lower()[:3], 1) if end_m else 1
            end_total_months = (e_year * 12) + e_month

            if end_total_months >= start_total_months:
                intervals.append((start_total_months, end_total_months))
                evidence.append(f"Detected period: {start_m or ''} {start_y} to {end_m or end_y}")
        except Exception:
            continue

    if not intervals:
        return 0.0, ["No chronological dates found. Defaulting to 0 years entry tier."]

    # Merge overlapping timeline boundaries
    intervals.sort(key=lambda x: x[0])
    merged: list[tuple[int, int]] = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)

    total_months = sum((end - start) for start, end in merged)
    return round(total_months / 12, 1), evidence


# ── Dimension 1: ATS Compatibility ────────────────────────────────────────────
def _evaluate_ats(parsed: ParsedResume) -> DimScore:
    pts = 0;
    ev = [];
    pen = []

    # Map section normalization aliases across structural criteria
    norm_sections = [s.lower() for s in parsed.sections_found]
    has_projects = any(s in norm_sections for s in ["projects", "project experience"])
    has_skills = any(s in norm_sections for s in ["skills", "technical skills", "soft skills"])

    if parsed.name:    pts += 15; ev.append(f"Name identified: {parsed.name}")
    if parsed.contact.get("email"): pts += 15; ev.append("Contact channel: Email present")
    if parsed.contact.get("phone"): pts += 15; ev.append("Contact channel: Phone present")
    if has_projects:   pts += 20; ev.append("Section layout verification: Projects block confirmed")
    if has_skills:     pts += 20; ev.append("Section layout verification: Skills block confirmed")
    if parsed.word_count >= 250: pts += 15; ev.append(f"Document density target hit: {parsed.word_count} words")

    raw = min(100, pts)
    return DimScore(
        name="ATS Compatibility Analysis", weight=0.10, score=raw,
        confidence="High" if raw >= 70 else "Medium", evidence=ev,
        reasoning="Evaluates structural consistency, reliable parse boundaries, and file accessibility."
    )


# ── Dimension 2: Skill Evidence Engine ────────────────────────────────────────
def _evaluate_skills(parsed: ParsedResume, emb: EmbeddingResult) -> DimScore:
    ev = [];
    pts = 0

    # Map and unify text tokens to canonical base frames to prevent evidence dropping
    canonical_skills = {_canonical_skill(s) for s in parsed.skills}
    text_blob_lower = parsed.raw_text.lower()

    verified_tokens = []
    for s in parsed.skills:
        c_skill = _canonical_skill(s)
        # Check if skill or synonym maps straight into contextual experience/projects text pool
        if c_skill in text_blob_lower or s.lower() in text_blob_lower:
            verified_tokens.append(c_skill)

    evidence_ratio = len(verified_tokens) / max(1, len(parsed.skills))
    pts += int(evidence_ratio * 60)

    if evidence_ratio >= 0.75:
        ev.append(f"Semantic match high: {int(evidence_ratio * 100)}% of skills supported by context arrays")
    else:
        ev.append(
            f"Partial semantic map match: {len(parsed.skills) - len(verified_tokens)} skills lack project mapping")

    # Reward skill deployment depth across experience
    if len(canonical_skills) >= 8:
        pts += 25; ev.append("Skill volume footprint: Broad (8+ standalone tools)")
    elif len(canonical_skills) >= 4:
        pts += 15; ev.append("Skill volume footprint: Moderate (4-7 standalone tools)")
    else:
        pts += 5

    if emb.support_ratio > 0.5:        pts += 15; ev.append("Vector embedding matrix confirms validation support")

    raw = min(100, pts)
    return DimScore(
        name="Skill Evidence Engine", weight=0.20, score=raw,
        confidence="High" if len(parsed.skills) > 0 else "Low", evidence=ev,
        reasoning="Measures semantic tool mappings and cross-section validation context."
    )


# ── Dimension 3: Project Quality Analysis ──────────────────────────────────────
def _evaluate_projects(parsed: ParsedResume) -> DimScore:
    pts = 0;
    ev = [];
    pen = []

    if not parsed.projects:
        return DimScore("Project Quality Analysis", 0.20, 0, "Low", ["No projects block identified"], "Critical gap.")

    proj_blob = " ".join(parsed.projects).lower()

    # 1. Verify Deployment or Infrastructure Tracking (Do NOT just look for numbers)
    has_github = any(kw in proj_blob for kw in ["github", "repo", "repository", "git"])
    has_hosting = any(
        kw in proj_blob for kw in ["live dashboard", "deployed", "netlify", "vercel", "aws link", "live app"])

    if Skinner := sum(
            1 for ind in ["architecture", "pipeline", "etl", "database", "scale", "dataset"] if ind in proj_blob):
        pts += min(Skinner * 10, 30)
        ev.append(f"Architectural depth signals verified inside text body ({Skinner} markers)")

    if has_github:  pts += 25; ev.append("Artifact check: GitHub tracking repositories linked")
    if has_hosting: pts += 25; ev.append("Artifact check: Live application deployment verification links present")

    # Cross check action verb density frames
    verb_hits = sum(1 for v in _ACTION_VERBS if re.search(rf"\b{re.escape(v)}\b", proj_blob))
    pts += min(verb_hits * 4, 20)
    if verb_hits >= 4: ev.append(f"Execution tracking: {verb_hits} clear action verbs confirmed inside project context")

    raw = min(100, pts)
    return DimScore(
        name="Project Quality Analysis", weight=0.20, score=raw,
        confidence="High" if (has_github or has_hosting) else "Medium", evidence=ev,
        reasoning="Scores architectural framework complexity, dataset scale mapping, and validation reproducibility profiles."
    )


# ── Dimension 4: Impact Validation ────────────────────────────────────────────
def _evaluate_impact(parsed: ParsedResume) -> DimScore:
    pts = 0;
    ev = []

    all_bullets = []
    for blob in parsed.experience + parsed.internship + parsed.projects:
        all_bullets.extend(l.strip("•·–- ").strip() for l in blob.splitlines() if l.strip())

    if not all_bullets:
        return DimScore("Impact Validation", 0.15, 0, "Low", ["No performance bullets isolated"],
                        "Metrics context zero.")

    strong_count = 0;
    med_count = 0;
    weak_count = 0

    for b in all_bullets:
        has_metric = bool(_NUMBER_RE.search(b))
        # Check for Method/Strategy (e.g., "using...", "via...", "by optimization")
        has_method = any(
            kw in b.lower() for kw in ["using", "via", "through", "by designing", "optimization", "implemented"])
        # Check for Context/Scale scale markers (e.g., names of tools, scope size "across 50k", "in Power BI")
        has_context = len(b.split()) > 12 and any(
            kw in b.lower() for kw in ["source", "request", "user", "dashboard", "database", "report", "server"])

        if has_metric and has_method and has_context:
            strong_count += 1
        elif has_metric and has_method:
            med_count += 1
        elif has_metric:
            weak_count += 1

    pts += (strong_count * 25) + (med_count * 12) + (weak_count * 4)
    raw = min(100, max(5, pts))

    ev.append(f"Isolated {strong_count} High-Fidelity Impact statements (Metric + Method + Context)")
    ev.append(
        f"Isolated {med_count} Mid-Fidelity Impact statements (Metric + Method without clear structural scale context)")
    ev.append(f"Isolated {weak_count} Raw Percentages (Unsupported raw numbers missing execution strategies)")

    return DimScore(
        name="Impact Validation", weight=0.15, score=raw,
        confidence="High" if strong_count >= 2 else "Medium", evidence=ev,
        reasoning="Validates quantified outcomes strictly checking for methodology and scaling context to filter empty assertions."
    )


# ── Dimension 5: Authenticity & Trust Analysis ────────────────────────────────
def _evaluate_authenticity(ag: AntiGamingResult) -> DimScore:
    # Scale base score from baseline authenticity scores directly
    raw = ag.authenticity_score
    ev = []

    if ag.keyword_stuffing:   ev.append(
        f"Anomalous layout density: Keyword density trigger detected ({ag.stuffing_score:.2f})")
    if ag.hidden_text_detected: ev.append(
        "Malicious exploit flag: Hidden text artifacts matching background detected inside canvas layers")
    if len(ag.top_buzzwords) > 3: ev.append(
        f"Template phrasing density high: Found buzzwords ({', '.join(ag.top_buzzwords[:3])})")

    if raw >= 80: ev.append("Profile compliance checks completely clear. High human structural integrity signature.")

    return DimScore(
        name="Authenticity and Trust Analysis", weight=0.10, score=raw,
        confidence="High", evidence=ev,
        reasoning="Audits hidden text artifacts, pattern template repetitions, and buzzword distribution profiles."
    )


# ── Global Router Orchestration ───────────────────────────────────────────────
def score(parsed: ParsedResume, ag: AntiGamingResult, emb: EmbeddingResult,
          extraction_confidence: float) -> ScoringResult:
    """
    Main orchestration loop. Compiles the strict multi-factor dimensional scoring array.
    """
    # 1. Run Chronological Lifecycles calculations to determine career classification
    years_exp, date_evidence = _calculate_total_experience(parsed)

    if years_exp >= 7.0:
        tier = "Senior Professional"
    elif years_exp >= 3.0:
        tier = "Experienced Professional"
    elif years_exp >= 1.0:
        tier = "Early Career Professional"
    else:
        tier = "Fresh Graduate"

    # 2. Transition Tracking Logic Block
    text_lower = parsed.raw_text.lower()
    cs_indicators = ["computer science", "software", "developer", "data science", "data analyst", "systems", "networks"]
    has_cs_academic = any(
        ind in text_lower for ind in ["b.sc cs", "m.sc cs", "mca", "b.tech computer", "b.e. computer"])

    # Check if they are jumping tracks into tech from another engineering/science segment
    non_cs_track = any(kw in text_lower for kw in
                       ["biomedical", "pharmacy", "civil", "mechanical", "commerce", "accounting", "nursing",
                        "hospital"])

    is_transition = False
    transition_score = None
    if non_cs_track and not has_cs_academic:
        is_transition = True
        tier = f"{tier} (Career Transition Candidate)"

        # Calculate transition readiness factor: Do they possess transferable tech skills?
        tech_skills = ["python", "sql", "power bi", "excel", "javascript", "mysql", "tableau"]
        hits = sum(1 for s in tech_skills if s in text_lower)
        transition_score = min(100, hits * 20)

    # 3. Assemble Evaluation Dimensions
    dim_ats = _evaluate_ats(parsed)
    dim_skills = _evaluate_skills(parsed, emb)
    dim_proj = _evaluate_projects(parsed)
    dim_impact = _evaluate_impact(parsed)
    dim_auth = _evaluate_authenticity(ag)

    # 4. Mathematical Weighted Aggregation Loop
    dims = [dim_ats, dim_skills, dim_proj, dim_impact, dim_auth]

    # Normalize baseline weights to equal 100% distribution metrics
    total_weight = sum(d.weight for d in dims)
    overall = int(sum((d.score * d.weight) for d in dims) / total_weight)

    missing = parsed.sections_missing[:]
    if not parsed.contact.get("email"): missing.append("email address")

    return ScoringResult(
        dimensions=dims,
        overall_score=overall,
        overall_range=_score_range(overall),
        overall_label=_label(overall),
        classification=tier,
        is_transition=is_transition,
        transition_score=transition_score,
        missing_critical=missing
    )