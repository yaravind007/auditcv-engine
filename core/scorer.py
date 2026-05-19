"""
core/scorer.py
7-dimension deterministic scoring. No LLM. Same input = same output always.
Weights: ATS 20% | Skills 20% | Projects 20% | Impact 15% | Structure 10% | Auth 10% | Fresher 5%
"""
from __future__ import annotations
import re, json
from dataclasses import dataclass, field
from pathlib import Path
from core.parser      import ParsedResume
from core.anti_gaming import AntiGamingResult
from core.embeddings  import EmbeddingResult

_DATA = Path(__file__).parent.parent / "data"

def _load_verbs() -> set[str]:
    p = _DATA / "action_verbs.txt"
    return {l.strip().lower() for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()

_ACTION_VERBS = _load_verbs()
_NUMBER_RE    = re.compile(r"\b\d+[\d,.]*\s*(%|percent|x|k|m|ms|s|gb|mb|hrs?|days?|weeks?|months?)?\b", re.IGNORECASE)
_PASSIVE_RE   = re.compile(r"\b(was|were|is|are|been|being)\s+\w+ed\b", re.IGNORECASE)

def _label(s: int) -> str:
    return "Strong" if s >= 80 else "Good" if s >= 60 else "Needs Work" if s >= 40 else "Weak"

def _score_range(s: int) -> str:
    lo = max(0, (s//10)*10 - 5); hi = min(100, (s//10)*10 + 5)
    return f"{lo}–{hi}"

def _color(s: int) -> str:
    return "#00E5A0" if s >= 80 else "#FF8C42" if s >= 40 else "#FF3333"

@dataclass
class DimScore:
    name: str; weight: float; raw_score: int; weighted: float
    label: str; score_range: str
    evidence:    list[str] = field(default_factory=list)
    penalties:   list[str] = field(default_factory=list)
    explanation: str       = ""

@dataclass
class ScoringResult:
    dimensions:       list[DimScore]
    overall_score:    int
    overall_range:    str
    overall_label:    str
    confidence:       str
    missing_critical: list[str] = field(default_factory=list)

    def dim(self, name: str) -> DimScore | None:
        return next((d for d in self.dimensions if d.name == name), None)

# ── Dimension 1: ATS Compatibility ────────────────────────────────────────────
def _ats(parsed: ParsedResume, extr_conf: float) -> DimScore:
    pts = 0; ev = []; pen = []
    c = parsed.contact
    if parsed.name:             pts += 8;  ev.append(f"Name: {parsed.name}")
    if c.get("email"):          pts += 8;  ev.append("Email present")
    if c.get("phone"):          pts += 7;  ev.append("Phone present")
    if c.get("linkedin") or c.get("github"): pts += 7; ev.append("Professional link present")
    for sec, w in [("education",12),("skills",12),("projects",10),("summary",6)]:
        if sec in parsed.sections_found: pts += w; ev.append(f"{sec.title()} section present")
        else: pen.append(f"Missing {sec.title()} section")
    if extr_conf >= 0.75:   pts += 30; ev.append("High PDF extraction quality")
    elif extr_conf >= 0.45: pts += 18; pen.append("Moderate extraction quality")
    else:                    pts += 6;  pen.append("Low extraction confidence")
    raw = min(100, pts)
    return DimScore("ATS Compatibility", 0.20, raw, raw*0.20, _label(raw), _score_range(raw),
                    ev, pen, "Parsability, contact info, section clarity, and PDF quality.")

# ── Dimension 2: Skills Evidence ──────────────────────────────────────────────
def _skills(parsed: ParsedResume, emb: EmbeddingResult) -> DimScore:
    pts = 0; ev = []; pen = []
    if not parsed.skills:
        return DimScore("Skills Evidence", 0.20, 0, 0.0, "Weak", "0–10", [], ["No skills section found"],
                        "No skills section detected — critical gap.")
    ev.append(f"{len(parsed.skills)} skills listed")
    pts += int(emb.support_ratio * 60)
    if emb.support_ratio >= 0.7:  ev.append(f"{int(emb.support_ratio*100)}% of skills backed by evidence")
    elif emb.support_ratio >= 0.4: pen.append(f"{len(emb.unsupported)} skills lack project evidence")
    else: pen.append(f"{len(emb.unsupported)} skills unsupported — add project evidence")
    skill_count = len(parsed.skills)
    if skill_count >= 10:   pts += 25; ev.append("Good skill variety (10+)")
    elif skill_count >= 6:  pts += 15; ev.append("Adequate skill count (6–9)")
    else:                   pts += 5;  pen.append("Very few skills — expand this section")
    taxonomy = json.loads((_DATA/"tech_skills.json").read_text())
    skills_lower = {s.lower() for s in parsed.skills}
    cats = sum(1 for items in taxonomy.values() if any(i in skills_lower for i in items))
    pts += min(cats*3, 15)
    if cats >= 3: ev.append(f"Skills span {cats} technical categories")
    raw = min(100, pts)
    return DimScore("Skills Evidence", 0.20, raw, raw*0.20, _label(raw), _score_range(raw),
                    ev, pen, "Skills backed by project/experience evidence via semantic matching.")

# ── Dimension 3: Project Quality ──────────────────────────────────────────────
def _projects(parsed: ParsedResume) -> DimScore:
    pts = 0; ev = []; pen = []
    if not parsed.projects:
        pen.append("No Projects section — critical for freshers. Add at least 2 projects.")
        return DimScore("Project Quality", 0.20, 5, 1.0, "Weak", "0–10", [], pen,
                        "Projects section is missing.")
    pc = len(parsed.projects)
    if pc >= 3:   pts += 30; ev.append(f"{pc} projects listed")
    elif pc == 2: pts += 20; ev.append("2 projects — consider adding a third")
    else:         pts += 10; pen.append("Only 1 project — add more")
    taxonomy = json.loads((_DATA/"tech_skills.json").read_text())
    all_tech = {t for items in taxonomy.values() for t in items}
    depths = [sum(1 for t in all_tech if t in p.lower()) for p in parsed.projects]
    avg_depth = sum(depths)/max(len(depths),1)
    if avg_depth >= 4:   pts += 25; ev.append(f"Strong technical depth (avg {avg_depth:.1f} tech terms/project)")
    elif avg_depth >= 2: pts += 15; ev.append(f"Moderate depth (avg {avg_depth:.1f})")
    else:                pts += 5;  pen.append("Projects lack tech specificity — name frameworks/tools used")
    total_metrics = sum(len(_NUMBER_RE.findall(p)) for p in parsed.projects)
    if total_metrics >= 3:   pts += 25; ev.append(f"{total_metrics} measurable outcomes found")
    elif total_metrics >= 1: pts += 12; pen.append(f"{total_metrics} metric(s) — add more quantified outcomes")
    else: pen.append("No measurable outcomes — add numbers: users, accuracy, performance")
    proj_blob = " ".join(parsed.projects).lower()
    verb_hits = sum(1 for v in _ACTION_VERBS if re.search(rf"\b{re.escape(v)}\b", proj_blob))
    if verb_hits >= 5:   pts += 20; ev.append(f"{verb_hits} strong action verbs in projects")
    elif verb_hits >= 2: pts += 10; pen.append(f"Only {verb_hits} action verbs — use more")
    else: pen.append("No action verbs — start bullets with: Built, Developed, Deployed...")
    raw = min(100, pts)
    return DimScore("Project Quality", 0.20, raw, raw*0.20, _label(raw), _score_range(raw),
                    ev, pen, "Depth, measurability, tech specificity, and action verbs in projects.")

# ── Dimension 4: Impact Statements ───────────────────────────────────────────
def _impact(parsed: ParsedResume) -> DimScore:
    pts = 0; ev = []; pen = []
    all_bullets = []
    for blob in parsed.experience + parsed.internship + parsed.projects:
        all_bullets.extend(l.strip("•·–- ").strip() for l in blob.splitlines() if l.strip())
    if not all_bullets:
        return DimScore("Impact Statements", 0.15, 10, 1.5, "Weak", "5–15", [], ["No bullet points detected"],
                        "No achievement lines found.")
    total = len(all_bullets)
    quantified = [b for b in all_bullets if _NUMBER_RE.search(b)]
    q_ratio = len(quantified)/total
    pts += int(q_ratio*35)
    if q_ratio >= 0.4: ev.append(f"{len(quantified)}/{total} bullets quantified")
    else: pen.append(f"Only {len(quantified)}/{total} bullets have numbers")
    verb_starts = [b for b in all_bullets if b.split() and b.split()[0].lower() in _ACTION_VERBS]
    v_ratio = len(verb_starts)/total
    pts += int(v_ratio*35)
    if v_ratio >= 0.5: ev.append(f"{len(verb_starts)}/{total} bullets start with action verb")
    else: pen.append(f"Only {len(verb_starts)}/{total} bullets start with action verbs")
    passives = len(_PASSIVE_RE.findall(" ".join(all_bullets)))
    pts += max(0, 20 - min(passives*4, 20))
    if passives > 2: pen.append(f"{passives} passive voice constructions — rewrite in active voice")
    avg_len = sum(len(b.split()) for b in all_bullets)/total
    if 8 <= avg_len <= 20: pts += 10; ev.append(f"Good bullet length (avg {avg_len:.0f} words)")
    elif avg_len < 5: pen.append("Bullets too short — add context and impact")
    raw = min(100, pts)
    return DimScore("Impact Statements", 0.15, raw, raw*0.15, _label(raw), _score_range(raw),
                    ev, pen, "Quantified achievements and strong active-voice language.")

# ── Dimension 5: Structure & Readability ──────────────────────────────────────
def _structure(parsed: ParsedResume) -> DimScore:
    pts = 0; ev = []; pen = []
    found = [s for s in ["education","skills","projects","summary"] if s in parsed.sections_found]
    pts += len(found)*10; ev.append(f"{len(parsed.sections_found)} sections detected")
    if parsed.sections_missing: pen.append(f"Missing: {', '.join(parsed.sections_missing)}")
    wc = parsed.word_count
    if 300 <= wc <= 900:   pts += 30; ev.append(f"Good resume length ({wc} words)")
    elif wc < 200:          pts += 10; pen.append(f"Too sparse ({wc} words)")
    elif wc > 1100:         pts += 15; pen.append(f"Too long ({wc} words) — aim for 1 page")
    c = parsed.contact
    filled = sum(1 for v in [c.get("email"),c.get("phone"),c.get("linkedin")] if v)
    pts += filled*5
    if filled == 3: ev.append("All contact fields present")
    if c.get("github"):   pts += 10; ev.append("GitHub linked")
    if c.get("linkedin"): pts += 5
    raw = min(100, pts)
    return DimScore("Structure & Readability", 0.10, raw, raw*0.10, _label(raw), _score_range(raw),
                    ev, pen, "Section completeness, length, and contact info.")

# ── Dimension 6: Authenticity & Trust ────────────────────────────────────────
def _authenticity(ag: AntiGamingResult) -> DimScore:
    raw = ag.authenticity_score; ev = []; pen = []
    if not ag.keyword_stuffing: ev.append("No keyword stuffing detected")
    else: pen.append(f"Keyword stuffing (density: {ag.stuffing_score:.2f})")
    if not ag.top_buzzwords: ev.append("Low buzzword usage")
    else: pen.append(f"Buzzwords: {', '.join(ag.top_buzzwords[:4])}")
    for claim in ag.unsupported_claims[:3]:
        pen.append(f"Unsupported claim: \"{claim.claim}\"")
    for metric in ag.suspicious_metrics[:2]:
        pen.append(f"Suspicious metric: {metric}")
    if ag.hidden_text_detected: pen.append("⚠ Hidden/invisible text detected — serious red flag")
    else: ev.append("No hidden text in PDF")
    return DimScore("Authenticity & Trust", 0.10, raw, raw*0.10, _label(raw), _score_range(raw),
                    ev, pen, "Keyword stuffing, buzzwords, unsupported claims, hidden text.")

# ── Dimension 7: Fresher Fairness ─────────────────────────────────────────────
def _fresher(parsed: ParsedResume) -> DimScore:
    pts = 0; ev = []; pen = []
    if not parsed.is_fresher:
        return DimScore("Fresher Fairness", 0.05, 70, 3.5, "Good", "65–75",
                        ["Experienced candidate — standard scoring applied"], [],
                        "Adjusts scoring to not unfairly penalise fresh graduates.")
    ev.append("Fresher profile — experience penalty waived"); pts += 30
    if parsed.projects:         pts += 30; ev.append(f"{len(parsed.projects)} project(s) compensate for no experience")
    else:                        pts -= 10; pen.append("Freshers need strong projects — missing")
    if parsed.certifications:   pts += 20; ev.append(f"{len(parsed.certifications)} certification(s)")
    else: pen.append("Add online certifications (Google, Meta, AWS)")
    if parsed.internship:       pts += 20; ev.append("Internship found — good signal")
    if parsed.achievements:     pts += 10; ev.append("Academic achievements listed")
    raw = max(0, min(100, pts))
    return DimScore("Fresher Fairness", 0.05, raw, raw*0.05, _label(raw), _score_range(raw),
                    ev, pen, "Rewards projects, certifications, and internships for fresh graduates.")

# ── Confidence + Public API ───────────────────────────────────────────────────
def _confidence(extr_conf: float, parsed: ParsedResume, emb: EmbeddingResult) -> str:
    sig = 0
    if extr_conf >= 0.75: sig += 2
    elif extr_conf >= 0.45: sig += 1
    if len(parsed.sections_found) >= 4: sig += 2
    elif len(parsed.sections_found) >= 2: sig += 1
    if not emb.fallback: sig += 2
    elif emb.support_ratio > 0: sig += 1
    return "High" if sig >= 5 else "Medium" if sig >= 3 else "Low"

def score(parsed: ParsedResume, ag: AntiGamingResult,
          emb: EmbeddingResult, extraction_confidence: float) -> ScoringResult:
    dims = [_ats(parsed, extraction_confidence), _skills(parsed, emb),
            _projects(parsed), _impact(parsed), _structure(parsed),
            _authenticity(ag), _fresher(parsed)]
    overall = int(sum(d.weighted for d in dims))
    missing = parsed.sections_missing[:]
    if not parsed.contact.get("email"): missing.append("email address")
    return ScoringResult(dims, overall, _score_range(overall), _label(overall),
                         _confidence(extraction_confidence, parsed, emb), missing)