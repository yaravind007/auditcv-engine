"""
core/parser.py
────────────────────────
Hybrid Parse Architecture: Integrates deep multi-parameter cross-scanning logic.
Validates transitions via project and skill metrics to prevent false rejections.
"""

from __future__ import annotations
import re
import json
import google.generativeai as genai
from dataclasses import dataclass, field

_SECTION_MAP: dict[str, list[str]] = {
    "summary":        ["summary","objective","profile","about me","about"],
    "education":      ["education","academic","qualification","degree","university"],
    "skills":         ["skills","technical skills","technologies","tools","competencies","tech stack"],
    "experience":     ["experience","work experience","employment","professional experience"],
    "internship":     ["internship","intern","trainee","apprentice"],
    "projects":       ["projects","personal projects","academic projects","work samples","portfolio"],
    "certifications": ["certification","certifications","courses","training","credential"],
    "achievements":   ["achievements","awards","honors","honours","accomplishments"],
    "links":          ["links","profiles","online presence","github","portfolio link"],
    "publications":   ["publications","papers","research"],
    "volunteering":   ["volunteer","volunteering","community","social work"],
}

_EMAIL_RE    = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
_PHONE_RE    = re.compile(r"(\+?\d[\d\s\-().]{8,14}\d)")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
_GITHUB_RE   = re.compile(r"github\.com/[\w\-]+", re.IGNORECASE)
_URL_RE      = re.compile(r"https?://[^\s]+")
_YEAR_RE     = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class ParsedResume:
    name:             str       = ""
    contact:          dict      = field(default_factory=dict)
    summary:          str       = ""
    education:        list[str] = field(default_factory=list)
    skills:           list[str] = field(default_factory=list)
    projects:         list[str] = field(default_factory=list)
    experience:       list[str] = field(default_factory=list)
    internship:       list[str] = field(default_factory=list)
    certifications:   list[str] = field(default_factory=list)
    achievements:     list[str] = field(default_factory=list)
    links:            list[str] = field(default_factory=list)
    publications:     list[str] = field(default_factory=list)
    volunteering:     list[str] = field(default_factory=list)
    raw_text:         str       = ""
    sections_found:   list[str] = field(default_factory=list)
    sections_missing: list[str] = field(default_factory=list)
    is_fresher:       bool      = False
    word_count:       int       = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def classify_profile_domain(resume_text: str, api_key: str) -> dict:
    """
    Performs deep scanning across Education, Skills, and Projects matrices.
    If any sector demonstrates strong tech evidence, the candidate passes.
    """
    text_lower = resume_text.lower()
    
    # ── Deterministic Scoring Weights For Fallback/Guardrail ──────────────────
    edu_score   = 100 if any(kw in text_lower for kw in ["computer science", "information technology", "b.sc cs", "m.sc cs", "b.tech cs", "m.tech cs", "mca"]) else 0
    
    tech_skills = ["python", "java", "sql", "javascript", "typescript", "react", "django", "fastapi", "spring boot", "c++", "c#", "mongodb", "postgresql", "power bi", "tableau", "dax studio", "mysql"]
    skill_hits  = sum(1 for skill in tech_skills if re.search(rf"\b{re.escape(skill)}\b", text_lower))
    skills_score = min(100, skill_hits * 25)  # 4+ core tech skills = 100 points
    
    proj_indicators = ["github repo", "live dashboard", "presentation video", "project challenge", "deployed", "built a dashboard", "data modeling", "performed etl"]
    proj_hits       = sum(1 for ind in proj_indicators if ind in text_lower)
    projects_score  = min(100, proj_hits * 34)  # 3+ build signals = 100 points
    
    # Pass if any matrix factor contains strong evidence (threshold >= 75)
    deterministic_pass = (edu_score >= 75 or skills_score >= 75 or projects_score >= 75)
    
    if not api_key:
        return {
            "is_computer_science": deterministic_pass,
            "is_fresher": True,
            "extracted_skills": ["Python", "SQL"] if deterministic_pass else []
        }
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are an advanced, analytical ATS Routing Gatekeeper. Your objective is to run a deep, multi-parameter scan on a candidate's text profile to evaluate tech alignment.

        DEEP MATRIX SCANNING PROTOCOLS:
        1. EDUCATION MATRIX: Identify formal computing degrees.
        2. SKILLS MATRIX: Look for developer tools, programming languages, and engineering software packages.
        3. PROJECTS MATRIX: Check for active development markers, portfolio references (GitHub, Live Dashboards), ETL pipelines, analytics modeling, or hackathon wins.

        EVALUATION LOGIC BLOCK:
        Even if the candidate has an unrelated formal background (e.g., Pharmacy, Medicine, Biomedical Engineering, Civil, Commerce), you MUST set "is_computer_science" to true if EITHER their Skills Matrix or Projects Matrix shows strong evidence of transition capabilities (e.g., building data engines, designing database systems, setting up BI architectures). 
        Only set to false if ALL three matrices show zero alignment with computer systems or digital analytics architecture.

        You MUST output a valid JSON object matching this schema structure exactly:
        {{
            "is_computer_science": true/false,
            "is_fresher": true/false,
            "extracted_skills": ["skill1", "skill2"]
        }}
        
        Do not wrap the output in markdown boxes.
        
        Candidate Text Pool:
        {resume_text}
        """
        
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        raw_payload = response.text.strip()
        bracket_match = re.search(r'\{.*\}', raw_payload, re.DOTALL)
        if bracket_match:
            raw_payload = bracket_match.group(0)
            
        result = json.loads(raw_payload)
        
        # Guardrail cross-check: Force pass if local metrics find absolute project/skill depth
        if deterministic_pass:
            result["is_computer_science"] = True
            
        return result
        
    except Exception:
        return {
            "is_computer_science": deterministic_pass,
            "is_fresher": True,
            "extracted_skills": []
        }


def _detect_section(line: str) -> str | None:
    stripped = line.strip().rstrip(":").strip().lower()
    if not stripped or len(stripped) > 40: return None
    for key, variants in _SECTION_MAP.items():
        if any(stripped == v or stripped.startswith(v) for v in variants):
            return key
    if line.strip().isupper() and 3 <= len(line.strip()) <= 35:
        for key, variants in _SECTION_MAP.items():
            if any(line.strip().lower() == v for v in variants):
                return key
    return None


def _extract_name(lines: list[str]) -> str:
    contact_signals = {"@", "http", "linkedin", "github", "phone", "+91", "www."}
    for line in lines[:6]:
        l = line.strip()
        if not l: continue
        if any(s in l.lower() for s in contact_signals): continue
        if _EMAIL_RE.search(l) or _PHONE_RE.search(l): continue
        if re.match(r"^[A-Za-z][a-zA-Z\s.]{3,40}$", l): return l.strip()
    return ""


def _extract_skills_from_text(text: str) -> list[str]:
    raw = re.split(r"[,|•·\n\t/\\]+", text)
    skills = []
    for item in raw:
        item = item.strip().strip("–-").strip()
        if 1 <= len(item.split()) <= 5 and len(item) >= 2:
            skills.append(item)
    return list(dict.fromkeys(skills))


def _is_fresher(parsed: "ParsedResume") -> bool:
    no_exp = not parsed.experience and not parsed.internship
    edu_text = " ".join(parsed.education).lower()
    recent_grad = bool(_YEAR_RE.search(edu_text))
    return no_exp or (recent_grad and not parsed.experience)


def parse(text: str, identified_skills: list[str] | None = None) -> ParsedResume:
    """
    Parses resume layout sections deterministically. If AI-extracted skills
    are supplied by the app pipeline orchestration layer, they override standard string splits.
    """
    result = ParsedResume(raw_text=text)
    lines = text.splitlines()
    result.word_count = len(text.split())

    email_m = _EMAIL_RE.search(text)
    phone_m = _PHONE_RE.search(text)
    linkedin_m = _LINKEDIN_RE.search(text)
    github_m = _GITHUB_RE.search(text)

    result.contact = {
        "email": email_m.group() if email_m else "",
        "phone": phone_m.group() if phone_m else "",
        "linkedin": linkedin_m.group() if linkedin_m else "",
        "github": github_m.group() if github_m else "",
    }

    for url in _URL_RE.findall(text):
        if url not in result.links: result.links.append(url)
    result.name = _extract_name(lines)
    current_section: str | None = None
    buffer: list[str] = []

    def _flush(section: str | None, buf: list[str]) -> None:
        if not section or not buf: return
        blob = "\n".join(b for b in buf if b.strip())
        if not blob.strip(): return
        if section == "summary":
            result.summary = blob.strip()
        elif section == "education":
            result.education.append(blob.strip())
        elif section == "skills":
            if identified_skills is not None and len(identified_skills) > 0:
                pass
            else:
                result.skills.extend(_extract_skills_from_text(blob))
        elif section == "projects":
            result.projects.extend(c.strip() for c in re.split(r"\n{2,}", blob) if c.strip())
        elif section == "experience":
            result.experience.append(blob.strip())
        elif section == "internship":
            result.internship.append(blob.strip())
        elif section == "certifications":
            result.certifications.extend(l.strip() for l in buf if l.strip() and len(l.strip()) > 3)
        elif section == "achievements":
            result.achievements.extend(l.strip("•·–- ").strip() for l in buf if l.strip())
        elif section == "publications":
            result.publications.append(blob.strip())
        elif section == "volunteering":
            result.volunteering.append(blob.strip())

    for line in lines:
        detected = _detect_section(line)
        if detected:
            _flush(current_section, buffer)
            current_section = detected
            buffer = []
        else:
            buffer.append(line)
    _flush(current_section, buffer)

    if identified_skills is not None and len(identified_skills) > 0:
        result.skills = identified_skills

    all_sections = ["education", "skills", "projects", "experience",
                    "internship", "certifications", "achievements", "summary"]
    result.sections_found = [s for s in all_sections if getattr(result, s, None)]
    result.sections_missing = [s for s in ["education", "skills", "projects"] if s not in result.sections_found]
    result.is_fresher = _is_fresher(result)

    seen: set[str] = set()
    result.skills = [s for s in result.skills if s.lower().strip() not in seen and not seen.add(s.lower().strip())]
    return result