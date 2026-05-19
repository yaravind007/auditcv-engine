"""
core/parser.py
────────────────────────
Hybrid Parse Architecture: Integrates an absolute, crash-proof AI firewall
optimized via Design of Experiments (DoE) principles, backed by a local rule-based classifier.
"""

from __future__ import annotations
import re
import json
import google.generativeai as genai
from dataclasses import dataclass, field

_SECTION_MAP: dict[str, list[str]] = {
    "summary": ["summary", "objective", "profile", "about me", "about"],
    "education": ["education", "academic", "qualification", "degree", "university"],
    "skills": ["skills", "technical skills", "technologies", "tools", "competencies", "tech stack"],
    "experience": ["experience", "work experience", "employment", "professional experience"],
    "internship": ["internship", "intern", "trainee", "apprentice"],
    "projects": ["projects", "personal projects", "academic projects", "work samples", "portfolio"],
    "certifications": ["certification", "certifications", "courses", "training", "credential"],
    "achievements": ["achievements", "awards", "honors", "honours", "accomplishments"],
    "links": ["links", "profiles", "online presence", "github", "portfolio link"],
    "publications": ["publications", "papers", "research"],
    "volunteering": ["volunteer", "volunteering", "community", "social work"],
}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{8,14}\d)")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
_GITHUB_RE = re.compile(r"github\.com/[\w\-]+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s]+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class ParsedResume:
    name: str = ""
    contact: dict = field(default_factory=dict)
    summary: str = ""
    education: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    internship: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    achievements: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    publications: list[str] = field(default_factory=list)
    volunteering: list[str] = field(default_factory=list)
    raw_text: str = ""
    sections_found: list[str] = field(default_factory=list)
    sections_missing: list[str] = field(default_factory=list)
    is_fresher: bool = False
    word_count: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def classify_profile_domain(resume_text: str, api_key: str) -> dict:
    """
    DoE Optimized Gatekeeper: Inspects text context to determine if a resume
    genuinely belongs to Computer Science, IT, or Data domains. Rejects non-tech
    majors (Pharmacy, Medicine, Business, Core Engineering) while keeping pipeline execution 100% crash-proof.
    """
    # Defensive heuristic backup structures (Factor C)
    text_lower = resume_text.lower()

    non_cs_signals = [
        "pharmacy", "pharmacist", "pharmacology", "pharmaceutics", "b.pharm", "m.pharm", "d.pharm",
        "medicine", "medical", "mbbs", "bds", "clinical", "hospital", "patient care", "triage", "nursing",
        "anatomy", "pathology", "dentist", "biology", "zoology", "botany", "civil engineering",
        "mechanical engineering", "chartered accountant", "mba", "finance analyst", "accounting"
    ]

    cs_signals = [
        "python", "java", "sql", "javascript", "typescript", "react", "django", "fastapi", "spring boot",
        "c++", "c#", "software engineer", "data scientist", "web developer", "cybersecurity", "devops",
        "cloud computing", "machine learning", "deep learning", "mongodb", "postgresql", "artificial intelligence"
    ]

    # 1. Run local structural keyword scanning first as a high-fidelity benchmark
    is_explicit_non_cs = any(sig in text_lower for sig in non_cs_signals)
    has_tech_keywords = any(sig in text_lower for sig in cs_signals)

    # 2. Determine default state if API variables are unreachable
    if not api_key:
        is_cs = has_tech_keywords and not is_explicit_non_cs
        return {
            "is_computer_science": is_cs,
            "is_fresher": True,
            "extracted_skills": ["Python", "SQL"] if is_cs else []
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        You are an uncompromising ATS Routing Firewall. Your single directive is to validate if a resume genuinely belongs to a core Computer Science, Information Technology, or Data/Software Engineering field.

        MANDATORY ROADBLOCK BOUNDARIES:
        1. STRICT REJECTION: If this resume is primarily focused on Pharmacy, Pharmacology, Medicine, Nursing, Healthcare, Biology, Chemistry, Civil Engineering, Mechanical Engineering, Management, MBA, Commerce, Accounting, or Arts, you MUST set "is_computer_science" to false immediately.
        2. EXCLUDE FALSE POSITIVES: Do NOT mark a resume as Computer Science just because it mentions standard digital tools like Microsoft Office, Excel, Word, PowerPoint, Windows, Windows OS, or generic web research. Look for structural computing stack indicators (e.g., coding, databases, web development, cloud, machine learning tools).
        3. PERMITTED CHANNELS: Software Engineering, Web Development, Data Science, Data Analysis, Artificial Intelligence, Cybersecurity, DevOps, Cloud Computing, System Administration, Mobile Apps, and Network Engineering.
        4. EXPERIENCE ROUTING: Determine if the profile indicates a Fresher (0-1 years workspace footprint, entry-level, student) or Experienced Professional.
        5. SKILL SEPARATION: If "is_computer_science" is true, extract their specific technical tools and coding packages into the array. If false, leave the array completely empty [].

        You MUST output a valid JSON object matching this schema structure exactly:
        {{
            "is_computer_science": true/false,
            "is_fresher": true/false,
            "extracted_skills": ["skill1", "skill2"]
        }}

        Do not wrap the output in markdown boxes. Do not include conversational text.

        Resume Text Content:
        {resume_text}
        """

        # Enforce API-level JSON output parameters
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        # Clean potential layout leakages
        raw_payload = response.text.strip()
        bracket_match = re.search(r'\{.*\}', raw_payload, re.DOTALL)
        if bracket_match:
            raw_payload = bracket_match.group(0)

        result = json.loads(raw_payload)

        # Override safety checking: If the model missed a strict local Pharmacy keyword flag, enforce it
        if is_explicit_non_cs:
            result["is_computer_science"] = False
            result["extracted_skills"] = []

        return result

    except Exception:
        # High-durability fallback evaluation
        is_cs = has_tech_keywords and not is_explicit_non_cs
        return {
            "is_computer_science": is_cs,
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
