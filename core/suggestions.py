"""
core/suggestions.py
Rule-based suggestions engine. Gemini used ONLY for coaching paragraph phrasing — never for scoring.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from core.scorer      import ScoringResult
from core.parser      import ParsedResume
from core.anti_gaming import AntiGamingResult

@dataclass
class Suggestion:
    priority: int; category: str; issue: str; action: str; example: str = ""

@dataclass
class SuggestionsResult:
    suggestions: list[Suggestion]; ai_summary: str = ""; ai_used: bool = False

def _rule_based(parsed: ParsedResume, scoring: ScoringResult, ag: AntiGamingResult) -> list[Suggestion]:
    sugs: list[Suggestion] = []; p = 1
    if "projects" in parsed.sections_missing:
        sugs.append(Suggestion(p, "Projects", "Projects section missing — critical for freshers.",
            "Add 2-3 projects with tech stack, problem solved, and a metric.",
            "Built XGBoost churn model (Python, Pandas) — 89% AUC on 50K records.")); p+=1
    if "skills" in parsed.sections_missing:
        sugs.append(Suggestion(p, "Skills", "No Skills section detected.",
            "Add a dedicated Skills section with languages, tools, and frameworks.",
            "Languages: Python, SQL | Tools: Power BI, Pandas | Cloud: GCP")); p+=1
    if "education" in parsed.sections_missing:
        sugs.append(Suggestion(p, "Education", "Education section not found.",
            "Add degree, institution, year, and GPA if strong (7.5+/10).")); p+=1
    dim_impact = scoring.dim("Impact Statements")
    if dim_impact and dim_impact.raw_score < 60:
        sugs.append(Suggestion(p, "Impact Statements", "Bullets lack quantified achievements.",
            "Rewrite each bullet: [Action Verb] + [What you did] + [Measurable result].",
            "Reduced report generation time by 40% by automating Excel pipelines with Python.")); p+=1
    dim_proj = scoring.dim("Project Quality")
    if dim_proj and dim_proj.raw_score < 60 and parsed.projects:
        sugs.append(Suggestion(p, "Projects", "Projects lack depth or measurable outcomes.",
            "Name the tech stack, state the problem solved, add a metric for each project.",
            "Developed churn model (XGBoost) — 87% AUC on 50K records.")); p+=1
    dim_skills = scoring.dim("Skills Evidence")
    if dim_skills and dim_skills.raw_score < 55:
        sugs.append(Suggestion(p, "Skills", "Skills lack project evidence.",
            "Every listed skill should appear in at least one project description.",
            "If listing 'Tableau': 'Built Tableau dashboard for monthly sales across 3 regions.'")); p+=1
    if ag.keyword_stuffing:
        sugs.append(Suggestion(p, "Authenticity", "Keyword stuffing detected.",
            "Remove repeated keywords. ATS systems penalise stuffing. Quality over repetition.")); p+=1
    if ag.top_buzzwords:
        sugs.append(Suggestion(p, "Authenticity", f"Buzzwords: {', '.join(ag.top_buzzwords[:4])}.",
            "Replace vague buzzwords with specific, verifiable claims.",
            "Instead of 'passionate team player', write 'Collaborated with 4-person team to deliver X in Y weeks.'")); p+=1
    for claim in ag.unsupported_claims[:2]:
        sugs.append(Suggestion(p, "Credibility", f"Unsupported claim: '{claim.claim}'.",
            "Add project/experience evidence or downgrade level (Expert → Proficient).")); p+=1
    if not parsed.contact.get("github"):
        sugs.append(Suggestion(p, "Profile", "GitHub not linked.", "Add GitHub URL to resume header.")); p+=1
    if not parsed.contact.get("linkedin"):
        sugs.append(Suggestion(p, "Profile", "LinkedIn not linked.", "Add linkedin.com/in/yourname to header.")); p+=1
    if parsed.is_fresher and not parsed.certifications:
        sugs.append(Suggestion(p, "Certifications", "No certifications — important for freshers.",
            "Complete 1-2 free certifications: Google Data Analytics, Meta Marketing, AWS Cloud Practitioner.")); p+=1
    if not parsed.summary:
        sugs.append(Suggestion(p, "Summary", "No summary section.",
            "Add 2-3 lines: degree, key skills, career goal.",
            "MSc CS grad skilled in Python & Power BI. Seeking data analyst role to drive insights from complex data.")); p+=1
    return sugs

def _gemini_coaching(suggestions: list[Suggestion], parsed: ParsedResume, api_key: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel("gemini-1.5-flash")
        issues = "\n".join(f"- {s.issue}" for s in suggestions[:5])
        resp = m.generate_content(
            f"Resume coach. Candidate is {'fresher' if parsed.is_fresher else 'professional'}.\n"
            f"Issues:\n{issues}\n\nWrite 3-4 sentence direct coaching note. "
            "Acknowledge strengths, prioritise top 2 issues, end with motivation. No buzzwords.",
            generation_config=genai.GenerationConfig(temperature=0.4, max_output_tokens=300),
        )
        return resp.text.strip()
    except Exception: return ""

def generate(parsed: ParsedResume, scoring: ScoringResult,
             ag: AntiGamingResult, gemini_api_key: str = "") -> SuggestionsResult:
    sugs = _rule_based(parsed, scoring, ag)
    ai_summary = _gemini_coaching(sugs, parsed, gemini_api_key) if gemini_api_key and sugs else ""
    return SuggestionsResult(sugs, ai_summary, bool(ai_summary))