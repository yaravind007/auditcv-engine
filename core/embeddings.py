"""
core/embeddings.py
MiniLM-L6-v2 semantic skill↔project matching. Falls back to exact-string if model unavailable.
"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class SkillMatch:
    skill: str; evidence: str; score: float; method: str; supported: bool

@dataclass
class EmbeddingResult:
    matches:       list[SkillMatch] = field(default_factory=list)
    unsupported:   list[str]        = field(default_factory=list)
    support_ratio: float            = 0.0
    model_used:    str              = "none"
    fallback:      bool             = False

_MODEL_NAME = "all-MiniLM-L6-v2"
_THRESHOLD  = 0.35
_model_cache = None
_model_failed = False

def _get_model():
    global _model_cache, _model_failed
    if _model_failed: return None
    if _model_cache: return _model_cache
    try:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer(_MODEL_NAME)
        return _model_cache
    except Exception:
        _model_failed = True; return None

def _exact_match(skill: str, corpus: list[str]) -> tuple[str, float]:
    skill_lower = skill.lower()
    for chunk in corpus:
        if skill_lower in chunk.lower(): return chunk[:120], 1.0
    return "", 0.0

def _semantic_match(model, skill: str, corpus: list[str]) -> tuple[str, float]:
    import numpy as np
    skill_emb   = model.encode([skill], convert_to_numpy=True, normalize_embeddings=True)
    corpus_embs = model.encode(corpus,  convert_to_numpy=True, normalize_embeddings=True)
    scores = (corpus_embs @ skill_emb.T).flatten()
    best_i = int(np.argmax(scores))
    return corpus[best_i][:120], float(scores[best_i])

def match_skills(skills: list[str], projects: list[str],
                 experience: list[str], internship: list[str]) -> EmbeddingResult:
    if not skills: return EmbeddingResult(support_ratio=0.0, model_used="none")
    all_evidence = projects + experience + internship
    if not all_evidence:
        return EmbeddingResult(unsupported=skills[:], support_ratio=0.0, model_used="none")
    corpus: list[str] = []
    for blob in all_evidence:
        sentences = [s.strip() for s in blob.replace("\n", ". ").split(".") if len(s.strip()) > 10]
        corpus.extend(sentences[:20])
    if not corpus: corpus = all_evidence
    model    = _get_model()
    fallback = model is None
    matches: list[SkillMatch] = []; unsupported: list[str] = []
    for skill in skills[:30]:
        if fallback:
            evidence, score = _exact_match(skill, corpus); method = "exact"
        else:
            try:    evidence, score = _semantic_match(model, skill, corpus); method = "semantic"
            except: evidence, score = _exact_match(skill, corpus); method = "exact"
        supported = (score >= _THRESHOLD if not fallback else len(evidence) > 0)
        matches.append(SkillMatch(skill=skill, evidence=evidence, score=round(score,3),
                                  method=method, supported=supported))
        if not supported: unsupported.append(skill)
    support_ratio = len([m for m in matches if m.supported]) / len(matches) if matches else 0.0
    return EmbeddingResult(matches=matches, unsupported=unsupported,
                           support_ratio=round(support_ratio,3),
                           model_used=_MODEL_NAME if not fallback else "exact-string-fallback",
                           fallback=fallback)