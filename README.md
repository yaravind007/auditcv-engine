# 🎯 AuditCV

An explainable, fairness-aware, and deterministic resume auditing intelligence engine tailored for freshers. Built to replace opaque black-box resume screening algorithms with absolute data transparency, structural validation, and robust anti-gaming protections.

---

#  System Architecture & Engineering Highlights

Unlike traditional Applicant Tracking Systems (ATS) or modern wrappers that hide behind uncalibrated LLM match scores (e.g., a generic “78% match”), **AuditCV** operates on an objective, deterministic multi-stage evaluation pipeline:

1. **Deterministic Scoring Matrix**  
   Core metrics are fully rule-backed, mathematical, and predictable. The engine eliminates AI hallucinations by mapping scoring parameters directly to verifiable structural patterns.

2. **Anti-Gaming Vetting Layer**  
   Programmatically analyzes the underlying source layers of uploaded documents to:
   - Flag hidden background keyword stuffing
   - Detect statistically improbable or inflated performance metrics
   - Isolate exploits designed to manipulate legacy keyword parsers

3. **Semantic Skill Alignment**  
   Employs advanced token vector embedding associations to cross-verify listed technical capabilities against context-driven project descriptions and experience sections.

4. **Structural Guardrail Vetting**  
   Operates a gatekeeping layer that identifies invalid document taxonomy patterns. If a non-resume document (e.g., university fee receipt or textbook chapter) is uploaded, the system intercepts processing with a clean fallback alert.

5. **Dual-Action Optimization Loop**  
   Outputs:
   - A localized engineering audit report
   - A tailored **Master AI Re-Writer Prompt**

   Users can safely refactor resume bullet points using their preferred private LLM workflows.

6. **Telemetry Streaming Pipeline**  
   Implements a completely serverless, cloud-ready telemetry infrastructure that mirrors client-side evaluation sessions into automated Google Sheets analytics pipelines through an HTTPS webhook API.

---

# 📂 Project Repository Anatomy

The codebase follows strict separation-of-concerns architecture boundaries:

```text
auditcv-engine/
│
├── .streamlit/
│   └── config.toml            # Streamlit deployment configuration
│
├── core/                      # Stateless backend analytics pipeline
│   ├── __init__.py
│   ├── extractor.py           # PDF extraction & raw text tracking
│   ├── parser.py              # Resume taxonomy segmentation
│   ├── anti_gaming.py         # Hidden text & keyword stuffing detection
│   ├── embeddings.py          # Semantic vector comparison logic
│   ├── scorer.py              # Deterministic scoring engine
│   └── suggestions.py         # Resume refinement generation
│
├── data/                      # Verification datasets & dictionaries
│   ├── action_verbs.txt
│   ├── buzzwords.txt
│   └── tech_skills.json
│
├── .gitignore
├── app.py                     # Streamlit presentation layer
├── README.md                  # Project documentation
└── requirements.txt           # Dependency manifest
```

---

# 🚀 Local Installation & Deployment Blueprint

## Prerequisites

- Python 3.12+
- Git installed on your system

---

## 1. Clone the Repository

```bash
git clone https://github.com/yaravind007/auditcv-engine.git
cd auditcv-engine
```

---

## 2. Configure a Virtual Environment

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (Command Prompt)

```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---


## 4. Launch the Platform

```bash
streamlit run app.py
```

---

# 🎨 Creative Manifesto: Code Directed by Strategy

> “The modern developer is no longer a code compiler, but an architect of complete systems.”

AuditCV represents a paradigm shift in how intelligent tooling is engineered. The platform was not created through blind code generation. Instead, it was systematically designed through disciplined modular architecture planning, deterministic validation logic, and explicitly defined execution boundaries.

Every parsing loop, scoring parameter, regex guardrail, telemetry stream, and visualization layer was structured through intentional engineering strategy before implementation.

---

# 👨‍💻 Core System Architect

**Aravind Kumar Yedida**

- Portfolio: `www.aravindyedida.com`
- LinkedIn: `www.linkedin.com/in/aravindyedida`
- GitHub: `www.github.com/yaravind007`

---

# 📜 License

Licensed under the **MIT License** — feel free to use, fork, extend, and scale this architecture to support freshers worldwide.