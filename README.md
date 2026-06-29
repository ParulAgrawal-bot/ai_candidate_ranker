# Redrob Intelligent Candidate Discovery & Ranking Engine

This repository contains the full execution roadmap and implementation for the Redrob Senior AI Engineer shortlist and candidate ranking engine.

## 🎯 Architecture Overview

The system uses a two-stage architecture to meet the strict compute constraints of the hackathon (ranking completed in under 5 minutes on a CPU-only machine, no live network access):

```
                                  +------------------------------------+
                                  |    Raw Candidates JSONL Dataset    |
                                  +-----------------+------------------+
                                                    |
                                                    v
+---------------------------------------------------+---------------------------------------------------+
|                                        OFFLINE PRE-COMPUTE STAGE                                      |
|                                                                                                       |
|  +--------------------+      +--------------------+      +--------------------+      +-------------+  |
|  |   data_parser.py   | ---> | feature_builder.py | ---> |    embedder.py     | ---> |  Save NPY/  |  |
|  | (Honeypot Checks & |      | (Extract skills,   |      | (Local sentence-   |      |  PKL Files  |  |
|  |   Normalizations)  |      | career & signals)  |      |  transformer L6)   |      | (Artifacts) |  |
|  +--------------------+      +--------------------+      +--------------------+      +------+------+  |
+---------------------------------------------------------------------------------------------|---------+
                                                                                              |
                                                                                              v
+---------------------------------------------------------------------------------------------|---------+
|                                         ONLINE FAST-RANK STAGE                              |
|                                                                                             |
|  +--------------------+      +--------------------+      +--------------------+      +------+------+  |
|  |    Jinja2 & LTR    | <--- |  hybrid_scorer.py  | <--- | behavioral_scorer  | <--- |   Load NPY/ |  |
|  |  (Reasoning Gen &  |      |   (Combine core    |      | (23 Availability & |      |  PKL Files  |  |
|  |  shortlisted CSV)  |      |    score weights)  |      |  reliability mult) |      | (Artifacts) |  |
|  +--------------------+      +--------------------+      +--------------------+      +-------------+  |
+-------------------------------------------------------------------------------------------------------+
```

1. **Offline Precomputation**: Processes all candidate profiles, validates consistency (detects honeypots), extracts structured features, and generates profile embeddings using a local `all-MiniLM-L6-v2` SentenceTransformer. Saves embeddings and metadata in `artifacts/`.
2. **Online Fast Ranking**: Run by `src/rank.py`, this step loads precomputed artifacts, calculates cosine similarity with the Job Description (JD) vector, computes career, skill, and rule scores, applies the availability multiplier, selects the top 100, generates fact-grounded reasonings, and outputs the final CSV.

---

## 📈 Hybrid Scoring Formula

The final candidate ranking score is calculated as a weighted sum of four components, scaled by a behavioral availability multiplier:

$$\text{Final Score} = (0.35 \times \text{Skill} + 0.30 \times \text{Semantic} + 0.25 \times \text{Career} + 0.10 \times \text{Rule}) \times \text{Behavioral Multiplier}$$

### Component Breakdown
* **Semantic Score (30%):** Vector similarity between the candidate's parsed text blob and the JD profile.
* **Skill Match Score (35%):** Weighted overlap across must-have categories (Embeddings/Retrieval, Vector DB, Evaluation/Ranking, Python), factoring in proficiency and endorsements.
* **Career Score (25%):** Analyzes product-company tenure ratio, job-hopping penalty, and recency-weighted title relevance.
* **Rule Score (10%):** Validates explicit requirements: YoE band (5-9 years preferred), notice period, local/relocation availability, and CS degree.
* **Behavioral Multiplier (0.5x - 1.2x):** Blends 23 behavioral availability and trustworthiness signals (e.g. days active, response rates, interview completion) to filter out inactive "ghost" profiles.

---

## 🚀 Reproduction Instructions

### 1. Installation
Install the required dependencies on a clean environment:
```bash
pip install -r requirements.txt
```

### 2. Run Precomputation (Offline)
Generate candidate embeddings and feature files. Make sure your raw candidates file is at `data/candidates.jsonl.gz` (or specify the path):
```bash
python precompute/precompute_embeddings.py --candidates data/candidates.jsonl.gz
```

### 3. Run Ranker (Online)
Run the fast ranking pipeline to generate the shortlisted CSV. This completes in under 60 seconds:
```bash
python src/rank.py --candidates data/candidates.jsonl.gz --out output/team_xxx.csv
```

### 4. Validate Submission Format
Verify the output CSV format, row counts, monotonicity, and tie-breaking ordering:
```bash
python validate_submission.py output/team_xxx.csv --candidates data/candidates.jsonl.gz
```

### 5. Start the Interactive Sandbox (Streamlit)
Launch the Streamlit dashboard for interactive uploads and candidate inspection:
```bash
streamlit run src/app.py
```

---

## 📁 Repository Layout

```
ai_candidate_ranker/
├── .github/workflows/       # GitHub Actions CI configurations
│   └── validate.yml         # Run integration test & validation
├── precompute/
│   └── precompute_embeddings.py  # Offline embedding & feature builder script
├── src/
│   ├── app.py               # Streamlit application sandbox
│   ├── rank.py              # Main CLI entrypoint
│   ├── data_parser.py       # JSONL stream and honeypot validation
│   ├── feature_builder.py   # Extracts candidate features & computes rules
│   ├── embedder.py          # Sentence-transformer encoding helper
│   ├── skill_scorer.py      # Skill matching algorithms
│   ├── career_scorer.py     # Job history, tenure and title evaluation
│   ├── behavioral_scorer.py # 23-signal availability multiplier
│   ├── hybrid_scorer.py     # Scoring combinations
│   └── reasoning_gen.py     # Jinja2-based reasoning compilation
├── data/                    # Candidate raw datasets (Gitignored)
├── artifacts/               # Precomputed candidate features & embeddings (Gitignored)
├── output/                  # Final submissions CSV outputs (Gitignored)
├── submission_metadata.yaml # Portal submission metadata
├── requirements.txt         # Package dependencies
└── validate_submission.py   # Final submission format validator
```
