import os
import sys
import json
import gzip
import tempfile
import pandas as pd
import numpy as np
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_parser import parse_and_validate_candidate
from src.skill_scorer import calculate_skill_score
from src.career_scorer import calculate_career_score
from src.feature_builder import calculate_rule_score, extract_features
from src.behavioral_scorer import compute_behavioral_multiplier
from src.hybrid_scorer import calculate_hybrid_score
from src.reasoning_gen import generate_reasoning
from src.embedder import get_model, encode_text, build_candidate_text_blob

# Page configuration
st.set_page_config(
    page_title="Redrob Candidate Ranking Sandbox",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and premium dark theme
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">

<style>
    /* Main Layout */
    .reportview-container {
        background: linear-gradient(135deg, #0e1117 0%, #161a24 100%);
    }
    
    body {
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
        font-weight: 700;
    }
    
    /* Header Gradient styling */
    .header-container {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subheader {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease-in-out, border-color 0.2s;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #6366f1;
        margin-top: 0.5rem;
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Score Indicator badges */
    .badge {
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .badge-info {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Upload Section */
    .stFileUploader {
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.01);
    }
</style>
""", unsafe_allow_html=True)

# Cache model load to keep it fast
@st.cache_resource
def load_local_model():
    return get_model('all-MiniLM-L6-v2')

# Cache JD embedding to prevent re-encoding on rerun
@st.cache_data
def get_cached_jd_embedding(jd_text):
    model = load_local_model()
    return model.encode(jd_text, convert_to_numpy=True)

def score_uploaded_candidates(candidates_list, jd_text):
    """
    Scores candidates uploaded dynamically.
    Since upload is <= 100 records, we can embed them on the fly!
    """
    model = load_local_model()
    jd_embedding = get_cached_jd_embedding(jd_text)
    
    results = []
    text_blobs = []
    processed_candidates = []
    
    # 1. First Pass: Parse and validate candidates, build text blobs
    for cand in candidates_list:
        parsed_cand = parse_and_validate_candidate(cand)
        # Extract features (which flags disqualifiers)
        feats = extract_features(parsed_cand)
        
        processed_candidates.append(parsed_cand)
        text_blobs.append(build_candidate_text_blob(parsed_cand))
        
    # 2. Second Pass: Generate candidate embeddings on the fly
    embeddings = model.encode(text_blobs, convert_to_numpy=True)
    
    # 3. Third Pass: Compute scores
    for idx, candidate in enumerate(processed_candidates):
        profile = candidate.get('profile', {})
        candidate_id = profile.get('candidate_id', 'UNKNOWN')
        
        # Calculate semantic score
        cand_emb = embeddings[idx]
        dot_product = np.dot(cand_emb, jd_embedding)
        norm_a = np.linalg.norm(cand_emb)
        norm_b = np.linalg.norm(jd_embedding)
        semantic_score = float(dot_product / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0
        
        # Calculate other components
        skill_score = calculate_skill_score(candidate)
        career_score = calculate_career_score(candidate)
        rule_score = calculate_rule_score(candidate)
        behavioral_multiplier = compute_behavioral_multiplier(candidate)
        
        # Hybrid combination
        score_details = calculate_hybrid_score(
            semantic_score=semantic_score,
            skill_score=skill_score,
            career_score=career_score,
            rule_score=rule_score,
            behavioral_multiplier=behavioral_multiplier,
            candidate=candidate
        )
        
        results.append({
            "candidate_id": candidate_id,
            "final_score": score_details["final_score"],
            "semantic_score": score_details["semantic_score"],
            "skill_score": score_details["skill_score"],
            "career_score": score_details["career_score"],
            "rule_score": score_details["rule_score"],
            "behavioral_multiplier": score_details["behavioral_multiplier"],
            "is_honeypot": score_details["is_honeypot"],
            "is_disqualified": score_details["is_disqualified"],
            "disqualification_reasons": score_details["disqualification_reasons"],
            "candidate_data": candidate
        })
        
    # Sort and rank
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    
    for i, res in enumerate(results):
        res["rank"] = i + 1
        res["reasoning"] = generate_reasoning(res["candidate_data"], res["rank"], res["final_score"])
        
    return results

def main():
    # Sidebar
    st.sidebar.markdown("<div style='text-align: center; padding: 1rem;'><h2 style='color:#6366f1; margin:0;'>REDROB</h2><p style='color:#9ca3af; font-size:0.8rem; margin:0;'>Intelligent Candidate Shortlisting</p></div>", unsafe_allow_html=True)
    st.sidebar.divider()
    
    # JD Textarea in sidebar
    st.sidebar.subheader("Job Description Configuration")
    jd_default = (
        "Senior AI Engineer. Mandatory skills: Production embeddings-based retrieval "
        "(sentence-transformers, BGE, E5, OpenAI embeddings deployed to real users), "
        "Vector DB and hybrid search experience (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, FAISS, Elasticsearch), "
        "Evaluation frameworks for ranking (NDCG, MRR, MAP, A/B testing, offline-to-online correlation), "
        "strong Python programming. Nice-to-haves: LLM fine-tuning (LoRA, QLoRA, PEFT), "
        "learning-to-rank experience (XGBoost or LTR), distributed systems or large-scale inference optimization, "
        "HR-tech/recruiting/marketplace product experience. Location: Pune / Noida preferred."
    )
    jd_text = st.sidebar.text_area("Configure the target JD used for semantic matching", value=jd_default, height=300)
    
    st.sidebar.divider()
    st.sidebar.info("Upload candidate JSON/JSONL file up to 100 rows to rank them interactively.")

    # Main Area
    st.markdown("<div class='header-container'>Intelligent Candidate Discovery & Ranking</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Redrob Senior AI Engineer Shortlist Engine • Powered by Hybrid Scoring</div>", unsafe_allow_html=True)

    # 1. File Upload
    uploaded_file = st.file_uploader("Upload Candidates JSONL or JSON file", type=["jsonl", "json", "gz"])
    
    if uploaded_file is not None:
        candidates = []
        try:
            # Parse file based on extension
            filename = uploaded_file.name
            file_bytes = uploaded_file.read()
            
            if filename.endswith('.gz'):
                decompressed = gzip.decompress(file_bytes)
                content = decompressed.decode('utf-8')
            else:
                content = file_bytes.decode('utf-8')
                
            if filename.endswith('.json'):
                candidates = json.loads(content)
                if not isinstance(candidates, list):
                    candidates = [candidates]
            else:
                # JSONL
                for line in content.split('\n'):
                    if line.strip():
                        candidates.append(json.loads(line))
            
            # Limit upload size
            if len(candidates) > 200:
                st.warning(f"Uploaded file has {len(candidates)} candidates. Sandbox mode is optimized for up to 100 candidates to remain responsive. Truncating to first 100.")
                candidates = candidates[:100]
                
            st.success(f"Successfully loaded {len(candidates)} candidates. Running hybrid ranker...")
            
            # Score
            with st.spinner("Embedding candidates on-the-fly and running hybrid ranking scorers..."):
                ranked_results = score_uploaded_candidates(candidates, jd_text)
                
            # Create metrics row
            total_uploaded = len(ranked_results)
            honeypots_count = sum(1 for r in ranked_results if r["is_honeypot"])
            disqualified_count = sum(1 for r in ranked_results if r["is_disqualified"] and not r["is_honeypot"])
            valid_shortlist = sum(1 for r in ranked_results if r["final_score"] > 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Candidates</div><div class='metric-val'>{total_uploaded}</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Valid Profiles</div><div class='metric-val' style='background: linear-gradient(90deg, #10b981, #34d399); -webkit-background-clip: text;'>{valid_shortlist}</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Honeypots Blocked</div><div class='metric-val' style='background: linear-gradient(90deg, #f59e0b, #fbbf24); -webkit-background-clip: text;'>{honeypots_count}</div></div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Disqualified</div><div class='metric-val' style='background: linear-gradient(90deg, #ef4444, #f87171); -webkit-background-clip: text;'>{disqualified_count}</div></div>", unsafe_allow_html=True)
                
            st.write("")
            
            # Build DataFrame for display
            display_data = []
            csv_data = []
            for r in ranked_results:
                profile = r["candidate_data"].get("profile", {})
                display_data.append({
                    "Rank": r["rank"],
                    "Candidate ID": r["candidate_id"],
                    "Score": r["final_score"],
                    "Headline": profile.get("headline", ""),
                    "YoE": profile.get("years_of_experience", 0.0),
                    "Honeypot": "🚨 Yes" if r["is_honeypot"] else "✅ No",
                    "Disqualified": "❌ Yes" if r["is_disqualified"] else "✅ No",
                })
                
                csv_data.append({
                    "candidate_id": r["candidate_id"],
                    "rank": r["rank"],
                    "score": round(r["final_score"], 4),
                    "reasoning": r["reasoning"]
                })
                
            df_display = pd.DataFrame(display_data)
            df_csv = pd.DataFrame(csv_data)
            
            # Download short-list CSV
            csv_string = df_csv.to_csv(index=False, encoding="utf-8")
            st.download_button(
                label="📥 Download Shortlist (team_xxx.csv)",
                data=csv_string,
                file_name="team_xxx.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.subheader(" शॉर्टलिस्टेड लीडरबोर्ड (Ranked Shortlist Leaderboard)")
            st.dataframe(
                df_display, 
                column_config={
                    "Score": st.column_config.NumberColumn(format="%.4f"),
                    "YoE": st.column_config.NumberColumn(format="%.1f")
                },
                hide_index=True, 
                use_container_width=True
            )
            
            # Interactive details viewer
            st.divider()
            st.subheader("🔍 Detailed Candidate Inspector")
            
            # Select candidate to inspect
            inspect_ids = [r["candidate_id"] for r in ranked_results]
            selected_id = st.selectbox("Select a Candidate ID to view details", inspect_ids)
            
            if selected_id:
                # Find details
                details = next(r for r in ranked_results if r["candidate_id"] == selected_id)
                cand = details["candidate_data"]
                profile = cand.get("profile", {})
                signals = cand.get("redrob_signals", {})
                skills = cand.get("skills", [])
                career_history = cand.get("career_history", [])
                
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.write(f"### Profile Summary: {selected_id}")
                    st.write(f"**Name/Headline:** {profile.get('headline', 'N/A')}")
                    st.write(f"**Summary:** {profile.get('summary', 'N/A')}")
                    st.write(f"**Location:** {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}")
                    st.write(f"**Current Title:** {profile.get('current_title', 'N/A')} at {profile.get('current_company', 'N/A')}")
                    st.write(f"**Years of Experience:** {profile.get('years_of_experience', 'N/A')}")
                    
                    st.write("#### Score Breakdown")
                    score_df = pd.DataFrame([
                        {"Component": "Semantic Score (30%)", "Value": f"{details['semantic_score']:.4f}"},
                        {"Component": "Skill Score (35%)", "Value": f"{details['skill_score']:.4f}"},
                        {"Component": "Career Score (25%)", "Value": f"{details['career_score']:.4f}"},
                        {"Component": "Rule Score (10%)", "Value": f"{details['rule_score']:.4f}"},
                        {"Component": "Behavioral Multiplier", "Value": f"{details['behavioral_multiplier']:.2f}x"},
                        {"Component": "Final Score", "Value": f"{details['final_score']:.4f}"}
                    ])
                    st.table(score_df)
                    
                    # Disqualification reason / Honeypot warning
                    if details["is_honeypot"]:
                        st.markdown("<span class='badge badge-danger'>🚨 Honeypot profile detected</span>", unsafe_allow_html=True)
                    elif details["is_disqualified"]:
                        st.markdown("<span class='badge badge-danger'>❌ Disqualified</span>", unsafe_allow_html=True)
                        st.write(f"Reasons: {', '.join(details['disqualification_reasons'])}")
                    else:
                        st.markdown("<span class='badge badge-success'>✅ Active Qualified Profile</span>", unsafe_allow_html=True)
                        
                with col_right:
                    st.write("### AI Generated Shortlisting Reasoning")
                    st.info(details["reasoning"])
                    
                    st.write("#### Behavioral Signals & Recruiter Availability")
                    sig_col1, sig_col2 = st.columns(2)
                    with sig_col1:
                        st.write(f"- **Last Active:** {signals.get('last_active_date', 'N/A')}")
                        st.write(f"- **Open to Work:** {signals.get('open_to_work_flag', 'N/A')}")
                        st.write(f"- **Recruiter Response Rate:** {signals.get('recruiter_response_rate', 'N/A')}")
                        st.write(f"- **Notice Period:** {signals.get('notice_period_days', 'N/A')} days")
                    with sig_col2:
                        st.write(f"- **GitHub Activity:** {signals.get('github_activity_score', 'N/A')}")
                        st.write(f"- **Profile Completeness:** {signals.get('profile_completeness_score', 'N/A')}%")
                        st.write(f"- **Willing to Relocate:** {signals.get('willing_to_relocate', 'N/A')}")
                        st.write(f"- **Preferred Work Mode:** {signals.get('preferred_work_mode', 'N/A')}")
                        
                    st.write("#### Top Skills & Endorsements")
                    skills_data = []
                    for s in skills[:8]:
                        skills_data.append({
                            "Skill": s.get("name"),
                            "Proficiency": s.get("proficiency"),
                            "Months": s.get("duration_months"),
                            "Endorsements": s.get("endorsements")
                        })
                    st.dataframe(pd.DataFrame(skills_data), hide_index=True)
                    
                    st.write("#### Career History (Top 3)")
                    for job in career_history[:3]:
                        st.write(f"**{job.get('title')}** at *{job.get('company')}* ({job.get('duration_months', 0)} months)")
                        st.caption(job.get('description', ''))
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
            
    else:
        # Initial Landing view when no file is uploaded
        st.info("👋 Welcome! Please upload a candidate JSONL or JSON file using the uploader above to get started.")
        
        # Show mock screenshot design or layout details
        st.subheader("💡 Engine Architecture & Formula Details")
        
        arch_col1, arch_col2 = st.columns(2)
        with arch_col1:
            st.write("""
            #### 1. Precomputation & Retrieval (Offline)
            - Streams candidates, checks schema validity, and flags honeypots.
            - Compiles key text sections (Headline, Summary, Top Skills, Career Histories) into a dense profile blob.
            - Encodes blobs into 384-dimensional embeddings using `all-MiniLM-L6-v2`.
            
            #### 2. Vectorized Cosine Similarity (Online)
            - Computes semantic similarity between the candidate embedding and Job Description embedding.
            - High-speed matrix-vector dot product runs in under 0.1s.
            """)
        with arch_col2:
            st.write("""
            #### 3. Core Score Weights
            - **Skill Match Score (35%):** Weighted keyword match based on must-haves, proficiency, endorsements, and duration.
            - **Semantic Match Score (30%):** Cosine similarity between candidate and JD text representation.
            - **Career Fit Score (25%):** Evaluates product-company experience ratio, tenure stability, and title relevance.
            - **Rule Score (10%):** Evaluates explicit requirements like experience years, notice period, and CS degree.
            
            #### 4. Availability Multiplier
            - Multiplies the base score by a factor of `0.5x` to `1.2x` using 23 availability and reliability metrics.
            """)

if __name__ == "__main__":
    main()
