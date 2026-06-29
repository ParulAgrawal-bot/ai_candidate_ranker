import re
from datetime import datetime

SERVICE_COMPANIES = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'
}

DISQUALIFYING_TITLES = {
    'marketing', 'hr', 'recruiter', 'sales', 'business analyst', 'product owner', 
    'operations manager', 'project manager', 'scrum master', 'administrative'
}

ENGINEERING_TITLES = {
    'engineer', 'developer', 'scientist', 'architect', 'programmer', 'coder', 'lead', 'tech'
}

def extract_features(candidate):
    """
    Extracts structured features and evaluates hard disqualifiers for a candidate profile.
    """
    profile = candidate.get('profile', {})
    career_history = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    education = candidate.get('education', [])
    signals = candidate.get('redrob_signals', {})
    
    # 1. Evaluate Hard Disqualifiers
    disqualified = False
    reasons = []
    
    # Extract skills set and job descriptions text
    skill_names = {str(s.get('name', '')).lower() for s in skills}
    all_job_titles = [str(j.get('title', '')).lower() for j in career_history]
    all_job_descs = [str(j.get('description', '')).lower() for j in career_history]
    all_companies = [str(j.get('company', '')).lower() for j in career_history]
    
    # A. Title is not an engineering title
    current_title = str(profile.get('current_title', '')).lower()
    if current_title:
        # If the title is explicitly in disqualifying set and does not contain engineering keywords
        is_disqualified_title = any(t in current_title for t in DISQUALIFYING_TITLES)
        has_eng_keywords = any(e in current_title for e in ENGINEERING_TITLES)
        if is_disqualified_title and not has_eng_keywords:
            disqualified = True
            reasons.append("Non-engineering current title")
            
    # B. Entire career spent at service companies (TCS, Infosys, etc.)
    if all_companies:
        spent_at_services = []
        for co in all_companies:
            is_service = any(sc in co for sc in SERVICE_COMPANIES)
            spent_at_services.append(is_service)
        if all(spent_at_services):
            disqualified = True
            reasons.append("Entire career spent at service-only companies")
            
    # C. Pure research / academic background with no production deployment
    is_academic = True
    has_production = False
    prod_keywords = ['production', 'deploy', 'scale', 'ship', 'product', 'system', 'customer', 'user', 'aws', 'docker', 'kubernetes', 'ci/cd']
    
    if career_history:
        for job in career_history:
            company = str(job.get('company', '')).lower()
            title = str(job.get('title', '')).lower()
            desc = str(job.get('description', '')).lower()
            
            # Check if company is academic
            is_job_academic = any(ac in company for ac in ['university', 'institute', 'college', 'academy', 'lab', 'research center'])
            is_title_academic = any(ac in title for ac in ['phd', 'student', 'postdoc', 'research assistant', 'fellow'])
            
            if not (is_job_academic or is_title_academic):
                is_academic = False
                
            if any(pk in desc for pk in prod_keywords):
                has_production = True
                
        if is_academic and not has_production:
            disqualified = True
            reasons.append("Pure research/academic background without production deployment")
            
    # D. Only LangChain + OpenAI under 12 months with no pre-LLM ML background
    has_llm_skills = any(ls in skill_names for ls in ['langchain', 'openai', 'llm', 'gpt', 'prompt engineering'])
    has_traditional_ml = any(ms in skill_names for ms in [
        'scikit-learn', 'pandas', 'numpy', 'pytorch', 'tensorflow', 'keras', 
        'machine learning', 'deep learning', 'data science', 'statistics', 
        'random forest', 'xgboost', 'svm', 'regression', 'clustering'
    ])
    
    years_exp = float(profile.get('years_of_experience', 0) or 0)
    if has_llm_skills and not has_traditional_ml and years_exp < 1.0:
        disqualified = True
        reasons.append("Only LangChain/OpenAI experience under 12 months with no traditional ML foundation")
        
    # E. No code written in last 18 months ("architect-only" roles)
    current_job = career_history[0] if career_history else {}
    current_desc = str(current_job.get('description', '')).lower()
    current_job_title = str(current_job.get('title', '')).lower()
    
    is_architect_or_mgr = any(t in current_job_title for t in ['architect', 'manager', 'director', 'vp', 'lead'])
    has_coding_in_desc = any(ck in current_desc for ck in ['python', 'code', 'write', 'develop', 'build', 'git', 'program', 'implementation'])
    
    # Simple rule: if they are in architect/mgr role and current description shows no hands-on coding signals
    if is_architect_or_mgr and not has_coding_in_desc and len(career_history) > 0:
        # Check duration of current role
        duration = current_job.get('duration_months', 0) or 0
        if duration > 18:
            disqualified = True
            reasons.append("Architect-only or manager-only role with no coding in the last 18 months")
            
    # F. Primary CV/speech/robotics and no NLP/IR work
    has_cv_robotics = any(cv in skill_names for cv in [
        'computer vision', 'opencv', 'image processing', 'cnn', 'object detection', 
        'speech recognition', 'asr', 'audio', 'robotics', 'ros', 'slam'
    ])
    has_nlp_ir = any(nlp in skill_names for nlp in [
        'nlp', 'natural language processing', 'search', 'information retrieval', 
        'elasticsearch', 'faiss', 'qdrant', 'milvus', 'pinecone', 'embeddings', 
        'sentence-transformers', 'bert', 'transformers', 'gpt', 'llm', 'indexing', 
        'vector search'
    ])
    if has_cv_robotics and not has_nlp_ir:
        disqualified = True
        reasons.append("Primary CV/robotics/speech focus with no NLP/IR foundation")

    # 2. Extract Rule Score features
    # A. Location check: Pune / Noida / Hyderabad / Mumbai
    cand_loc = str(profile.get('location', '')).lower()
    in_tier_1_loc = any(loc in cand_loc for loc in ['pune', 'noida', 'delhi', 'ncr', 'hyderabad', 'mumbai'])
    
    # B. Notice period
    notice_days = signals.get('notice_period_days', 60)
    try:
        notice_days = int(notice_days)
    except (ValueError, TypeError):
        notice_days = 60
        
    # C. CS/engineering degree
    has_cs_degree = False
    for edu in education:
        deg = str(edu.get('degree', '')).lower()
        field = str(edu.get('field_of_study', '')).lower()
        if any(f in field or f in deg for f in ['computer', 'cs', 'it', 'software', 'engineering', 'technology', 'information']):
            has_cs_degree = True
            break
            
    candidate["is_disqualified"] = disqualified
    candidate["disqualification_reasons"] = reasons
    
    return {
        "candidate_id": profile.get('candidate_id', 'UNKNOWN'),
        "years_of_experience": years_exp,
        "in_tier_1_loc": in_tier_1_loc,
        "notice_period_days": notice_days,
        "has_cs_degree": has_cs_degree,
        "is_disqualified": disqualified
    }

def calculate_rule_score(candidate):
    """
    Calculates the Rule Score (0.0 to 1.0) based on candidate features:
    - Years of experience band (5-9 years preferred)
    - Location fit (Pune/Noida preferred)
    - Notice period (sub-30 day preferred)
    - Education field (CS/Engineering degree)
    """
    if isinstance(candidate, dict) and "in_tier_1_loc" in candidate:
        feats = candidate
    else:
        feats = extract_features(candidate)
        
    # 1. Years of experience score (5-9 years preferred)
    yoe = feats.get("years_of_experience", 0.0)
    if 5.0 <= yoe <= 9.0:
        yoe_score = 1.0
    elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 12.0:
        yoe_score = 0.8
    elif yoe < 4.0:
        yoe_score = max(0.2, yoe / 4.0)
    else:
        yoe_score = max(0.4, 1.0 - (yoe - 12.0) * 0.05)
        
    # 2. Location score (Pune/Noida/Delhi/NCR/Hyderabad/Mumbai are tier 1)
    is_tier_1 = feats.get("in_tier_1_loc", False)
    willing_relocate = False
    if isinstance(candidate, dict) and "redrob_signals" in candidate:
        willing_relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
        
    if is_tier_1:
        loc_score = 1.0
    elif willing_relocate:
        loc_score = 0.7
    else:
        loc_score = 0.3
        
    # 3. Notice period score (sub-30 days preferred)
    notice = feats.get("notice_period_days", 60)
    if notice < 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.1
        
    # 4. Education score (CS/engineering degree)
    has_cs = feats.get("has_cs_degree", False)
    edu_score = 1.0 if has_cs else 0.5
    
    # Combine scores with weights
    rule_score = 0.3 * yoe_score + 0.3 * loc_score + 0.2 * notice_score + 0.2 * edu_score
    return max(0.0, min(1.0, rule_score))

