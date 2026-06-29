import os
from sentence_transformers import SentenceTransformer

_model = None

def get_model(model_name='all-MiniLM-L6-v2'):
    """
    Returns a cached SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        # Load local model if cached, otherwise download (allowed during setup/precompute)
        _model = SentenceTransformer(model_name)
    return _model

def encode_text(text_or_texts, model_name='all-MiniLM-L6-v2'):
    """
    Encodes text or list of texts into embeddings.
    """
    model = get_model(model_name)
    return model.encode(text_or_texts, convert_to_numpy=True)

def build_candidate_text_blob(candidate):
    """
    Extracts and compiles a text blob representing the candidate's profile
    for embedding purposes. Combines headline, summary, top skills, and career descriptions.
    """
    profile = candidate.get('profile', {})
    skills = candidate.get('skills', [])
    career_history = candidate.get('career_history', [])
    
    parts = []
    
    # Title & Headline
    headline = profile.get('headline', '')
    if headline:
        parts.append(f"Headline: {headline}")
        
    summary = profile.get('summary', '')
    if summary:
        parts.append(f"Summary: {summary}")
        
    # Current Title & Company
    current_title = profile.get('current_title', '')
    current_company = profile.get('current_company', '')
    if current_title:
        parts.append(f"Current Role: {current_title} at {current_company}")
        
    # Top Skills
    if skills:
        skill_names = [s.get('name', '') for s in skills if s.get('name')]
        parts.append(f"Skills: {', '.join(skill_names[:15])}")
        
    # Career History Description
    jobs_parts = []
    for job in career_history[:3]:  # Top 3 jobs
        title = job.get('title', '')
        company = job.get('company', '')
        desc = job.get('description', '')
        job_str = f"{title} at {company}"
        if desc:
            job_str += f" - {desc}"
        jobs_parts.append(job_str)
        
    if jobs_parts:
        parts.append(f"Experience:\n" + "\n".join(jobs_parts))
        
    return "\n\n".join(parts)
