import jinja2
import hashlib

def generate_reasoning(candidate, rank, score):
    """
    Generates a 1-2 sentence reasoning string for the candidate.
    Must be fact-grounded, rank-consistent, and highly varied.
    """
    profile = candidate.get('profile', {})
    skills = candidate.get('skills', [])
    signals = candidate.get('redrob_signals', {})
    
    # 1. Gather facts
    candidate_id = profile.get('candidate_id', 'UNKNOWN')
    yoe = profile.get('years_of_experience', 0)
    current_role = profile.get('current_title', 'Engineer')
    company = profile.get('current_company', 'Product Co')
    
    # Extract relevant skills matching must-haves
    must_have_keywords = {
        'embeddings': ['sentence-transformers', 'bge', 'e5', 'embeddings'],
        'vector_db': ['pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'elasticsearch', 'opensearch'],
        'eval': ['ndcg', 'mrr', 'map', 'a/b testing', 'learning-to-rank', 'ltr'],
        'python': ['python']
    }
    
    found_skills = []
    for skill in skills:
        skill_name = str(skill.get('name', '')).lower()
        for cat, keywords in must_have_keywords.items():
            for kw in keywords:
                if kw in skill_name and kw not in found_skills:
                    found_skills.append(kw)
                    
    # Format key skills
    if found_skills:
        key_skills_str = ", ".join(found_skills[:3])
    else:
        key_skills_str = "Python/ML"
        
    # Availability and behavioral signals
    rrr = signals.get('recruiter_response_rate', 0.5)
    last_active_str = signals.get('last_active_date', '')
    
    # Days active calculation (mock/relative to 2026-06-28)
    days_active = 30 # default
    if last_active_str:
        try:
            # Assuming format like "2026-06-20"
            from datetime import datetime
            active_date = datetime.strptime(last_active_str.split('T')[0], "%Y-%m-%d")
            ref_date = datetime.strptime("2026-06-28", "%Y-%m-%d")
            days_active = max(0, (ref_date - active_date).days)
        except Exception:
            pass
            
    notice = signals.get('notice_period_days', 60)
    github = signals.get('github_activity_score', -1)
    
    # 2. Identify concerns
    concerns = []
    if notice > 60:
        concerns.append(f"notice period of {notice} days exceeds preference")
    
    location = profile.get('location', '')
    is_local = any(loc in str(location).lower() for loc in ['pune', 'noida', 'delhi', 'ncr', 'hyderabad', 'mumbai'])
    willing_relocate = signals.get('willing_to_relocate', False)
    if not is_local:
        if willing_relocate:
            concerns.append("requires relocation to Pune/Noida (willing)")
        else:
            concerns.append("not in target location and unwilling to relocate")
            
    # Hash candidate ID to get deterministic variety
    hash_val = int(hashlib.md5(candidate_id.encode('utf-8')).hexdigest(), 16)
    
    # 3. Formulate templates based on rank
    # Rank 1-10 (Enthusiastic & highly detailed)
    # Rank 11-50 (Strong, positive, objective)
    # Rank 51-100 (Measured, cautious, highlighting concerns)
    
    concern_text = ""
    if concerns:
        concern_text = "Concern: " + " and ".join(concerns) + "."
        
    if rank <= 10:
        # Templates for Top Tier
        templates = [
            "Exceptional match with {{ yoe }} years in ML. Currently {{ role }} at {{ company }}, shipping {{ skills }}. Outstanding availability (recruiter response {{ rrr }}, active {{ days }}d ago). {{ concern }}",
            "A top-tier candidate offering {{ yoe }} years of applied AI experience, currently {{ role }} at {{ company }}. Strong hands-on work with {{ skills }}. {{ concern }}",
            "Excellent profile with {{ yoe }} years of experience as {{ role }} at {{ company }}. Deep skills in {{ skills }} combined with very strong behavioral metrics (response rate {{ rrr }}). {{ concern }}"
        ]
    elif rank <= 50:
        # Templates for Mid Tier
        templates = [
            "Strong fit with {{ yoe }} years of experience. Working as {{ role }} at {{ company }} with skills in {{ skills }}. Active {{ days }} days ago. {{ concern }}",
            "Demonstrated competency in {{ skills }} across {{ yoe }} years. Currently {{ role }} at {{ company }} with a solid response rate of {{ rrr }}. {{ concern }}",
            "Solid experience ({{ yoe }} years) matching the JD must-haves: {{ skills }}. Active profile with notice period of {{ notice }} days. {{ concern }}"
        ]
    else:
        # Templates for Lower Tier
        templates = [
            "Candidate has {{ yoe }} years of experience, currently {{ role }} at {{ company }}. Profile shows some experience with {{ skills }}. {{ concern }}",
            "Fits basic requirements with {{ yoe }} years of experience and exposure to {{ skills }}. {{ concern }}",
            "Has {{ yoe }} years of experience. Current title is {{ role }} and skills include {{ skills }}. {{ concern }}"
        ]
        
    template_str = templates[hash_val % len(templates)]
    template = jinja2.Template(template_str)
    
    reasoning = template.render(
        yoe=f"{yoe:.1f}" if isinstance(yoe, (int, float)) else yoe,
        role=current_role,
        company=company,
        skills=key_skills_str,
        rrr=f"{rrr:.2f}" if isinstance(rrr, (int, float)) else rrr,
        days=days_active,
        notice=notice,
        github=github,
        concern=concern_text
    )
    
    # Post-process to ensure formatting is clean (no double spaces/periods)
    reasoning = reasoning.replace("  ", " ").strip()
    if not reasoning.endswith("."):
        reasoning += "."
        
    return reasoning
