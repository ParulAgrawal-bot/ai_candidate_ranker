SERVICE_KEYWORDS = [
    'client', 'consulting', 'services', 'project delivery', 'outsourcing', 'staffing', 
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'
]

PRODUCT_KEYWORDS = [
    'product', 'saas', 'platform', 'subscription', 'b2b', 'b2c', 'e-commerce', 'fintech', 
    'marketplace', 'shipped', 'users', 'scale', 'infrastructure'
]

def classify_company_type(company_name, description):
    """
    Classifies a company as 'product' or 'service' based on text analysis.
    Returns 1.0 for product, 0.2 for service, and 0.6 for neutral/unknown.
    """
    company_lower = company_name.lower()
    desc_lower = description.lower() if description else ''
    
    # Direct service company match
    for sk in ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'tata consultancy']:
        if sk in company_lower:
            return 0.2
            
    # Check keyword overlap
    service_score = sum(1 for kw in SERVICE_KEYWORDS if kw in desc_lower)
    product_score = sum(1 for kw in PRODUCT_KEYWORDS if kw in desc_lower)
    
    if product_score > service_score:
        return 1.0
    elif service_score > product_score:
        return 0.2
    return 0.6

def calculate_career_score(candidate):
    """
    Calculates the Career Score (0.0 to 1.0) for a candidate.
    """
    career_history = candidate.get('career_history', [])
    profile = candidate.get('profile', {})
    
    # If the candidate is disqualified by career flags, return 0.0
    if candidate.get('is_disqualified', False):
        # We only zero out if the reason is career-related (pure research or services-only)
        career_reasons = [r for r in candidate.get('disqualification_reasons', []) 
                          if 'career' in r or 'research' in r or 'architect' in r]
        if career_reasons:
            return 0.0
            
    if not career_history:
        return 0.0
        
    total_months = 0
    product_company_months = 0
    job_count = len(career_history)
    
    # 1. Product vs Service company tenure ratio
    for job in career_history:
        duration = job.get('duration_months', 0) or 0
        company = job.get('company', '')
        desc = job.get('description', '')
        
        co_type_weight = classify_company_type(company, desc)
        total_months += duration
        product_company_months += duration * co_type_weight
        
    product_ratio = (product_company_months / total_months) if total_months > 0 else 0.5
    
    # 2. Job-hop penalty (average tenure < 18 months)
    avg_tenure_months = (total_months / job_count) if job_count > 0 else 0
    tenure_multiplier = 1.0
    if avg_tenure_months < 12:
        tenure_multiplier = 0.5
    elif avg_tenure_months < 18:
        tenure_multiplier = 0.8
    elif avg_tenure_months >= 36:
        tenure_multiplier = 1.1  # boost for long tenure / high stability
        
    # 3. Title-relevance score (evaluate roles in career history)
    title_scores = []
    for i, job in enumerate(career_history):
        title = str(job.get('title', '')).lower()
        
        # Determine weight by recency (more weight to current/recent jobs)
        recency_weight = 1.0 if i == 0 else (0.5 if i == 1 else 0.2)
        
        title_score = 0.2  # default baseline
        if any(kw in title for kw in ['senior machine learning', 'senior ml', 'senior ai', 'lead ml', 'lead ai', 'principal ml', 'ml lead', 'ai lead']):
            title_score = 1.0
        elif any(kw in title for kw in ['machine learning', 'ml engineer', 'ai engineer', 'nlp engineer', 'search engineer', 'retrieval engineer']):
            title_score = 0.9
        elif any(kw in title for kw in ['data scientist', 'data science', 'ml research', 'ai research']):
            title_score = 0.8
        elif any(kw in title for kw in ['software engineer', 'backend engineer', 'systems engineer', 'data engineer', 'developer']):
            title_score = 0.7
        elif any(kw in title for kw in ['research assistant', 'postdoc', 'researcher']):
            title_score = 0.4
            
        title_scores.append(title_score * recency_weight)
        
    # Normalized title score (sum of weighted scores divided by sum of weights)
    weights = [1.0 if i == 0 else (0.5 if i == 1 else 0.2) for i in range(len(career_history))]
    sum_weights = sum(weights[:len(title_scores)])
    title_relevance = (sum(title_scores) / sum_weights) if sum_weights > 0 else 0.5
    
    # Combine components
    # Career Score = (0.4 * product_ratio + 0.6 * title_relevance) * tenure_multiplier
    career_score = (0.4 * product_ratio + 0.6 * title_relevance) * tenure_multiplier
    
    return max(0.0, min(1.0, career_score))
