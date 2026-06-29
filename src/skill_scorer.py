import math

MUST_HAVE_MAPPINGS = {
    "embeddings_retrieval": [
        "sentence-transformers", "sentence transformers", "bge", "e5", "openai embeddings", 
        "embeddings", "vector embedding", "dense retrieval", "bi-encoder", "cross-encoder"
    ],
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "faiss", "elasticsearch", 
        "vector database", "vector search", "hybrid search"
    ],
    "evaluation_ranking": [
        "ndcg", "mrr", "map", "ab testing", "a/b testing", "learning-to-rank", "ltr", 
        "evaluation framework", "ranking evaluation"
    ],
    "python": [
        "python", "strong python"
    ]
}

PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 1.0,
    "intermediate": 0.7,
    "mid": 0.7,
    "beginner": 0.3,
    "basic": 0.3
}

def get_proficiency_weight(proficiency):
    if not proficiency:
        return 0.5
    prof_lower = str(proficiency).lower()
    for key, weight in PROFICIENCY_WEIGHTS.items():
        if key in prof_lower:
            return weight
    return 0.5

def calculate_skill_score(candidate):
    """
    Calculates the normalized Skill Match Score (0.0 to 1.0) based on 4 must-have categories:
    1. Embeddings/Retrieval
    2. Vector DB
    3. Evaluation/Ranking
    4. Python
    """
    skills = candidate.get('skills', [])
    val_flags = candidate.get('validation_flags', {})
    
    category_scores = {cat: 0.0 for cat in MUST_HAVE_MAPPINGS}
    
    # Group skills by their matched category
    for skill in skills:
        name = str(skill.get('name', '')).lower()
        proficiency = str(skill.get('proficiency', ''))
        duration = skill.get('duration_months', 0) or 0
        endorsements = skill.get('endorsements', 0) or 0
        
        # Honeypot check: expert with 0 duration gets zeroed out
        prof_weight = get_proficiency_weight(proficiency)
        if 'expert' in proficiency.lower() and duration == 0:
            prof_weight = 0.0
            
        for cat, keywords in MUST_HAVE_MAPPINGS.items():
            if any(kw in name for kw in keywords):
                # Calculate score for this skill
                # We use (1.0 + log1p(endorsements)) so candidates with 0 endorsements still get credit.
                skill_score = prof_weight * min(duration / 24.0, 1.0) * (1.0 + math.log1p(endorsements))
                
                # Keep the maximum score for each category
                category_scores[cat] = max(category_scores[cat], skill_score)
                
    # Normalize skill score (max possible score per category is roughly 1.0 * 1.0 * (1.0 + ln(1 + 100)) = ~5.6, 
    # but we can scale or just cap it at 1.0 after dividing by a normalization factor)
    # Let's say a reasonable high-scoring skill has 10 endorsements. log1p(10) = 2.4, so total term is 3.4.
    # We will normalize each category by dividing by 3.5 and capping at 1.0.
    normalized_cat_scores = []
    for cat, score in category_scores.items():
        norm_score = min(1.0, score / 3.5)
        normalized_cat_scores.append(norm_score)
        
    # Return average of the 4 must-have categories
    final_skill_score = sum(normalized_cat_scores) / len(MUST_HAVE_MAPPINGS)
    
    # If the candidate was flagged for expert zero duration or is a honeypot, we can penalize the skill score heavily
    if val_flags.get('expert_zero_duration') is True:
        final_skill_score *= 0.1
        
    return final_skill_score
