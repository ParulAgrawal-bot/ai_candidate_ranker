def calculate_hybrid_score(semantic_score, skill_score, career_score, rule_score, behavioral_multiplier, candidate):
    """
    Integrates all score components into the final score using the formula:
    Final Score = (0.35 * Skill + 0.30 * Semantic + 0.25 * Career + 0.10 * Rule) * Behavioral Multiplier
    
    If the candidate is flagged as a honeypot or triggers a hard disqualifier, the final score is set to 0.0.
    """
    # Base weighted score
    weighted_score = (
        0.35 * skill_score +
        0.30 * semantic_score +
        0.25 * career_score +
        0.10 * rule_score
    )
    
    final_score = weighted_score * behavioral_multiplier
    
    # Check for Honeypots
    validation_flags = candidate.get('validation_flags', {})
    is_honeypot = validation_flags.get('is_honeypot', False)
    
    # Check for Hard Disqualifiers (computed during career_scorer/feature_builder)
    disqualified = candidate.get('is_disqualified', False)
    disqualification_reasons = candidate.get('disqualification_reasons', [])
    
    # If the candidate is a honeypot or disqualified, zero out their score
    if is_honeypot or disqualified:
        final_score = 0.0
        
    return {
        "final_score": final_score,
        "base_weighted_score": weighted_score,
        "semantic_score": semantic_score,
        "skill_score": skill_score,
        "career_score": career_score,
        "rule_score": rule_score,
        "behavioral_multiplier": behavioral_multiplier,
        "is_honeypot": is_honeypot,
        "is_disqualified": disqualified,
        "disqualification_reasons": disqualification_reasons
    }
