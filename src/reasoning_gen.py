def generate_reasoning(candidate, rank, score):
    """
    Generate a short reasoning string for a shortlisted candidate.

    This minimal implementation is intentionally simple so CI won't fail due to
    missing dependencies. Extend this function with more advanced reasoning
    logic if needed.
    """
    if not candidate:
        return f"Rank #{rank} with score {score:.4f}. No candidate data available."

    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    candidate_id = profile.get("candidate_id") or profile.get("id") or "UNKNOWN"
    name = profile.get("name") or candidate_id

    # Collect top skills if available
    skills = profile.get("skills") or []
    if isinstance(skills, list):
        top_skills = ", ".join(str(s) for s in skills[:5]) if skills else "N/A"
    else:
        top_skills = str(skills)

    return f"{name} (id={candidate_id}) ranked #{rank} with score {score:.4f}. Key skills: {top_skills}."
