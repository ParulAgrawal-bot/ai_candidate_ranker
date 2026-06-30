def generate_reasoning(candidate, rank, score):
    """
    Return a short human-readable reasoning string for a shortlisted candidate.
    Minimal implementation to ensure CI doesn't fail. Replace with full logic as needed.
    """
    profile = candidate.get("profile", {})
    name = profile.get("name") or profile.get("candidate_id") or "Candidate"
    skills = profile.get("skills", []) or []
    top_skills = ", ".join(skills[:3]) if skills else "N/A"

    return (
        f"{rank}. {name} — score {score:.4f}. "
        f"Top skills: {top_skills}."
    )
