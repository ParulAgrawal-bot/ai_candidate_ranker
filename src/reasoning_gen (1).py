"""
Reasoning Generator Module - Member C responsibility.

For each top-100 candidate, generates 1-2 sentences of fact-grounded reasoning
explaining their rank. Pulls only from fields that actually exist in the raw
candidate dict (per candidate_schema.json) plus whatever scoring metadata the
pipeline has already attached to it - never invents facts.

Tone varies by rank tier:
    1-10    : confident, leads with strongest evidence
    11-30   : positive but measured, one strength + one honest gap if present
    31-70   : balanced, names the gap plainly
    71-100  : cautious/qualified - "included despite X", not "great fit"

Call signature (matches rank.py exactly):
    generate_reasoning(candidate: dict, rank: int, score: float) -> str

By the time `candidate` reaches rank.py, feature_builder.extract_features()
has already run and attached:
    candidate['is_disqualified']         -> bool
    candidate['disqualification_reasons'] -> list[str], human-readable sentences,
                                              e.g. "Non-engineering current title",
                                              "Pure research/academic background
                                              without production deployment"
And skill_scorer.py / data_parser.py may have attached:
    candidate['validation_flags']        -> dict, e.g. {'is_honeypot': True,
                                              'expert_zero_duration': True}

This module reads those fields directly (they're already plain English, no
re-translation needed) and falls back gracefully to schema-only fields
(profile, career_history, skills, redrob_signals) if upstream enrichment
hasn't happened for some reason.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rank tier configuration
# ---------------------------------------------------------------------------

def get_rank_tier(rank: int) -> str:
    """Map a rank (1-100) to a tone tier."""
    if rank <= 10:
        return "top"
    elif rank <= 30:
        return "strong"
    elif rank <= 70:
        return "moderate"
    else:
        return "borderline"


# ---------------------------------------------------------------------------
# Fact extraction helpers - read only from the raw candidate dict, never invent
# ---------------------------------------------------------------------------

def _format_years(years) -> Optional[str]:
    if years is None:
        return None
    try:
        years = float(years)
    except (TypeError, ValueError):
        return None
    if years == int(years):
        return f"{int(years)} yrs"
    return f"{years:.1f} yrs"


def _top_evidenced_skills(skills: List[Dict[str, Any]], max_skills: int = 2) -> List[str]:
    """
    Strongest skills with REAL evidence (duration_months > 0). Never returns a
    skill whose only evidence is a proficiency label with no duration behind it.
    Mirrors skill_scorer.py's own honeypot rule: expert + 0 duration = no credit.
    """
    evidenced = [
        s for s in skills
        if s.get("duration_months", 0) and s.get("duration_months", 0) > 0
    ]
    if not evidenced:
        return []

    def strength(s):
        prof = str(s.get("proficiency", "")).lower()
        prof_rank = {"expert": 3, "advanced": 3, "intermediate": 2, "mid": 2, "beginner": 1, "basic": 1}.get(prof, 0)
        return (prof_rank, s.get("duration_months", 0), s.get("endorsements", 0))

    evidenced.sort(key=strength, reverse=True)
    return [s.get("name", "") for s in evidenced[:max_skills] if s.get("name")]


def _stuffed_skill_names(skills: List[Dict[str, Any]], max_names: int = 2) -> List[str]:
    """Skills claimed at expert level with 0 duration - the exact pattern
    skill_scorer.py zeroes out via prof_weight = 0.0."""
    names = [
        s.get("name", "")
        for s in skills
        if "expert" in str(s.get("proficiency", "")).lower() and (s.get("duration_months", 0) or 0) == 0
    ]
    return names[:max_names]


def _looks_like_keyword_stuffing(skills: List[Dict[str, Any]]) -> bool:
    """Local heuristic, independent of skill_scorer's validation_flags, so this
    module still works even if that flag wasn't attached upstream."""
    if any("expert" in str(s.get("proficiency", "")).lower() and (s.get("duration_months", 0) or 0) == 0 for s in skills):
        return True
    expert_count = sum(1 for s in skills if "expert" in str(s.get("proficiency", "")).lower())
    return expert_count >= 10


def _current_or_latest_job(career_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not career_history:
        return None
    current = next((j for j in career_history if j.get("is_current")), None)
    return current or career_history[0]


def _company_context(career_history: List[Dict[str, Any]]) -> Optional[str]:
    job = _current_or_latest_job(career_history)
    if not job:
        return None
    company = job.get("company")
    size = job.get("company_size")
    if not company:
        return None
    if size:
        return f"{company} ({size} employees)"
    return company


def _avg_tenure_months(career_history: List[Dict[str, Any]]) -> Optional[float]:
    if not career_history:
        return None
    durations = [j.get("duration_months", 0) or 0 for j in career_history]
    if not durations:
        return None
    return sum(durations) / len(durations)


def _availability_signals(signals: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Split availability/trust signals from redrob_signals into 'positive' and
    'negative' buckets so a good signal (e.g. active recently) never gets
    merged into the same fragment as a bad one (e.g. long notice period)
    under a single 'Concern:' label.
    """
    last_active_str = signals.get("last_active_date")
    days_inactive = None
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
            days_inactive = (datetime.now() - last_active).days
        except (ValueError, TypeError):
            days_inactive = None

    response_rate = signals.get("recruiter_response_rate")
    notice = signals.get("notice_period_days")
    open_to_work = signals.get("open_to_work_flag")

    positive, negative = [], []

    if days_inactive is not None:
        if days_inactive <= 14:
            positive.append("active recently")
        elif days_inactive > 90:
            negative.append(f"inactive {days_inactive} days")

    if response_rate is not None:
        if response_rate >= 0.7:
            positive.append(f"recruiter_response_rate {response_rate:.2f}")
        elif response_rate < 0.3:
            negative.append(f"recruiter_response_rate only {response_rate:.2f}")

    if notice is not None and notice > 90:
        negative.append(f"notice period {notice} days")

    if open_to_work and not positive:
        positive.append("open to work")

    return {"positive": positive, "negative": negative}


# ---------------------------------------------------------------------------
# Clause builders
# ---------------------------------------------------------------------------

def _build_headline_clause(profile: Dict[str, Any]) -> str:
    title = profile.get("current_title") or "Candidate"
    years_str = _format_years(profile.get("years_of_experience"))
    if years_str:
        return f"{title} with {years_str}"
    return title


def _build_skill_clause(skills: List[Dict[str, Any]], stuffer_flag: bool) -> Optional[str]:
    if stuffer_flag:
        stuffed = _stuffed_skill_names(skills)
        if stuffed:
            shown = ", ".join(stuffed)
            return f"lists {shown} at expert level with no recorded duration"
    top_skills = _top_evidenced_skills(skills)
    if top_skills:
        return f"evidenced strength in {', '.join(top_skills)}"
    return None


def _build_career_clause(career_history: List[Dict[str, Any]]) -> Optional[str]:
    company_ctx = _company_context(career_history)
    avg_tenure = _avg_tenure_months(career_history)

    parts = []
    if company_ctx:
        parts.append(f"currently at {company_ctx}")
    if avg_tenure is not None:
        if avg_tenure >= 36:
            parts.append("long, stable tenure history")
        elif avg_tenure < 12:
            parts.append(f"average tenure ~{avg_tenure:.0f} months")

    if not parts:
        return None
    return "; ".join(parts)


def _build_concern_clause(
    disqualification_reasons: List[str],
    is_honeypot: bool,
    availability_concern: Optional[str],
    stuffer_flag: bool,
) -> Optional[str]:
    """
    Honest gap, if one exists, in priority order:
    honeypot > hard disqualifier > keyword stuffing > availability concern.
    Returns None only if there's genuinely nothing to flag.
    """
    if is_honeypot:
        return "Concern: profile flagged as internally inconsistent (honeypot check)"
    if disqualification_reasons:
        # These are already human-readable sentences from feature_builder.py
        # (e.g. "Non-engineering current title") - use directly, lowercase the
        # first letter so it reads naturally after "Concern:".
        reason = disqualification_reasons[0]
        reason = reason[0].lower() + reason[1:] if reason else reason
        return f"Concern: {reason}"
    if stuffer_flag:
        return "Concern: skill claims not backed by duration evidence"
    if availability_concern:
        return f"Concern: {availability_concern}"
    return None


# ---------------------------------------------------------------------------
# Main entry point - this is the function rank.py imports and calls directly
# ---------------------------------------------------------------------------

def generate_reasoning(candidate: Dict[str, Any], rank: int, score: float) -> str:
    """
    Build a 1-2 sentence, fact-grounded reasoning string for one candidate.

    candidate: raw candidate dict (per candidate_schema.json), enriched
               in-place upstream by feature_builder.extract_features()
               with 'is_disqualified' / 'disqualification_reasons', and
               possibly by skill_scorer / data_parser with 'validation_flags'.
    rank:      1-100, final position in the shortlist.
    score:     final_score from hybrid_scorer. Not echoed verbatim in the
               text (the CSV already has a score column) but used to confirm
               tone matches a near-zero score for disqualified/honeypot rows.
    """
    try:
        profile = candidate.get("profile", {}) or {}
        career_history = candidate.get("career_history", []) or []
        skills = candidate.get("skills", []) or []
        signals = candidate.get("redrob_signals", {}) or {}
        validation_flags = candidate.get("validation_flags", {}) or {}

        disqualification_reasons = candidate.get("disqualification_reasons", []) or []
        is_honeypot = bool(validation_flags.get("is_honeypot", False))
        stuffer_flag = (
            bool(validation_flags.get("expert_zero_duration", False))
            or _looks_like_keyword_stuffing(skills)
        )

        tier = get_rank_tier(rank)

        headline = _build_headline_clause(profile)
        skill_clause = _build_skill_clause(skills, stuffer_flag)
        career_clause = _build_career_clause(career_history)
        availability = _availability_signals(signals)
        availability_positive = "; ".join(availability["positive"]) if availability["positive"] else None
        availability_concern = "; ".join(availability["negative"]) if availability["negative"] else None
        concern_clause = _build_concern_clause(
            disqualification_reasons, is_honeypot, availability_concern, stuffer_flag
        )

        primary_facts = [c for c in [skill_clause, career_clause] if c]

        # A disqualified/honeypot candidate should never read as a confident
        # top-tier pick even if it somehow lands at a low rank index (e.g. a
        # thin candidate pool). Force borderline tone whenever score is ~0
        # or a hard disqualifier/honeypot fired, regardless of numeric rank.
        forced_borderline = is_honeypot or bool(disqualification_reasons) or score <= 0.0
        effective_tier = "borderline" if forced_borderline else tier

        if effective_tier == "top":
            sentence = headline
            if primary_facts:
                sentence += "; " + "; ".join(primary_facts[:2])
            if availability_positive:
                sentence += f"; {availability_positive}"
            sentence += "."
            return sentence

        elif effective_tier == "strong":
            sentence = headline
            if primary_facts:
                sentence += "; " + primary_facts[0]
            sentence += "."
            if concern_clause:
                sentence += f" {concern_clause}."
            elif availability_positive:
                sentence += f" {availability_positive.capitalize()}."
            return sentence

        elif effective_tier == "moderate":
            sentence = headline
            if primary_facts:
                sentence += "; " + primary_facts[0]
            sentence += "."
            if concern_clause:
                sentence += f" {concern_clause}."
            elif availability_positive:
                sentence += f" Note: {availability_positive}."
            else:
                sentence += " No major concerns identified, but profile is average rather than standout."
            return sentence

        else:  # borderline (includes forced-borderline for disqualified/honeypot)
            sentence = "Included with reservations: " + headline
            if primary_facts:
                sentence += "; " + primary_facts[0]
            sentence += "."
            if concern_clause:
                sentence += f" {concern_clause}."
            else:
                sentence += " Concern: profile is marginal against JD must-haves; ranks in shortlist tail."
            return sentence

    except Exception as e:
        logger.warning(f"Reasoning generation failed for {candidate.get('candidate_id', 'UNKNOWN')}: {e}")
        # Fail safe, never block CSV generation for 100k candidates - but make
        # the failure obvious for QA rather than silently producing a generic line.
        return "Reasoning unavailable - manual review required."


# ---------------------------------------------------------------------------
# Quick manual self-test (run directly: python src/reasoning_gen.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    top_candidate = {
        "candidate_id": "CAND_0001234",
        "profile": {"current_title": "Senior ML Engineer", "years_of_experience": 7.2},
        "career_history": [
            {"company": "Razorpay", "company_size": "501-1000", "is_current": True, "duration_months": 30}
        ],
        "skills": [
            {"name": "sentence-transformers", "proficiency": "expert", "duration_months": 30, "endorsements": 12},
            {"name": "FAISS", "proficiency": "advanced", "duration_months": 18, "endorsements": 5},
        ],
        "redrob_signals": {
            "last_active_date": datetime.now().strftime("%Y-%m-%d"),
            "recruiter_response_rate": 0.82,
            "notice_period_days": 20,
        },
        "is_disqualified": False,
        "disqualification_reasons": [],
    }

    disqualified_candidate = {
        "candidate_id": "CAND_0009999",
        "profile": {"current_title": "HR Manager", "years_of_experience": 8.0},
        "career_history": [
            {"company": "TCS", "company_size": "10001+", "is_current": True, "duration_months": 36}
        ],
        "skills": [
            {"name": "LLMs", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
            {"name": "RAG", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
        ],
        "redrob_signals": {
            "last_active_date": "2025-01-01",
            "recruiter_response_rate": 0.4,
            "notice_period_days": 30,
        },
        "is_disqualified": True,
        "disqualification_reasons": ["Non-engineering current title"],
    }

    honeypot_candidate = {
        "candidate_id": "CAND_0005555",
        "profile": {"current_title": "AI Engineer", "years_of_experience": 8.0},
        "career_history": [
            {"company": "StartupX", "company_size": "1-10", "is_current": True, "duration_months": 96}
        ],
        "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 96, "endorsements": 20}],
        "redrob_signals": {"last_active_date": "2026-06-20", "recruiter_response_rate": 0.9, "notice_period_days": 15},
        "is_disqualified": False,
        "disqualification_reasons": [],
        "validation_flags": {"is_honeypot": True},
    }

    moderate_candidate = {
        "candidate_id": "CAND_0004444",
        "profile": {"current_title": "Backend Engineer", "years_of_experience": 5.0},
        "career_history": [
            {"company": "Freshworks", "company_size": "1001-5000", "is_current": True, "duration_months": 14}
        ],
        "skills": [{"name": "Elasticsearch", "proficiency": "intermediate", "duration_months": 14, "endorsements": 2}],
        "redrob_signals": {"last_active_date": "2026-05-20", "recruiter_response_rate": 0.55, "notice_period_days": 45},
        "is_disqualified": False,
        "disqualification_reasons": [],
    }

    print("Rank 3 (top):       ", generate_reasoning(top_candidate, rank=3, score=0.91))
    print("Rank 45 (moderate): ", generate_reasoning(moderate_candidate, rank=45, score=0.41))
    print("Rank 88 (disq.):    ", generate_reasoning(disqualified_candidate, rank=88, score=0.0))
    print("Rank 97 (honeypot): ", generate_reasoning(honeypot_candidate, rank=97, score=0.0))
