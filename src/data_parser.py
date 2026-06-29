import json
import gzip
import re
from datetime import datetime

def stream_candidates(file_path):
    """
    Streams candidate records from a JSONL file (supports gzip compression).
    Yields parsed and validated candidate dictionaries.
    """
    open_func = gzip.open if file_path.endswith('.gz') else open
    mode = 'rt' if file_path.endswith('.gz') else 'r'
    
    with open_func(file_path, mode, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
                yield parse_and_validate_candidate(candidate)
            except json.JSONDecodeError:
                continue

def extract_founding_year(text):
    """
    Attempts to extract a founding year from text (e.g. company name or description).
    Looks for patterns like 'founded 2015', 'founded in 2012', etc.
    """
    if not text:
        return None
    match = re.search(r'\bfounded\s+(?:in\s+)?([12][0-9]{3})\b', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Check parenthetical years as well, e.g. "Acme Corp (2018)"
    match = re.search(r'\b(?:[a-zA-Z0-9_\-\s]+)\s*\(([12][0-9]{3})\)\b', text)
    if match:
        return int(match.group(1))
    return None

def parse_and_validate_candidate(candidate):
    """
    Normalizes candidate fields and runs honeypot validation checks.
    Adds a 'validation_flags' dictionary to the candidate record.
    """
    profile = candidate.get('profile', {})
    career_history = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    
    candidate_id = profile.get('candidate_id', 'UNKNOWN')
    
    # Flags dictionary
    flags = {
        "impossible_company_timeline": False,
        "expert_zero_duration": False,
        "too_many_expert_skills": False,
        "experience_duration_mismatch": False,
        "is_honeypot": False
    }
    
    # Check 1: Validate company founding date vs. experience duration
    current_year = datetime.now().year
    for job in career_history:
        company_name = job.get('company', '')
        desc = job.get('description', '')
        start_date_str = job.get('start_date', '')
        
        founding_year = extract_founding_year(company_name) or extract_founding_year(desc)
        
        if founding_year:
            # Parse start year
            start_year = None
            if start_date_str:
                # Assuming date format YYYY-MM-DD or YYYY
                match = re.search(r'^([12][0-9]{3})', start_date_str)
                if match:
                    start_year = int(match.group(1))
            
            if start_year and start_year < founding_year:
                flags["impossible_company_timeline"] = True
                
            # If duration is longer than company existence
            duration_months = job.get('duration_months', 0) or 0
            company_age_years = current_year - founding_year
            if duration_months / 12.0 > max(1.0, company_age_years + 2.0):  # Allow 2 years buffer for rough dates
                flags["impossible_company_timeline"] = True

    # Check 2: Skill duration_months against proficiency ("expert" with 0 months)
    expert_skills_count = 0
    for skill in skills:
        name = skill.get('name', '')
        proficiency = str(skill.get('proficiency', '')).lower()
        duration = skill.get('duration_months', 0) or 0
        
        if 'expert' in proficiency:
            expert_skills_count += 1
            if duration == 0:
                flags["expert_zero_duration"] = True
                
    # Check 3: Flag profiles with 10+ skills marked "expert"
    if expert_skills_count >= 10:
        flags["too_many_expert_skills"] = True
        
    # Check 4: Cross-check years_of_experience against the sum of career history durations
    claimed_exp = float(profile.get('years_of_experience', 0) or 0)
    
    # Calculate sum of career history durations
    total_months = 0
    for job in career_history:
        total_months += job.get('duration_months', 0) or 0
    total_history_exp = total_months / 12.0
    
    # Mismatch is flagged if there is a massive discrepancy (e.g. difference > 3.0 years)
    if claimed_exp > 0 and abs(claimed_exp - total_history_exp) > 3.0:
        flags["experience_duration_mismatch"] = True
        
    # Final honeypot decision
    # If it fails critical consistency checks, flag as a honeypot
    if (flags["impossible_company_timeline"] or 
        flags["expert_zero_duration"] or 
        (flags["experience_duration_mismatch"] and claimed_exp > 0 and total_history_exp == 0)):
        flags["is_honeypot"] = True
        
    candidate["validation_flags"] = flags
    return candidate
