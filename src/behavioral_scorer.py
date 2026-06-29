from datetime import datetime
import math

def compute_behavioral_multiplier(candidate, reference_date_str="2026-06-28"):
    """
    Computes a behavioral multiplier for a candidate based on 23 behavioral signals.
    Range: 0.5x to 1.2x.
    """
    signals = candidate.get('redrob_signals', {})
    profile = candidate.get('profile', {})
    
    multiplier = 1.0
    
    # Reference date for activity calculation
    try:
        ref_date = datetime.strptime(reference_date_str, "%Y-%m-%d")
    except ValueError:
        ref_date = datetime.now()
        
    # 1. last_active_date: date string (e.g. "2026-05-15")
    last_active_str = signals.get('last_active_date', '')
    if last_active_str:
        try:
            # support YYYY-MM-DD or other formats
            active_date = datetime.strptime(last_active_str.split('T')[0], "%Y-%m-%d")
            days_inactive = (ref_date - active_date).days
            if days_inactive > 90:
                multiplier *= 0.6
            elif days_inactive <= 15:
                multiplier *= 1.1  # boost for high recency
        except Exception:
            pass
            
    # 2. open_to_work_flag: bool
    if signals.get('open_to_work_flag') is True:
        multiplier *= 1.15
        
    # 3. recruiter_response_rate: 0 to 1
    rrr = signals.get('recruiter_response_rate', None)
    if rrr is not None:
        try:
            rrr = float(rrr)
            if rrr < 0.3:
                multiplier *= 0.7
            elif rrr > 0.7:
                multiplier *= 1.1
        except ValueError:
            pass
            
    # 4. avg_response_time_hours: >= 0
    arth = signals.get('avg_response_time_hours', None)
    if arth is not None:
        try:
            arth = float(arth)
            if arth < 24:
                multiplier *= 1.05
            elif arth > 72:
                multiplier *= 0.8
        except ValueError:
            pass
            
    # 5. interview_completion_rate: 0 to 1
    icr = signals.get('interview_completion_rate', None)
    if icr is not None:
        try:
            icr = float(icr)
            if icr < 0.5:
                multiplier *= 0.6
        except ValueError:
            pass
            
    # 6. offer_acceptance_rate: -1 to 1 (-1 means no history)
    oar = signals.get('offer_acceptance_rate', None)
    if oar is not None:
        try:
            oar = float(oar)
            if oar != -1 and oar < 0.3:
                multiplier *= 0.9
        except ValueError:
            pass
            
    # 7. notice_period_days: 0 to 180
    npd = signals.get('notice_period_days', None)
    if npd is not None:
        try:
            npd = int(npd)
            if npd < 30:
                multiplier *= 1.15
            elif npd > 90:
                multiplier *= 0.7
        except ValueError:
            pass
            
    # 8. willing_to_relocate: bool & location fit
    # JD location is Pune/Noida
    cand_loc = str(profile.get('location', '')).lower()
    in_target_location = 'pune' in cand_loc or 'noida' in cand_loc or 'ncr' in cand_loc
    
    willing_relocate = signals.get('willing_to_relocate', False)
    if not in_target_location:
        if willing_relocate is True:
            multiplier *= 1.1  # partial credit boost
        else:
            multiplier *= 0.8  # penalty for not local and unwilling to relocate
            
    # 9. preferred_work_mode: hybrid / flexible
    pwm = str(signals.get('preferred_work_mode', '')).lower()
    if 'hybrid' in pwm or 'flexible' in pwm or 'remote' in pwm:
        multiplier *= 1.05
        
    # 10. github_activity_score: -1 to 100 (-1 means no Github)
    gas = signals.get('github_activity_score', -1)
    if gas is not None:
        try:
            gas = float(gas)
            if gas > 60:
                multiplier *= 1.1
        except ValueError:
            pass
            
    # 11. profile_completeness_score: 0 to 100
    pcs = signals.get('profile_completeness_score', None)
    if pcs is not None:
        try:
            pcs = float(pcs)
            if pcs < 50:
                multiplier *= 0.9
            elif pcs > 80:
                multiplier *= 1.05
        except ValueError:
            pass
            
    # 12. skill_assessment_scores: dict
    sas = signals.get('skill_assessment_scores', {})
    if isinstance(sas, dict) and sas:
        # If there are verified test scores, calculate average
        scores = [float(v) for v in sas.values() if isinstance(v, (int, float))]
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score > 70:
                multiplier *= 1.1
            elif avg_score < 40:
                multiplier *= 0.9
                
    # 13. verified_email & verified_phone: bool
    verified_email = signals.get('verified_email', True)
    verified_phone = signals.get('verified_phone', True)
    if (verified_email is False) and (verified_phone is False):
        multiplier *= 0.9
        
    # 14. saved_by_recruiters_30d: int
    sbr = signals.get('saved_by_recruiters_30d', 0)
    if sbr is not None:
        try:
            sbr = int(sbr)
            if sbr > 3:
                multiplier *= 1.1
        except ValueError:
            pass
            
    # 15. applications_submitted_30d: int
    ast = signals.get('applications_submitted_30d', 0)
    if ast is not None:
        try:
            ast = int(ast)
            if ast > 0:
                multiplier *= 1.02
        except ValueError:
            pass

    # Apply Honeypot discount if flagged in validation
    # "Flag profiles with 10+ skills marked 'expert' — statistically unrealistic → apply a credibility discount."
    val_flags = candidate.get('validation_flags', {})
    if val_flags.get('too_many_expert_skills') is True:
        multiplier *= 0.5  # credibility discount
        
    # Clip multiplier to the range [0.5, 1.2]
    return max(0.5, min(1.2, multiplier))
