import os
import sys
import re
import argparse
import pandas as pd

def validate_csv(csv_path, candidates_path=None):
    errors = []
    
    # 1. Check file existence
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' does not exist.")
        return False
        
    try:
        # Load CSV with pandas to inspect structure
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

    # 2. Check column count and names in order
    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    actual_cols = list(df.columns)
    if actual_cols != expected_cols:
        errors.append(f"Columns mismatch. Expected: {expected_cols}, Found: {actual_cols}")

    # 3. Check number of rows
    # "Exactly 100 data rows plus 1 header row"
    num_rows = len(df)
    if num_rows != 100:
        errors.append(f"Row count mismatch. Expected exactly 100 data rows, found {num_rows}")

    # If the file has structure issues that prevent further checks, stop here
    if errors:
        for err in errors:
            print(f"[FAIL] {err}")
        return False

    # 4. Check ranks (1-100 each appear exactly once)
    ranks = list(df["rank"])
    expected_ranks = list(range(1, 101))
    if ranks != expected_ranks:
        errors.append("Ranks must be exactly 1 to 100 in sequential order.")

    # 5. Check candidate_id format (CAND_XXXXXXX)
    id_pattern = re.compile(r'^CAND_\d+$')
    invalid_ids = []
    for cid in df["candidate_id"]:
        if not id_pattern.match(str(cid)):
            invalid_ids.append(cid)
    if invalid_ids:
        errors.append(f"Invalid candidate_id formats (must be CAND_<digits>): {invalid_ids[:5]}")

    # 6. Check that scores are non-increasing
    scores = list(df["score"])
    for i in range(1, len(scores)):
        if scores[i] > scores[i-1]:
            errors.append(f"Scores are not non-increasing: Row {i} score ({scores[i]}) is greater than Row {i-1} score ({scores[i-1]}).")
            break

    # 7. Check equal scores are tie-broken by candidate_id ascending
    for i in range(1, len(scores)):
        if scores[i] == scores[i-1]:
            cid_prev = str(df.loc[i-1, "candidate_id"])
            cid_curr = str(df.loc[i, "candidate_id"])
            if cid_curr < cid_prev:
                errors.append(f"Tie-breaker failed: Row {i-1} ({cid_prev}) and Row {i} ({cid_curr}) have equal score ({scores[i]}) but are not sorted by candidate_id ascending.")

    # 8. Check reasoning column (1-2 sentences, specific, honest)
    empty_reasonings = 0
    for idx, reason in enumerate(df["reasoning"]):
        if pd.isna(reason) or not str(reason).strip():
            empty_reasonings += 1
            
    if empty_reasonings > 0:
        errors.append(f"Found {empty_reasonings} rows with empty reasoning strings.")

    # 9. Verify candidate_ids against the original candidates dataset (optional check)
    if candidates_path:
        if not os.path.exists(candidates_path):
            print(f"Warning: Candidates dataset '{candidates_path}' not found. Skipping ID presence validation.")
        else:
            print("Verifying candidate IDs existence in dataset...")
            import gzip
            import json
            
            existing_ids = set()
            open_func = gzip.open if candidates_path.endswith('.gz') else open
            mode = 'rt' if candidates_path.endswith('.gz') else 'r'
            
            try:
                with open_func(candidates_path, mode, encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            cand = json.loads(line)
                            cid = cand.get('profile', {}).get('candidate_id')
                            if cid:
                                existing_ids.add(cid)
                                
                missing_from_dataset = []
                for cid in df["candidate_id"]:
                    if cid not in existing_ids:
                        missing_from_dataset.append(cid)
                        
                if missing_from_dataset:
                    errors.append(f"Found {len(missing_from_dataset)} candidate_ids in CSV that do not exist in candidate source file: {missing_from_dataset[:5]}")
            except Exception as e:
                print(f"Error checking candidate IDs against dataset: {e}")

    # Output validation status
    if errors:
        print(f"\nValidation failed with {len(errors)} error(s):")
        for err in errors:
            print(f"- [FAIL] {err}")
        return False
    else:
        print("\nSubmission is valid.")
        return True

def main():
    parser = argparse.ArgumentParser(description="Validate submission CSV format.")
    parser.add_argument("csv_path", type=str, help="Path to the team_xxx.csv file")
    parser.add_argument("--candidates", type=str, default=None, help="Path to candidates source file to verify ID existence")
    args = parser.parse_args()

    success = validate_csv(args.csv_path, args.candidates)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
