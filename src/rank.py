import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

# Add the parent directory to the path to make imports work when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_parser import stream_candidates
from src.skill_scorer import calculate_skill_score
from src.career_scorer import calculate_career_score
from src.feature_builder import calculate_rule_score
from src.behavioral_scorer import compute_behavioral_multiplier
from src.hybrid_scorer import calculate_hybrid_score
from src.reasoning_gen import generate_reasoning

def calculate_cosine_similarity(vec_a, vec_b):
    """
    Computes cosine similarity between two numpy arrays.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def main():
    parser = argparse.ArgumentParser(description="Rank candidates based on Job Description match.")
    parser.add_argument("--candidates", type=str, required=True, help="Path to input candidates JSONL (or JSONL.gz) file")
    parser.add_argument("--out", type=str, required=True, help="Path to output ranked CSV file")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory containing precomputed artifacts")
    args = parser.parse_args()

    # Define paths to artifacts
    embeddings_path = os.path.join(args.artifacts_dir, "embeddings.npy")
    jd_embedding_path = os.path.join(args.artifacts_dir, "jd_embedding.npy")
    features_path = os.path.join(args.artifacts_dir, "features.pkl")

    # Check if artifacts exist
    if not (os.path.exists(embeddings_path) and os.path.exists(jd_embedding_path) and os.path.exists(features_path)):
        print(f"Error: Precomputed artifacts not found in '{args.artifacts_dir}'.", file=sys.stderr)
        print("Please run the offline precomputation script first:", file=sys.stderr)
        print("  python precompute/precompute_embeddings.py --candidates <candidates_file>", file=sys.stderr)
        sys.exit(1)

    print("Loading precomputed artifacts...")
    candidate_embeddings = np.load(embeddings_path)
    jd_embedding = np.load(jd_embedding_path)
    
    with open(features_path, "rb") as f:
        features_data = pickle.load(f)

    id_to_index = features_data.get("id_to_index", {})
    
    print(f"Loaded {len(candidate_embeddings)} candidate embeddings of dimension {candidate_embeddings.shape[1]}")

    results = []

    print("Streaming and scoring candidates...")
    count = 0
    
    # Pre-calculate semantic scores for all candidates in features_data if we want to vectorize
    # However, since we want to be robust to subset/new candidates, we will look up by candidate_id
    for candidate in stream_candidates(args.candidates):
        profile = candidate.get('profile', {})
        candidate_id = profile.get('candidate_id', 'UNKNOWN')
        
        # 1. Semantic Score (cosine similarity against JD)
        semantic_score = 0.0
        if candidate_id in id_to_index:
            idx = id_to_index[candidate_id]
            cand_emb = candidate_embeddings[idx]
            semantic_score = calculate_cosine_similarity(cand_emb, jd_embedding)
        else:
            # Fallback if candidate was not in precomputed set
            # We can log this but to keep output clean, we just set semantic_score to 0.0 or a default
            pass
            
        # 2. Skill Match Score (0 to 1)
        skill_score = calculate_skill_score(candidate)
        
        # 3. Career Score (0 to 1)
        career_score = calculate_career_score(candidate)
        
        # 4. Rule Score (0 to 1)
        rule_score = calculate_rule_score(candidate)
        
        # 5. Behavioral Multiplier (0.5 to 1.2)
        behavioral_multiplier = compute_behavioral_multiplier(candidate)
        
        # 6. Hybrid Score combination
        score_details = calculate_hybrid_score(
            semantic_score=semantic_score,
            skill_score=skill_score,
            career_score=career_score,
            rule_score=rule_score,
            behavioral_multiplier=behavioral_multiplier,
            candidate=candidate
        )
        
        results.append({
            "candidate_id": candidate_id,
            "score": score_details["final_score"],
            "candidate": candidate  # keep reference to generate reasoning
        })
        
        count += 1
        if count % 20000 == 0:
            print(f"Scored {count} candidates...")

    print(f"Scoring complete. Total candidates processed: {count}")

    # Sort candidates
    # Sort criteria:
    # 1. Score descending
    # 2. Candidate ID ascending (as tie-breaker)
    print("Sorting and selecting top-100 shortlisted candidates...")
    results.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))
    
    top_100 = results[:100]
    
    # Generate short-listed rows
    output_rows = []
    for rank_idx, item in enumerate(top_100):
        rank = rank_idx + 1
        candidate_id = item["candidate_id"]
        score = item["score"]
        candidate = item["candidate"]
        
        # Generate reasoning for top 100
        reasoning = generate_reasoning(candidate, rank, score)
        
        output_rows.append({
            "candidate_id": candidate_id,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reasoning
        })

    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write output to CSV
    df_output = pd.DataFrame(output_rows)
    # Ensure correct columns order
    df_output = df_output[["candidate_id", "rank", "score", "reasoning"]]
    
    df_output.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved top-100 shortlisted candidates to {args.out}")
    print("Ranking process completed successfully!")

if __name__ == "__main__":
    main()
