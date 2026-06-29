import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_parser import stream_candidates
import os
import pickle
import numpy as np
import argparse
from src.data_parser import stream_candidates
from src.feature_builder import extract_features
from src.embedder import build_candidate_text_blob, encode_text

def main():
    parser = argparse.ArgumentParser(description="Precompute candidate features and embeddings offline.")
    parser.add_argument("--candidates", type=str, default="data/candidates.jsonl.gz", help="Path to candidates JSONL file")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save precomputed artifacts")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    print(f"Streaming candidates from {args.candidates}...")
    candidate_ids = []
    text_blobs = []
    features_list = []
    candidates_cache = {}

    count = 0
    for cand in stream_candidates(args.candidates):
        profile = cand.get('profile', {})
        cid = profile.get('candidate_id', f"UNKNOWN_{count}")
        
        # Extract features (which flags disqualifiers)
        feats = extract_features(cand)
        features_list.append(feats)
        candidate_ids.append(cid)
        candidates_cache[cid] = cand
        
        # Build text blob for embedding
        text_blob = build_candidate_text_blob(cand)
        text_blobs.append(text_blob)
        
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} candidates...")

    print(f"Total candidates parsed: {count}")
    
    # 2. Generate and save candidate embeddings
    print("Generating candidate embeddings (this might take some time on CPU)...")
    # Batch size can be adjusted for memory efficiency
    embeddings = encode_text(text_blobs, model_name='all-MiniLM-L6-v2')
    embeddings_path = os.path.join(args.artifacts_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Saved candidate embeddings to {embeddings_path}. Shape: {embeddings.shape}")

    # 3. Create and embed the Job Description (JD)
    jd_text = (
        "Senior AI Engineer. Mandatory skills: Production embeddings-based retrieval "
        "(sentence-transformers, BGE, E5, OpenAI embeddings deployed to real users), "
        "Vector DB and hybrid search experience (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, FAISS, Elasticsearch), "
        "Evaluation frameworks for ranking (NDCG, MRR, MAP, A/B testing, offline-to-online correlation), "
        "strong Python programming. Nice-to-haves: LLM fine-tuning (LoRA, QLoRA, PEFT), "
        "learning-to-rank experience (XGBoost or LTR), distributed systems or large-scale inference optimization, "
        "HR-tech/recruiting/marketplace product experience. Location: Pune / Noida preferred. "
        "Shipper over researcher, product company tenure, and long-term commitment."
    )
    print("Generating Job Description (JD) embedding...")
    jd_embedding = encode_text(jd_text, model_name='all-MiniLM-L6-v2')
    jd_emb_path = os.path.join(args.artifacts_dir, "jd_embedding.npy")
    np.save(jd_emb_path, jd_embedding)
    print(f"Saved JD embedding to {jd_emb_path}. Shape: {jd_embedding.shape}")

    # 4. Save features dictionary
    features_data = {
        "candidate_ids": candidate_ids,
        "features_list": features_list,
        "id_to_index": {cid: idx for idx, cid in enumerate(candidate_ids)},
        "candidates_cache": candidates_cache  # contains validation_flags, etc.
    }
    features_path = os.path.join(args.artifacts_dir, "features.pkl")
    with open(features_path, "wb") as f:
        pickle.dump(features_data, f)
    print(f"Saved features and candidate metadata to {features_path}")
    print("Precomputation completed successfully!")

if __name__ == "__main__":
    main()
