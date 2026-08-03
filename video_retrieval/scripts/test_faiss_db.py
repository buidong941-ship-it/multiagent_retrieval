import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil
import numpy as np

from config.database_config import FaissConfig
from database.faiss.faiss_client import FaissVectorDatabase


def test_faiss_basic():
    test_dir = Path("./data/test_faiss_db")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    config = FaissConfig(index_dir=test_dir.as_posix())
    db = FaissVectorDatabase(config)

    # Generate mock embeddings (normalized)
    np.random.seed(42)
    dim = 1152
    num_samples = 10

    vecs = np.random.randn(num_samples, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    ids = list(range(1, num_samples + 1))
    metadata = [
        {
            "frame_id": f"video01_frame_{i}",
            "video_id": "video01",
            "frame_idx": i,
            "timestamp": float(i * 1.5),
        }
        for i in ids
    ]

    # Test Insert
    db.insert("clip_embeddings", ids, vecs, metadata)
    assert db.count("clip_embeddings") == num_samples, "Count mismatch after insert"
    print(f"[OK] Inserted {num_samples} vectors into 'clip_embeddings'. Total count: {db.count('clip_embeddings')}")

    # Test Search
    query = vecs[0]  # Exact match for the first vector
    hits = db.search("clip_embeddings", query, top_k=3)
    assert len(hits) == 3, f"Expected 3 hits, got {len(hits)}"
    assert hits[0]["frame_id"] == "video01_frame_1", f"Expected top hit video01_frame_1, got {hits[0]['frame_id']}"
    print(f"[OK] Top hit: {hits[0]['frame_id']} with score: {hits[0]['score']:.4f}")

    # Test get_embeddings_by_ids (for Reranker)
    emb_map = db.get_embeddings_by_ids(["video01_frame_1", "video01_frame_2"], "clip_embeddings")
    assert "video01_frame_1" in emb_map, "video01_frame_1 missing from emb_map"
    assert np.allclose(emb_map["video01_frame_1"], vecs[0], atol=1e-5), "Embedding vector mismatch"
    print("[OK] get_embeddings_by_ids verified successfully.")

    # Test Flush & Load
    db.flush()
    db.close()

    # Re-open database from disk
    db_reloaded = FaissVectorDatabase(config)
    assert db_reloaded.count("clip_embeddings") == num_samples, "Count mismatch after reload"
    reloaded_hits = db_reloaded.search("clip_embeddings", query, top_k=1)
    assert reloaded_hits[0]["frame_id"] == "video01_frame_1"
    print("[OK] Persisted FAISS index reloaded and searched successfully.")

    # Clean up test artifacts
    db_reloaded.close()
    if test_dir.exists():
        shutil.rmtree(test_dir)
    print("\nALL FAISS VECTOR DATABASE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_faiss_basic()
