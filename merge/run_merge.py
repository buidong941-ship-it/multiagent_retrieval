import os
import sys
import pickle
import numpy as np
import sqlite3
from pathlib import Path
from collections import defaultdict

# Add project root to PYTHONPATH so we can import from video_retrieval
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "video_retrieval"))

from database.faiss.faiss_client import FaissVectorDatabase
from database.bm25.bm25_index import BM25OcrIndex
from config.settings import get_settings
from database.metadata.schema import Base
from sqlalchemy import create_engine

def init_master_metadata(db_path: Path):
    """Ensure master DB has correct schema."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return sqlite3.connect(str(db_path))

def run_merge():
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    
    if not input_dir.exists():
        print(f"Error: Input directory not found at {input_dir}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
        
    print(f"Scanning for Kaggle outputs in {input_dir}...")
    print(f"Output will be saved to: {output_dir}")
    
    settings = get_settings()
    
    # OVERRIDE SETTINGS TO SAVE TO OUTPUT FOLDER INSTEAD OF LIVE DB
    settings.faiss.index_dir = str(output_dir / "faiss")
    settings.ocr.bm25_index_path = str(output_dir / "bm25_ocr.pkl")
    
    # ---------------------------------------------------------
    # 1. FAISS VECTOR MERGE
    # ---------------------------------------------------------
    collections = defaultdict(list)
    for pkl_file in input_dir.rglob("*.pkl"):
        collection_name = pkl_file.stem 
        if "embeddings" in collection_name:
            collections[collection_name].append(pkl_file)
            
    db = FaissVectorDatabase(settings.faiss)
    
    for collection_name, pkl_files in collections.items():
        print(f"\n{'='*50}")
        print(f"Merging FAISS collection: {collection_name}")
        print(f"Found {len(pkl_files)} parts.")
        
        all_embeddings = []
        all_metadata = []
        
        for pkl_path in pkl_files:
            print(f"  - Loading {pkl_path.name} from {pkl_path.parent.name}...")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            vectors_dict = data.get("vectors", {})
            metadata_dict = data.get("metadata", {})
            
            added_count = 0
            for old_id, vector in vectors_dict.items():
                meta = metadata_dict.get(old_id, {})
                all_embeddings.append(vector)
                all_metadata.append(meta)
                added_count += 1
                
            print(f"    -> Extracted {added_count} vectors.")
                
        if not all_embeddings:
            continue
            
        final_embeddings = np.vstack(all_embeddings)
        final_ids = list(range(len(all_metadata)))
        
        print(f"\nTotal frames merged: {len(final_ids)}")
        
        if collection_name in db.indices:
            del db.indices[collection_name]
        
        db.insert(
            collection_name=collection_name,
            ids=final_ids,
            embeddings=final_embeddings,
            metadata=all_metadata
        )
        db.flush(collection_name)
        print(f"✅ Saved FAISS index for {collection_name} to {settings.faiss.index_dir}!")

    # ---------------------------------------------------------
    # 2. SQLITE METADATA MERGE
    # ---------------------------------------------------------
    print(f"\n{'='*50}")
    print("Merging SQLite Metadata (metadata.db)...")
    
    master_db_path = output_dir / "metadata.db"
    
    # If there's an old one in output, delete it to avoid appending to previous runs
    if master_db_path.exists():
        master_db_path.unlink()
    
    master_conn = init_master_metadata(master_db_path)
    
    db_files = list(input_dir.rglob("*.db"))
    if not db_files:
        print("No metadata.db files found in input directory.")
    else:
        for i, db_file in enumerate(db_files):
            print(f"  - Attaching and merging {db_file.name} from {db_file.parent.name}...")
            alias = f"part_db_{i}"
            master_conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(db_file),))
            # Use INSERT OR IGNORE to prevent duplicate primary keys
            master_conn.execute(f"INSERT OR IGNORE INTO videos SELECT * FROM {alias}.videos")
            master_conn.execute(f"INSERT OR IGNORE INTO frames SELECT * FROM {alias}.frames")
            master_conn.execute(f"INSERT OR IGNORE INTO detections SELECT * FROM {alias}.detections")
            master_conn.commit()
            master_conn.execute(f"DETACH DATABASE {alias}")
            
        print("  - Normalizing file paths for local machine...")
        
        # Normalize Frame Paths
        cursor = master_conn.execute("SELECT frame_id, frame_path FROM frames")
        frame_updates = []
        for fid, fpath in cursor.fetchall():
            if not fpath: continue
            normalized_path = fpath.replace('\\', '/')
            if 'data/frames/' in normalized_path:
                suffix = normalized_path.split('data/frames/')[-1]
                new_path = str(PROJECT_ROOT / "video_retrieval" / "data" / "frames" / suffix)
                if new_path != fpath:
                    frame_updates.append((new_path, fid))
        
        if frame_updates:
            master_conn.executemany("UPDATE frames SET frame_path = ? WHERE frame_id = ?", frame_updates)
            
        # Normalize Video Paths
        cursor = master_conn.execute("SELECT id, video_path FROM videos")
        video_updates = []
        for vid, vpath in cursor.fetchall():
            if not vpath: continue
            normalized_path = vpath.replace('\\', '/')
            if 'data/videos/' in normalized_path:
                suffix = normalized_path.split('data/videos/')[-1]
                new_path = str(PROJECT_ROOT / "video_retrieval" / "data" / "videos" / suffix)
                if new_path != vpath:
                    video_updates.append((new_path, vid))
                    
        if video_updates:
            master_conn.executemany("UPDATE videos SET video_path = ? WHERE id = ?", video_updates)
            
        master_conn.commit()
        print(f"✅ Successfully merged and normalized all metadata into {master_db_path}!")

    # ---------------------------------------------------------
    # 3. BM25 REBUILD FROM MASTER DB
    # ---------------------------------------------------------
    print(f"\n{'='*50}")
    print("Rebuilding BM25 OCR Index from merged metadata.db...")
    cursor = master_conn.execute("SELECT frame_id, ocr_text FROM frames WHERE ocr_text IS NOT NULL AND ocr_text != ''")
    rows = cursor.fetchall()
    master_conn.close()
    
    if rows:
        frame_ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        
        print(f"Extracted {len(texts)} OCR text records.")
        bm25 = BM25OcrIndex(settings.ocr)
        bm25.build(texts, frame_ids)
        bm25.save()
        print(f"✅ Successfully rebuilt and saved BM25 OCR Index to {settings.ocr.bm25_index_path}!")
    else:
        print("No OCR text found in metadata.db. Skipping BM25 rebuild.")

if __name__ == "__main__":
    print("🚀 Starting Full Kaggle Merge Tool (FAISS + Metadata + BM25)...")
    run_merge()
    print("\n✅ All merge processes completed successfully!")
