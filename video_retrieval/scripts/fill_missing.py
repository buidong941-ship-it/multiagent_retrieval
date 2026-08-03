"""
Script to resume missing OCR and YOLO processing flexibly.
Supports dynamic frame directory resolution and selective skipping.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from database.metadata.metadata_db import MetadataDatabase
from services.detection.detection_service import DetectionService
from utils.logging_utils import get_logger
from sqlalchemy import text

log = get_logger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Resume missing OCR and YOLO data flexibly.")
    parser.add_argument("--frames-dir", type=str, default=None, help="Dynamic frames directory (does not modify DB)")
    parser.add_argument("--fix-paths", action="store_true", help="Fix Kaggle/absolute paths in DB to local output_dir")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR recovery")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO recovery")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 rebuild")
    args = parser.parse_args()

    settings = get_settings()
    meta_db = MetadataDatabase()
    
    await meta_db.init_db()
    
    # 0. Fix paths if requested
    if args.fix_paths:
        log.info("=" * 60)
        log.info("STEP 0: FIXING KAGGLE PATHS TO LOCAL PATHS IN DATABASE")
        log.info("=" * 60)
        async with meta_db.session() as session:
            result = await session.execute(text("SELECT frame_id, frame_path FROM frames"))
            frame_updates = []
            for fid, fpath in result.fetchall():
                if not fpath: continue
                normalized_path = fpath.replace('\\', '/')
                if 'data/frames/' in normalized_path:
                    suffix = normalized_path.split('data/frames/')[-1]
                    new_path = str(Path(settings.frame.output_dir).resolve() / suffix)
                    if new_path != fpath:
                        frame_updates.append({"fpath": new_path, "fid": fid})
                        
            if frame_updates:
                log.info(f"Normalizing {len(frame_updates)} frame paths in DB...")
                for update in frame_updates:
                    await session.execute(
                        text("UPDATE frames SET frame_path = :fpath WHERE frame_id = :fid"), 
                        update
                    )
                await session.commit()
                log.info("Paths normalized successfully!")
            else:
                log.info("All paths in DB are already correct!")
    
    # Common function to resolve paths in-memory
    def resolve_frame_path(fpath: str) -> str:
        if args.frames_dir and fpath:
            normalized = fpath.replace('\\', '/')
            if 'data/frames/' in normalized:
                suffix = normalized.split('data/frames/')[-1]
                return str(Path(args.frames_dir).resolve() / suffix)
            else:
                # If path doesn't have data/frames/, assume it's just video_id/frame_name
                parts = Path(fpath).parts
                if len(parts) >= 2:
                    return str(Path(args.frames_dir).resolve() / parts[-2] / parts[-1])
        return fpath

    # 1. OCR Recovery
    if not args.skip_ocr:
        log.info("=" * 60)
        log.info("STEP 1: RECOVERING MISSING OCR TEXT (EasyOCR only)")
        log.info("=" * 60)
        
        from models.ocr.easyocr_model import EasyOCRModel
        ocr_model = EasyOCRModel(settings.ocr)
        
        async with meta_db.session() as session:
            result = await session.execute(
                text("SELECT frame_id, frame_path FROM frames WHERE ocr_text IS NULL")
            )
            missing_frames = result.fetchall()
            
        log.info(f"Found {len(missing_frames)} frames missing OCR text.")
        
        if missing_frames:
            for frame_id, frame_path in tqdm(missing_frames, desc="Extracting OCR", unit="frame"):
                actual_path = resolve_frame_path(frame_path)
                if not os.path.exists(actual_path):
                    log.error(f"Image not found: {actual_path}. Skipping...")
                    continue
                    
                try:
                    results = ocr_model.extract(actual_path)
                    ocr_text = ocr_model.results_to_text(results)
                    ocr_json = [{"text": r.text, "confidence": r.confidence, "bbox": r.bbox} for r in results]
                    
                    await meta_db.upsert_frame({
                        "frame_id": frame_id,
                        "ocr_results": ocr_json,
                        "ocr_text": ocr_text,
                    })
                except Exception as e:
                    log.error(f"Failed OCR on {frame_id}: {e}")
            log.info("OCR recovery complete!")
        else:
            log.info("No frames missing OCR. Skipping.")

    # 2. YOLO Recovery
    if not args.skip_yolo:
        log.info("=" * 60)
        log.info("STEP 2: RECOVERING YOLO OBJECT DETECTION (Video-by-Video Check)")
        log.info("=" * 60)
        
        all_frames = await meta_db.get_all_frames_async()
        if not all_frames:
            log.error("No frames found in database to run YOLO!")
        else:
            frames_by_video = {}
            for f in all_frames:
                f.frame_path = resolve_frame_path(f.frame_path)
                frames_by_video.setdefault(f.video_id, []).append(f)
                
            missing_frames = []
            async with meta_db.session() as session:
                for video_id, v_frames in frames_by_video.items():
                    v_frames.sort(key=lambda x: x.frame_idx)
                    
                    query = text("""
                        SELECT MAX(f.frame_idx) 
                        FROM detections d 
                        JOIN frames f ON d.frame_id = f.frame_id 
                        WHERE f.video_id = :vid
                    """)
                    result = await session.execute(query, {"vid": video_id})
                    max_idx = result.scalar()
                    
                    if max_idx is None:
                        missing_frames.extend(v_frames)
                    else:
                        v_missing = [f for f in v_frames if f.frame_idx > max_idx]
                        missing_frames.extend(v_missing)
                        
            log.info(f"YOLO has already processed {len(all_frames) - len(missing_frames)} frames.")
            log.info(f"Resuming YOLO for the remaining {len(missing_frames)} frames across {len(frames_by_video)} videos...")

            if missing_frames:
                detection_svc = DetectionService(settings.detection, meta_db)
                await detection_svc.process_frames(missing_frames)
            else:
                log.info("YOLO object detection is already 100% complete!")

    # 3. BM25 Rebuild
    if not args.skip_bm25:
        log.info("=" * 60)
        log.info("STEP 3: REBUILDING BM25 OCR INDEX")
        log.info("=" * 60)
        try:
            from retrieval.branches.ocr_branches import BM25OcrIndex
            bm25 = BM25OcrIndex(settings.ocr)
            await bm25.build_from_db(meta_db)
            log.info("BM25 rebuilt successfully!")
        except Exception as e:
            log.error(f"Failed to rebuild BM25: {e}")

    log.info("==================================================")
    log.info("🎉 ALL TASKS HAVE COMPLETED!")
    log.info("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
