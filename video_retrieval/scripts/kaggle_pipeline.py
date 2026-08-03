"""
Kaggle Processing Pipeline Script (FAISS Support).

Script dành riêng để chạy trên Kaggle (hoặc Google Colab).
Nó sẽ chạy toàn bộ pipeline trích xuất (frame, clip embedding, ocr, object),
lưu dữ liệu vào FAISS Vector Database, và cuối cùng TỰ ĐỘNG nén toàn bộ
thư mục data/indexes thành 1 file .zip duy nhất để bạn có thể tải về máy Local.

Cách dùng:
1. Đảm bảo đã cài đặt faiss: !pip install faiss-cpu
2. Đặt video vào thư mục data/videos/ (hoặc thay đổi đường dẫn video_dir bên dưới)
3. Chạy script này trên Kaggle: !python scripts/kaggle_pipeline.py
4. Tải file "kaggle_indexes.zip" về máy Local và giải nén vào thư mục data/indexes/.
"""

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

# Add project root directory to sys.path for Kaggle environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.frame_config import ExtractionMode
from config.settings import get_settings
from pipelines.offline_pipeline import OfflinePipeline


async def main():
    settings = get_settings()
    # Cấu hình sử dụng TransNetV2 (10 Keyframes / shot) & Bật lọc trùng lặp bằng BEiT-3 Cosine Similarity (>0.90)
    settings.frame.mode = ExtractionMode.TRANSNETV2
    settings.frame.transnet_threshold = 0.5
    settings.frame.keyframes_per_shot = 10
    settings.embedding.enable_dedup = True
    settings.embedding.dedup_threshold = 0.90

    # Cho phép truyền --video_dir từ CLI / Env Vars, hoặc tự động dò tìm trong /kaggle/input/
    video_dir = os.getenv("VIDEO_DIR", "data/videos")
    for i, arg in enumerate(sys.argv):
        if arg == "--video_dir" and i + 1 < len(sys.argv):
            video_dir = sys.argv[i + 1]
            break

    # Nếu video_dir mặc định không tồn tại hoặc không có video, tự động quét tìm trong /kaggle/input/
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    has_videos = Path(video_dir).exists() and any(f.suffix.lower() in video_extensions for f in Path(video_dir).rglob("*"))
    
    if not has_videos:
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            print("Đang quét tìm thư mục chứa video trong /kaggle/input/...")
            for vid_file in kaggle_input.rglob("*"):
                if vid_file.suffix.lower() in video_extensions:
                    video_dir = str(vid_file.parent)
                    print(f"✅ Tự động phát hiện thư mục video: {video_dir}")
                    break

    pipeline = OfflinePipeline(settings)
    
    print("==================================================")
    print(f"1. KHỞI TẠO DATABASES (FAISS & SQLite)... | video_dir={video_dir}")
    print("==================================================")
    await pipeline.initialize()
    
    print("\n==================================================")
    print("2. CHẠY PIPELINE TRÍCH XUẤT (TransNetV2 Keyframe + SigLIP2 + OCR + YOLO)...")
    print("==================================================")
    # Bạn có thể bật/tắt các bước trích xuất tùy nhu cầu ở đây
    await pipeline.run(
        video_dir=video_dir,
        run_extraction=True,
        run_embedding=True,
        run_ocr=True,
        run_detection=True
    )
    
    # Đóng kết nối FAISS Database để nó flush toàn bộ chỉ mục xuống ổ cứng
    if hasattr(pipeline.vector_db, "close"):
        pipeline.vector_db.close()
        
    print("\nĐang chờ FAISS Database lưu dữ liệu (flush) xuống ổ cứng...")
    time.sleep(5)
    
    print("\n==================================================")
    print("3. NÉN THƯ MỤC THÀNH FILE ZIP ĐỂ TẢI VỀ...")
    print("==================================================")
    index_dir = "data/indexes"
    output_zip = "kaggle_indexes" # shutil.make_archive sẽ tự động thêm đuôi .zip
    
    if Path(index_dir).exists():
        print(f"Đang nén toàn bộ {index_dir} vào file {output_zip}.zip ...")
        shutil.make_archive(output_zip, 'zip', index_dir)
        print("HOÀN TẤT nén DB!")
    else:
        print(f"Lỗi: Không tìm thấy thư mục {index_dir} để nén.")

    frames_dir = "data/frames"
    frames_zip = "kaggle_frames"
    if Path(frames_dir).exists():
        print(f"\nĐang nén toàn bộ {frames_dir} vào file {frames_zip}.zip (file này có thể rất nặng)...")
        shutil.make_archive(frames_zip, 'zip', frames_dir)
        print("HOÀN TẤT nén Frames!")
        
    print("\n✅ XONG! Bạn hãy tải 2 file kaggle_indexes.zip và kaggle_frames.zip về máy Local và giải nén đè vào mục data/")

if __name__ == "__main__":
    asyncio.run(main())
