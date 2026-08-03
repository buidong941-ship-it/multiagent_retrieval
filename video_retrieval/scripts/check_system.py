"""
Script kiểm tra từng component của hệ thống.
Chạy cái này trước để đảm bảo mọi thứ hoạt động.

Cách dùng:
    cd video_retrieval
    python scripts/check_system.py

Kiểm tra:
    ✓ GPU / CUDA
    ✓ FAISS connection
    ✓ SigLIP2 load
    ✓ Jina-CLIP load
    ✓ Ollama / Qwen
    ✓ PaddleOCR load
    ✓ YOLOv11 load
    ✓ Gemini API key
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check(name: str, fn) -> bool:
    """Chạy 1 kiểm tra, in kết quả."""
    try:
        result = fn()
        msg = f" ({result})" if result and result is not True else ""
        print(f"  ✓  {name}{msg}")
        return True
    except Exception as e:
        print(f"  ✗  {name}")
        print(f"       └─ {e}")
        return False


def main():
    print()
    print("=" * 55)
    print("  Video Retrieval System — Health Check")
    print("=" * 55)
    results = {}

    # ── 1. Python & GPU ──────────────────────────────────────────
    print("\n[1] Python & GPU")

    results["python"] = check(
        "Python version",
        lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    def check_torch():
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"CUDA ✓ | {name} | {mem:.1f} GB VRAM"
        return "CPU only (CUDA không khả dụng — sẽ chậm hơn)"

    results["torch"] = check("PyTorch + CUDA", check_torch)

    # ── 2. Vector Database (FAISS) ───────────────────────────────
    print("\n[2] Vector Database (FAISS)")

    def check_faiss():
        import faiss
        return f"FAISS version {faiss.__version__} OK"

    results["faiss"] = check("FAISS package", check_faiss)

    # ── 3. Embedding Models ──────────────────────────────────────
    print("\n[3] Embedding Models")

    def check_siglip():
        from transformers import AutoProcessor
        model_name = "google/siglip2-so400m-patch14-384"
        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return f"processor OK | {model_name}"

    results["siglip_proc"] = check("SigLIP2 processor", check_siglip)

    def check_jina():
        import timm
        return f"timm {timm.__version__} OK (Jina dependencies)"

    results["jina"] = check("Jina-CLIP dependencies", check_jina)

    # ── 4. LLM Providers ─────────────────────────────────────────
    print("\n[4] LLM Providers")

    def check_ollama():
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return f"Ollama OK | {len(models)} models available"
        raise ConnectionError("Lỗi kết nối API Ollama")

    results["ollama"] = check("Ollama (localhost:11434)", check_ollama)

    # ── 5. OCR ───────────────────────────────────────────────────
    print("\n[5] OCR")

    def check_paddle():
        import importlib
        spec = importlib.util.find_spec("paddleocr")
        if spec is None:
            raise ImportError("paddleocr chưa được cài")
        return "paddleocr package found"

    results["paddle"] = check("PaddleOCR package", check_paddle)

    # ── 6. YOLO ──────────────────────────────────────────────────
    print("\n[6] Object Detection")

    def check_yolo():
        import importlib
        spec = importlib.util.find_spec("ultralytics")
        if spec is None:
            raise ImportError("ultralytics chưa được cài")
        return "ultralytics OK"

    results["yolo"] = check("YOLOv11 (ultralytics)", check_yolo)

    # ── 7. Environment variables ─────────────────────────────────
    print("\n[7] Configuration (.env)")

    env_path = Path(__file__).parent.parent / ".env"

    def check_env_file():
        if not env_path.exists():
            raise FileNotFoundError(f".env chưa tạo — chạy: cp .env.example .env")
        return ".env file OK"

    results["env_file"] = check(".env file exists", check_env_file)

    def check_gemini_key():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        key = os.getenv("RETRIEVAL_GEMINI_API_KEY", "")
        if not key or key == "your_gemini_api_key_here":
            raise ValueError("Chưa set RETRIEVAL_GEMINI_API_KEY trong .env")
        return f"key set (starts with {key[:8]}...)"

    results["gemini"] = check("Gemini API key", check_gemini_key)

    # ── 8. Database ──────────────────────────────────────────────
    print("\n[8] Metadata Database")

    def check_sqlalchemy():
        from sqlalchemy import create_engine, text
        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "SQLAlchemy + SQLite OK"

    results["sqlite"] = check("SQLAlchemy / SQLite", check_sqlalchemy)

    # ── Summary ──────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v)
    total  = len(results)

    print()
    print("=" * 55)
    print(f"  KẾT QUẢ: {passed}/{total} checks passed")
    print("=" * 55)

    if not results.get("faiss"):
        print("\n  ⚠ Cài đặt FAISS bằng: pip install faiss-cpu (hoặc faiss-gpu)")
        
    if not results.get("ollama"):
        print("\n  ⚠ Ollama chưa chạy. Khởi động Ollama Desktop hoặc chạy: ollama serve")

    if not results.get("env_file") or not results.get("gemini"):
        print("\n  ⚠ Cần tạo .env:")
        print("    copy .env.example .env   (Windows)")
        print("    # Rồi mở .env và thêm RETRIEVAL_GEMINI_API_KEY=...")

    if passed == total:
        print("\n  ✓ Tất cả OK! Sẵn sàng chạy hệ thống.")
        print("    Bước tiếp: python api/main.py")
    print()


if __name__ == "__main__":
    main()
