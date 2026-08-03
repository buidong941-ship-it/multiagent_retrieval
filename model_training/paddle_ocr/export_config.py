"""Export config sau khi train PaddleOCR."""
import argparse
from pathlib import Path


def export(det_model: str = None, rec_model: str = None) -> None:
    print("\n" + "=" * 65)
    print("  Copy đoạn config bên dưới vào:  video_retrieval/.env")
    print("=" * 65)
    print()
    print(f"# ── PaddleOCR Fine-tuned Model ────────────────────────────")
    print(f"OCR_LANG=vi")
    if det_model:
        det = Path(det_model).resolve()
        print(f"OCR_DET_MODEL_DIR={det}")
    if rec_model:
        rec = Path(rec_model).resolve()
        print(f"OCR_REC_MODEL_DIR={rec}")
    print(f"OCR_USE_GPU=true")
    print(f"OCR_MIN_CONFIDENCE=0.5")
    print()
    print("=" * 65)
    print()
    print("Lưu ý: Re-index OCR data sau khi đổi model!")
    print("  python scripts/index_videos.py --no-extraction --no-embedding --no-detection")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model", default=None)
    parser.add_argument("--rec_model", default=None)
    args = parser.parse_args()
    export(args.det_model, args.rec_model)
