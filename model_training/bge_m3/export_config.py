"""
Export config sau khi train BGE-M3 xong.

Cách dùng:
    python export_config.py --checkpoint ./outputs/best
"""

from __future__ import annotations
import argparse
from pathlib import Path


def export(checkpoint_path: str) -> None:
    ckpt = Path(checkpoint_path).resolve()

    print("\n" + "=" * 65)
    print("  Copy đoạn config bên dưới vào:  video_retrieval/.env")
    print("=" * 65)
    print()
    print(f"# ── BGE-M3 Fine-tuned Model ───────────────────────────────")
    print(f"OCR_BGE_MODEL_NAME={ckpt}")
    print(f"OCR_BGE_DEVICE=cuda")
    print(f"OCR_BGE_BATCH_SIZE=32")
    print(f"OCR_BGE_EMBEDDING_DIM=1024")
    print(f"OCR_BGE_NORMALIZE=true")
    print(f"OCR_BGE_MAX_LENGTH=512")
    print()
    print("=" * 65)
    print()
    print("Lưu ý: Re-index OCR embeddings sau khi đổi model!")
    print("  python scripts/index_videos.py --no-extraction --no-embedding --no-detection")
    print("  (chỉ chạy lại bước OCR)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    export(args.checkpoint)
