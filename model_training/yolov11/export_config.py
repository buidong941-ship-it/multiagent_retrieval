"""Export config sau khi train YOLOv11."""
import argparse
from pathlib import Path


def export(checkpoint_path: str) -> None:
    ckpt = Path(checkpoint_path).resolve()
    print("\n" + "=" * 65)
    print("  Copy đoạn config bên dưới vào:  video_retrieval/.env")
    print("=" * 65)
    print()
    print(f"# ── YOLOv11 Fine-tuned Model ──────────────────────────────")
    print(f"DETECT_MODEL_PATH={ckpt}")
    print(f"DETECT_DEVICE=cuda")
    print(f"DETECT_CONFIDENCE_THRESHOLD=0.3")
    print(f"DETECT_BATCH_SIZE=16")
    print()
    print("=" * 65)
    print()
    print("Lưu ý: Re-index detection data sau khi đổi model!")
    print("  python scripts/index_videos.py --no-extraction --no-embedding --no-ocr")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pt weights file")
    args = parser.parse_args()
    export(args.checkpoint)
