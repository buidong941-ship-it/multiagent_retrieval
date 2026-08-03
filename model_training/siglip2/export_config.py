"""
Export config sau khi train SigLIP2 xong.

Cách dùng:
    python export_config.py --checkpoint ./outputs/best

Output:
    In ra đoạn config để paste vào video_retrieval/.env
    Ví dụ:
        # ── Paste vào video_retrieval/.env ──
        EMBED_BACKEND=siglip2
        EMBED_SIGLIP_MODEL_NAME=C:/Users/.../model_training/siglip2/outputs/best
        EMBED_DEVICE=cuda
        EMBED_BATCH_SIZE=64
        EMBED_USE_FP16=true
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export(checkpoint_path: str) -> None:
    ckpt = Path(checkpoint_path).resolve()

    if not ckpt.exists():
        print(f"ERROR: Checkpoint không tồn tại: {ckpt}")
        return

    # Kiểm tra xem checkpoint có hợp lệ không
    config_json = ckpt / "config.json"
    if not config_json.exists():
        print(f"WARNING: Không tìm thấy config.json trong {ckpt}")
        print("         Checkpoint có thể chưa được lưu đúng cách.")

    print("\n" + "=" * 65)
    print("  Copy đoạn config bên dưới vào:  video_retrieval/.env")
    print("=" * 65)
    print()
    print(f"# ── SigLIP2 Fine-tuned Model ──────────────────────────────")
    print(f"EMBED_BACKEND=siglip2")
    print(f"EMBED_SIGLIP_MODEL_NAME={ckpt}")
    print(f"EMBED_DEVICE=cuda")
    print(f"EMBED_BATCH_SIZE=64")
    print(f"EMBED_USE_FP16=true")
    print(f"EMBED_EMBEDDING_DIM=1152")
    print(f"EMBED_NORMALIZE=true")
    print()
    print("=" * 65)
    print()
    print("Sau khi paste, khởi động lại API:")
    print("  uvicorn api.main:app --reload")
    print()
    print("Lưu ý: Re-index toàn bộ video sau khi đổi model embedding!")
    print("  python scripts/index_videos.py --video_dir /path/to/videos")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SigLIP2 config")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Đường dẫn tới checkpoint directory (ví dụ: ./outputs/best)",
    )
    args = parser.parse_args()
    export(args.checkpoint)
