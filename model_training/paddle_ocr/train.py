"""
PaddleOCR Fine-tuning Script.

PaddleOCR dùng PaddlePaddle trainer riêng biệt.
Script này là wrapper đơn giản.

Tài liệu đầy đủ:
    https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/model_train/recognition.html

Cách dùng:
    pip install paddlepaddle paddleocr
    python train.py --config config.yaml

Sau khi train:
    python export_config.py --det_model ./outputs/det/best_model \
                            --rec_model ./outputs/rec/best_model
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

THIS_DIR = Path(__file__).parent
PADDLEOCR_REPO = THIS_DIR / "PaddleOCR"  # Clone repo nếu cần


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_paddle():
    """Kiểm tra PaddlePaddle đã được cài chưa."""
    try:
        import paddle
        print(f"[PaddlePaddle] Version: {paddle.__version__}")
        return True
    except ImportError:
        print("ERROR: Cài PaddlePaddle trước!")
        print("  pip install paddlepaddle-gpu  # nếu có GPU")
        print("  pip install paddlepaddle       # CPU only")
        return False


def train_recognition(config: dict) -> None:
    """Fine-tune SVTR recognition model."""
    data_cfg  = config["data"]
    train_cfg = config["training"]

    if not check_paddle():
        return

    # PaddleOCR training dùng config YAML riêng của PaddlePaddle
    # Xem: PaddleOCR/configs/rec/PP-OCRv4/vi_PP-OCRv4_rec.yml
    print("\n[PaddleOCR] Recognition fine-tuning")
    print("Tài liệu: https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/model_train/recognition.html")
    print()
    print("Các bước cần làm:")
    print("1. Clone PaddleOCR repo:")
    print("   git clone https://github.com/PaddlePaddle/PaddleOCR.git")
    print()
    print("2. Download pretrained weights:")
    print("   Xem: https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/model_list.html")
    print()
    print("3. Chuẩn bị dữ liệu theo format:")
    print(f"   train_list: {data_cfg.get('rec_train_list', './data/rec/train_list.txt')}")
    print(f"   val_list:   {data_cfg.get('rec_val_list',   './data/rec/val_list.txt')}")
    print()
    print("4. Chạy training:")
    print("   cd PaddleOCR")
    print("   python tools/train.py \\")
    print("     -c configs/rec/PP-OCRv4/vi_PP-OCRv4_rec.yml \\")
    print("     -o Global.pretrained_model=./pretrain_models/vi_PP-OCRv4_rec_train \\")
    print(f"       Train.dataset.data_dir=./data \\")
    print(f"       Train.dataset.label_file_list=[{data_cfg.get('rec_train_list', '')}] \\")
    print(f"       Eval.dataset.label_file_list=[{data_cfg.get('rec_val_list', '')}] \\")
    print(f"       Global.epoch_num={train_cfg.get('num_epochs', 100)} \\")
    print(f"       Global.save_model_dir={train_cfg.get('output_dir', './outputs')}/rec")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune PaddleOCR")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    os.chdir(THIS_DIR)
    cfg = load_config(args.config)
    stage = cfg["model"].get("stage", "rec")

    if stage in ("rec", "both"):
        train_recognition(cfg)
    if stage in ("det", "both"):
        print("\n[PaddleOCR] Detection fine-tuning — xem tài liệu:")
        print("  https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/model_train/detection.html")
