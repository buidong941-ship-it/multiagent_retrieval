"""
YOLOv11 Fine-tuning Script.

YOLOv11 dùng Ultralytics trainer — rất đơn giản, chỉ cần:
    1. Chuẩn bị dataset theo YOLO format
    2. Tạo dataset.yaml
    3. Chạy script này

Cách dùng:
    python train.py                          # config.yaml mặc định
    python train.py --config my_config.yaml

Sau khi train:
    python export_config.py --checkpoint ./outputs/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

THIS_DIR = Path(__file__).parent


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    """Fine-tune YOLOv11 dùng Ultralytics API."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: cài ultralytics trước: pip install ultralytics")
        return

    model_cfg  = config["model"]
    train_cfg  = config["training"]
    data_cfg   = config["data"]
    output_dir = Path(train_cfg["output_dir"])

    print(f"[YOLOv11] Loading model: {model_cfg['name']}")
    model = YOLO(model_cfg["name"])

    print(f"[YOLOv11] Starting training...")
    print(f"  Dataset: {data_cfg['yaml_path']}")
    print(f"  Epochs:  {train_cfg['num_epochs']}")
    print(f"  Batch:   {train_cfg['batch_size']}")
    print(f"  Device:  {train_cfg['device']}")

    results = model.train(
        data=data_cfg["yaml_path"],
        epochs=train_cfg["num_epochs"],
        batch=train_cfg["batch_size"],
        imgsz=train_cfg["imgsz"],
        device=train_cfg["device"],
        workers=train_cfg.get("workers", 8),
        project=str(output_dir),
        name="train",
        exist_ok=True,

        # Learning rate
        lr0=train_cfg.get("lr0", 0.01),
        lrf=train_cfg.get("lrf", 0.01),

        # Augmentation
        augment=train_cfg.get("augment", True),
        mosaic=train_cfg.get("mosaic", 1.0),
        fliplr=train_cfg.get("fliplr", 0.5),

        # Early stopping
        patience=train_cfg.get("patience", 10),
        save_period=train_cfg.get("save_period", 10),

        # Verbose
        verbose=True,
    )

    best_weights = output_dir / "train" / "weights" / "best.pt"
    print(f"\n[Done] Training complete.")
    print(f"       Best weights: {best_weights}")
    print(f"       Run: python export_config.py --checkpoint {best_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    os.chdir(THIS_DIR)
    cfg = load_config(args.config)
    train(cfg)
