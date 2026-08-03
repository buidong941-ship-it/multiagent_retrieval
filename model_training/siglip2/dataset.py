"""
SigLIP2 Fine-tuning Dataset.

Format dữ liệu đầu vào (CSV):
    image_path,caption
    /data/frames/vid001_frame_000001.jpg,"Một người phụ nữ mặc áo đỏ..."
    /data/frames/vid001_frame_000045.jpg,"Xe máy chạy trên đường phố..."

Cách tạo dữ liệu:
    Option 1: Dùng LLM sinh caption tự động từ frame
    Option 2: Dùng caption query từ ground truth cuộc thi
    Option 3: Dùng cả hai (merge CSV)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


class SigLIP2Dataset(Dataset):
    """
    Dataset trả về cặp (image, caption) cho SigLIP2 fine-tuning.

    Args:
        csv_path:       Đường dẫn tới file CSV (image_path, caption).
        processor:      AutoProcessor của SigLIP2 (xử lý ảnh + text).
        augment:        Bật data augmentation cho ảnh.
        max_text_len:   Số token tối đa cho caption.
    """

    def __init__(
        self,
        csv_path: str,
        processor: AutoProcessor,
        augment: bool = False,
        max_text_len: int = 64,
    ) -> None:
        self.processor = processor
        self.augment = augment
        self.max_text_len = max_text_len
        self.samples: list[tuple[str, str]] = []

        self._load_csv(csv_path)

        if self.augment:
            self._build_augmentor()

    def _load_csv(self, csv_path: str) -> None:
        """Đọc file CSV, bỏ qua dòng có ảnh không tồn tại."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row.get("image_path", "").strip()
                caption = row.get("caption", "").strip()
                if img_path and caption and Path(img_path).exists():
                    self.samples.append((img_path, caption))

        print(f"[Dataset] Loaded {len(self.samples)} valid samples from {csv_path}")

    def _build_augmentor(self):
        """Xây dựng torchvision augmentation pipeline."""
        from torchvision import transforms
        self.augmentor = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomGrayscale(p=0.05),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Trả về dict với pixel_values và input_ids đã được xử lý.
        """
        image_path, caption = self.samples[idx]

        # Load ảnh
        image = Image.open(image_path).convert("RGB")

        # Augment (chỉ khi train)
        if self.augment and hasattr(self, "augmentor"):
            image = self.augmentor(image)

        # Xử lý bằng SigLIP2 processor
        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),  # (C, H, W)
            "input_ids":    encoding["input_ids"].squeeze(0),      # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def create_dataloaders(config: dict, processor: AutoProcessor):
    """
    Tạo train và val DataLoader từ config.

    Args:
        config:    Dict từ config.yaml.
        processor: SigLIP2 AutoProcessor.

    Returns:
        Tuple (train_loader, val_loader). val_loader có thể là None.
    """
    from torch.utils.data import DataLoader

    data_cfg = config["data"]
    train_cfg = config["training"]
    hw_cfg = config["hardware"]

    train_ds = SigLIP2Dataset(
        csv_path=data_cfg["train_csv"],
        processor=processor,
        augment=data_cfg.get("augment", True),
        max_text_len=64,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=hw_cfg.get("num_workers", 4),
        pin_memory=hw_cfg.get("pin_memory", True),
        drop_last=True,   # SigLIP loss cần batch đồng đều
    )

    val_loader = None
    val_csv = data_cfg.get("val_csv", "")
    if val_csv and Path(val_csv).exists():
        val_ds = SigLIP2Dataset(
            csv_path=val_csv,
            processor=processor,
            augment=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=hw_cfg.get("num_workers", 4),
        )

    return train_loader, val_loader
