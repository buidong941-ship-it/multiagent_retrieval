"""
BGE-M3 Fine-tuning Dataset.

Format đầu vào (JSONL — mỗi dòng là 1 JSON):
    {"query": "Highlands Coffee ở đâu",
     "pos": ["Highlands Coffee Việt Nam tại số 2 Phạm Ngọc Thạch"],
     "neg": ["The Coffee House", "Trung Nguyên Legend"]}

Cách tạo dữ liệu:
    - query: câu truy vấn của người dùng trong cuộc thi
    - pos:   OCR text của frame đúng (ground truth)
    - neg:   OCR text của frame sai (hard negatives)

Hard Negatives tốt hơn random:
    - Dùng BM25 để tìm frame "gần đúng nhưng sai" → làm hard negative
    - Ví dụ: query "Highlands Coffee" nhưng lấy frame có "Starbucks" làm neg
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BGEM3Dataset(Dataset):
    """
    Dataset cho BGE-M3 fine-tuning với triplets (query, positive, negatives).

    Mỗi sample gồm:
        - query:     câu truy vấn tiếng Việt
        - positive:  text liên quan (OCR text của frame đúng)
        - negatives: list text không liên quan

    Args:
        jsonl_path:         Đường dẫn JSONL.
        tokenizer:          BGE-M3 tokenizer.
        max_length:         Số token tối đa.
        num_hard_negatives: Số hard negatives lấy mỗi sample.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        num_hard_negatives: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_hard_negatives = num_hard_negatives
        self.samples: list[dict] = []
        self._load(jsonl_path)

    def _load(self, path: str) -> None:
        """Đọc JSONL, validate format."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(p, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    if "query" in sample and "pos" in sample:
                        self.samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Dòng {line_no} lỗi JSON: {e}")

        print(f"[Dataset] Loaded {len(self.samples)} samples from {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize(self, text: str) -> dict:
        """Tokenize một đoạn text."""
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> dict:
        """
        Trả về dict với query, positive, và negatives đã tokenize.
        """
        sample = self.samples[idx]
        query = sample["query"]
        positive = sample["pos"][0] if isinstance(sample["pos"], list) else sample["pos"]
        negatives = sample.get("neg", [])[:self.num_hard_negatives]

        q_enc = self._tokenize(query)
        p_enc = self._tokenize(positive)

        result = {
            "query_input_ids":      q_enc["input_ids"].squeeze(0),
            "query_attention_mask": q_enc["attention_mask"].squeeze(0),
            "pos_input_ids":        p_enc["input_ids"].squeeze(0),
            "pos_attention_mask":   p_enc["attention_mask"].squeeze(0),
        }

        # Thêm hard negatives nếu có
        for i, neg in enumerate(negatives):
            n_enc = self._tokenize(neg)
            result[f"neg{i}_input_ids"]      = n_enc["input_ids"].squeeze(0)
            result[f"neg{i}_attention_mask"] = n_enc["attention_mask"].squeeze(0)

        return result


def create_dataloaders(config: dict, tokenizer: AutoTokenizer):
    """Tạo train và val DataLoader từ config."""
    from torch.utils.data import DataLoader

    data_cfg  = config["data"]
    train_cfg = config["training"]
    hw_cfg    = config["hardware"]

    train_ds = BGEM3Dataset(
        jsonl_path=data_cfg["train_jsonl"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        num_hard_negatives=data_cfg.get("num_hard_negatives", 1),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=hw_cfg.get("num_workers", 4),
        pin_memory=hw_cfg.get("pin_memory", True),
    )

    val_loader = None
    val_jsonl = data_cfg.get("val_jsonl", "")
    if val_jsonl and Path(val_jsonl).exists():
        val_ds = BGEM3Dataset(
            jsonl_path=val_jsonl,
            tokenizer=tokenizer,
            max_length=config["model"]["max_length"],
            num_hard_negatives=data_cfg.get("num_hard_negatives", 1),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=hw_cfg.get("num_workers", 4),
        )

    return train_loader, val_loader
