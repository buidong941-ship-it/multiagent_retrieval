"""
SigLIP2 Fine-tuning Script với LoRA.

Cách dùng:
    python train.py                          # dùng config.yaml mặc định
    python train.py --config my_config.yaml  # dùng config khác
    python train.py --resume ./outputs/checkpoint-500  # tiếp tục train

Sau khi train xong:
    python export_config.py --checkpoint ./outputs/best
    # → Nhận đoạn config để paste vào video_retrieval/.env
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm

# ── Resolve paths ─────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).parent


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_lora(model, config: dict):
    """
    Áp dụng LoRA lên text encoder của SigLIP2.

    Chỉ cần cài peft: pip install peft
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("ERROR: cài peft trước: pip install peft")
        sys.exit(1)

    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )

    # Chỉ áp dụng LoRA cho text_model
    model.text_model = get_peft_model(model.text_model, lora_config)
    model.text_model.print_trainable_parameters()

    return model


def freeze_vision_encoder(model):
    """Đóng băng toàn bộ vision encoder — không train."""
    for param in model.vision_model.parameters():
        param.requires_grad = False
    print("[Setup] Vision encoder frozen.")


def get_embeddings(model, batch: dict, device: str):
    """
    Lấy image và text embeddings đã L2-normalize từ 1 batch.

    Returns:
        (image_embeddings, text_embeddings) — cả 2 shape (N, D).
    """
    pixel_values  = batch["pixel_values"].to(device)
    input_ids     = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Forward pass
    image_emb = model.get_image_features(pixel_values=pixel_values)
    text_emb  = model.get_text_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # L2-normalize
    image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
    text_emb  = torch.nn.functional.normalize(text_emb, dim=-1)

    return image_emb, text_emb


def build_optimizer(model, config: dict) -> AdamW:
    """AdamW với weight decay chỉ trên non-bias parameters."""
    train_cfg = config["training"]
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": train_cfg["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(params, lr=train_cfg["learning_rate"])


def build_scheduler(optimizer, config: dict, total_steps: int):
    """Cosine scheduler với warmup."""
    train_cfg = config["training"]
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                               eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_steps])


def train(config: dict, resume_from: str | None = None) -> None:
    """Main training loop."""
    from dataset import SigLIP2Dataset, create_dataloaders
    from loss import build_loss

    device = config["hardware"]["device"]
    train_cfg = config["training"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────
    model_name = config["model"]["name"]
    print(f"[Model] Loading: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config["model"]["use_fp16"] else torch.float32,
    ).to(device)

    # ── Freeze & LoRA ───────────────────────────────────────────────
    if config["model"]["freeze_vision_encoder"]:
        freeze_vision_encoder(model)

    if config["lora"]["enabled"]:
        model = setup_lora(model, config)

    # ── Dataset & DataLoader ────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(config, processor)

    # ── Loss ────────────────────────────────────────────────────────
    loss_fn = build_loss(config).to(device)

    # ── Optimizer & Scheduler ───────────────────────────────────────
    accum = train_cfg.get("gradient_accumulation_steps", 1)
    total_steps = math.ceil(len(train_loader) / accum) * train_cfg["num_epochs"]
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps)

    # ── Resume from checkpoint ──────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    patience_count = 0

    if resume_from:
        ckpt = torch.load(Path(resume_from) / "trainer_state.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[Resume] Continuing from epoch {start_epoch}")

    # ── Training Loop ───────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=config["model"]["use_fp16"])

    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']}")
        for step, batch in enumerate(pbar):
            with torch.cuda.amp.autocast(enabled=config["model"]["use_fp16"]):
                img_emb, txt_emb = get_embeddings(model, batch, device)
                loss = loss_fn(img_emb, txt_emb) / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accum
            pbar.set_postfix({"loss": f"{loss.item() * accum:.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # ── Validation ───────────────────────────────────────────────
        val_loss = avg_train_loss  # fallback nếu không có val set
        if val_loader:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    with torch.cuda.amp.autocast(enabled=config["model"]["use_fp16"]):
                        img_emb, txt_emb = get_embeddings(model, batch, device)
                        v_loss = loss_fn(img_emb, txt_emb)
                    val_losses.append(v_loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}")

        # ── Save checkpoint ─────────────────────────────────────────
        ckpt_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
        _save_checkpoint(model, processor, optimizer, epoch, val_loss,
                         best_val_loss, ckpt_dir)

        # ── Best model ───────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_dir = output_dir / "best"
            _save_checkpoint(model, processor, optimizer, epoch, val_loss,
                             best_val_loss, best_dir)
            print(f"[Best] Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            patience = train_cfg.get("early_stopping_patience", 99)
            if patience_count >= patience:
                print(f"[Early Stop] No improvement for {patience} epochs. Stopping.")
                break

    print(f"\n[Done] Training complete. Best checkpoint: {output_dir / 'best'}")
    print(f"        Run: python export_config.py --checkpoint {output_dir / 'best'}")


def _save_checkpoint(model, processor, optimizer, epoch, val_loss,
                     best_val_loss, save_dir: Path) -> None:
    """Lưu model weights + processor + trainer state."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Lưu LoRA adapter (hoặc toàn bộ model nếu không dùng LoRA)
    if hasattr(model, "text_model") and hasattr(model.text_model, "save_pretrained"):
        model.text_model.save_pretrained(str(save_dir / "text_lora"))
    model.save_pretrained(str(save_dir))
    processor.save_pretrained(str(save_dir))

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }, save_dir / "trainer_state.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SigLIP2")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    os.chdir(THIS_DIR)  # Run from script's directory
    cfg = load_config(args.config)
    train(cfg, resume_from=args.resume)
