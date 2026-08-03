"""
BGE-M3 Fine-tuning Script.

Dùng MultipleNegativesRankingLoss (MNRL):
    - Với mỗi (query, positive), mọi positive khác trong batch
      đều là negatives → không cần label thêm
    - Đây là loss tốt nhất cho retrieval tasks
    - Paper: "Efficient Natural Language Response Suggestion..."

Cách dùng:
    python train.py                          # config.yaml mặc định
    python train.py --config my_config.yaml
    python train.py --resume ./outputs/checkpoint-epoch-3
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

THIS_DIR = Path(__file__).parent


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cls_embedding(model, input_ids, attention_mask, normalize: bool = True):
    """
    Lấy CLS token embedding từ model.

    BGE-M3 dùng CLS token (vị trí 0) làm dense embedding.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    emb = outputs.last_hidden_state[:, 0, :]  # CLS token
    if normalize:
        emb = F.normalize(emb, dim=-1)
    return emb


def mnrl_loss(
    query_emb: torch.Tensor,        # (B, D)
    pos_emb: torch.Tensor,          # (B, D)
    neg_embs: list[torch.Tensor],   # list of (B, D)
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Multiple Negatives Ranking Loss (InfoNCE với in-batch negatives).

    Với mỗi query_i:
        - positive  = pos_emb[i]
        - negatives = tất cả pos_emb[j] (j ≠ i) + các hard neg

    Args:
        query_emb:   Query embeddings, shape (B, D), L2-normalized.
        pos_emb:     Positive embeddings, shape (B, D), L2-normalized.
        neg_embs:    List of hard negative embeddings, each (B, D).
        temperature: Scaling factor (nhỏ hơn → phân biệt hơn).

    Returns:
        Scalar loss.
    """
    # Gộp positives + hard negatives làm target matrix
    all_docs = [pos_emb] + neg_embs                        # list of (B, D)
    doc_matrix = torch.cat(all_docs, dim=0)                # (B*(1+num_neg), D)

    # Similarity: query × doc_matrix.T → (B, B*(1+num_neg))
    scores = torch.matmul(query_emb, doc_matrix.T) / temperature

    # Label: query_i nên match với pos_i → index i trong doc_matrix
    labels = torch.arange(query_emb.shape[0], device=query_emb.device)
    loss = F.cross_entropy(scores, labels)

    return loss


def train(config: dict, resume_from: str | None = None) -> None:
    """Main training loop cho BGE-M3."""
    from dataset import create_dataloaders

    device = config["hardware"]["device"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────
    model_name = model_cfg["name"]
    print(f"[Model] Loading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if model_cfg["use_fp16"] else torch.float32
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device)

    # ── LoRA (tùy chọn) ─────────────────────────────────────────────
    if config["lora"]["enabled"]:
        from peft import LoraConfig, get_peft_model
        lora_cfg = config["lora"]
        lora = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Setup] Full fine-tune | trainable params: {total:,}")

    # ── Dataset ─────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(config, tokenizer)

    # ── Optimizer & Scheduler ───────────────────────────────────────
    accum = train_cfg.get("gradient_accumulation_steps", 1)
    total_steps = math.ceil(len(train_loader) / accum) * train_cfg["num_epochs"]

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": train_cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=train_cfg["learning_rate"])

    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                            total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                             milestones=[warmup_steps])

    scaler = torch.cuda.amp.GradScaler(enabled=model_cfg["use_fp16"])
    temperature = config["loss"].get("temperature", 0.05)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_count = 0

    if resume_from:
        ckpt = torch.load(Path(resume_from) / "trainer_state.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[Resume] Tiếp tục từ epoch {start_epoch}")

    # ── Training Loop ───────────────────────────────────────────────
    num_neg = config["data"].get("num_hard_negatives", 1)

    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']}")
        for step, batch in enumerate(pbar):
            with torch.cuda.amp.autocast(enabled=model_cfg["use_fp16"]):
                # Query embedding
                q_emb = get_cls_embedding(
                    model,
                    batch["query_input_ids"].to(device),
                    batch["query_attention_mask"].to(device),
                )
                # Positive embedding
                p_emb = get_cls_embedding(
                    model,
                    batch["pos_input_ids"].to(device),
                    batch["pos_attention_mask"].to(device),
                )
                # Hard negatives
                neg_embs = []
                for i in range(num_neg):
                    key_id   = f"neg{i}_input_ids"
                    key_mask = f"neg{i}_attention_mask"
                    if key_id in batch:
                        n_emb = get_cls_embedding(
                            model,
                            batch[key_id].to(device),
                            batch[key_mask].to(device),
                        )
                        neg_embs.append(n_emb)

                loss = mnrl_loss(q_emb, p_emb, neg_embs, temperature) / accum

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

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

        # ── Validation ───────────────────────────────────────────────
        val_loss = avg_loss
        if val_loader:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    with torch.cuda.amp.autocast(enabled=model_cfg["use_fp16"]):
                        q = get_cls_embedding(model, batch["query_input_ids"].to(device),
                                              batch["query_attention_mask"].to(device))
                        p = get_cls_embedding(model, batch["pos_input_ids"].to(device),
                                              batch["pos_attention_mask"].to(device))
                        v_loss = mnrl_loss(q, p, [], temperature)
                    val_losses.append(v_loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}")

        # ── Checkpoint ───────────────────────────────────────────────
        ckpt_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
        _save(model, tokenizer, optimizer, epoch, val_loss, best_val_loss, ckpt_dir)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            _save(model, tokenizer, optimizer, epoch, val_loss, best_val_loss,
                  output_dir / "best")
            print(f"[Best] val_loss={val_loss:.4f} — saved to {output_dir / 'best'}")
        else:
            patience_count += 1
            if patience_count >= train_cfg.get("early_stopping_patience", 99):
                print(f"[Early Stop] Dừng sớm sau {patience_count} epoch không cải thiện.")
                break

    print(f"\n[Done] Training complete.")
    print(f"       Run: python export_config.py --checkpoint {output_dir / 'best'}")


def _save(model, tokenizer, optimizer, epoch, val_loss, best_val_loss,
          save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    torch.save({
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "epoch":          epoch,
        "val_loss":       val_loss,
        "best_val_loss":  best_val_loss,
    }, save_dir / "trainer_state.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BGE-M3")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    os.chdir(THIS_DIR)
    cfg = load_config(args.config)
    train(cfg, resume_from=args.resume)
