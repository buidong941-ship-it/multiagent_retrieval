"""
SigLIP Loss và InfoNCE Loss.

SigLIP Loss (khác CLIP):
    - CLIP dùng softmax → tất cả cặp trong batch cạnh tranh nhau
    - SigLIP dùng sigmoid → mỗi cặp (image, text) được xét độc lập
    - Tốt hơn khi batch nhỏ, và khi 1 ảnh có thể match nhiều caption
    - Công thức:
        loss = -mean(y * log(σ(logit)) + (1-y) * log(1 - σ(logit)))
        trong đó y = 1 nếu cặp dương, y = 0 nếu cặp âm

InfoNCE Loss (CLIP gốc):
    - Dùng khi muốn tương thích với CLIP training scheme
    - Phù hợp hơn khi batch size lớn (≥ 256)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPLoss(nn.Module):
    """
    SigLIP (Sigmoid Language-Image Pre-training) loss.

    Với batch gồm N cặp (image_i, text_i):
        - Ma trận logits shape (N, N)
        - Diagonal = positive pairs (label = 1)
        - Off-diagonal = negative pairs (label = -1)

    Args:
        init_temperature: Nhiệt độ khởi đầu (learnable).
        init_bias:        Bias khởi đầu (learnable).
    """

    def __init__(
        self,
        init_temperature: float = 10.0,
        init_bias: float = -10.0,
    ) -> None:
        super().__init__()
        # t và b được học cùng model (như trong paper SigLIP)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def forward(
        self,
        image_embeddings: torch.Tensor,  # (N, D)
        text_embeddings: torch.Tensor,   # (N, D)
    ) -> torch.Tensor:
        """
        Tính SigLIP loss.

        Args:
            image_embeddings: L2-normalized image vectors, shape (N, D).
            text_embeddings:  L2-normalized text vectors, shape (N, D).

        Returns:
            Scalar loss tensor.
        """
        n = image_embeddings.shape[0]
        temperature = self.log_temperature.exp()

        # Ma trận cosine similarity (N, N)
        # L2-normalized → dot product = cosine similarity
        logits = torch.matmul(image_embeddings, text_embeddings.T) * temperature
        logits = logits + self.bias

        # Labels: +1 cho diagonal (positive), -1 cho off-diagonal (negative)
        labels = 2 * torch.eye(n, device=logits.device) - 1  # {-1, +1}

        # Sigmoid binary cross-entropy
        # log(σ(y * logit)) = -softplus(-y * logit)
        loss = -F.logsigmoid(labels * logits).sum() / n

        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE (CLIP) loss — symmetric cross-entropy.

    Dùng khi batch size lớn (≥ 128) để có đủ negatives.

    Args:
        temperature: Nhiệt độ cố định (hoặc learnable).
        learnable_temp: Nếu True, temperature được học.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temp: bool = True,
    ) -> None:
        super().__init__()
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(temperature)))

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính InfoNCE loss (symmetric).

        Args:
            image_embeddings: (N, D) L2-normalized.
            text_embeddings:  (N, D) L2-normalized.

        Returns:
            Scalar loss.
        """
        temp = self.log_temp.exp()
        logits = torch.matmul(image_embeddings, text_embeddings.T) / temp  # (N, N)

        # Labels = diagonal indices (0, 1, 2, ..., N-1)
        labels = torch.arange(logits.shape[0], device=logits.device)

        # Image → Text và Text → Image
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2


def build_loss(config: dict) -> nn.Module:
    """
    Factory: tạo loss function từ config.

    Args:
        config: Dict từ config.yaml (phần "loss").

    Returns:
        SigLIPLoss hoặc InfoNCELoss.
    """
    loss_type = config["loss"].get("type", "siglip")
    if loss_type == "siglip":
        return SigLIPLoss()
    elif loss_type == "infonce":
        temp = config["loss"].get("temperature", 0.07)
        return InfoNCELoss(temperature=temp)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
