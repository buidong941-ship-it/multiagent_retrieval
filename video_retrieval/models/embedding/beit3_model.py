"""
BEiT-3 image-text embedding model wrapper.

Architecture:
    - BEiT-3 multimodal foundation model.
    - Treats images as foreign languages (Masked Data Modeling).
    - Excellent for multimodal retrieval tasks.

"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from config.embedding_config import EmbeddingConfig
from interfaces.base_interfaces import BaseEmbedder
from utils.gpu_utils import get_device
from utils.logging_utils import get_logger

log = get_logger(__name__)


class Beit3Embedder(BaseEmbedder):
    """
    BEiT-3 image encoder.

    Uses timm (PyTorch Image Models) to load Microsoft's official BEiT-3 weights.
    Thread-safe after initialization.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """
        Initialize Beit3Embedder.

        Args:
            config: EmbeddingConfig with model name, device, batch_size, etc.
        """
        self.config = config
        self.device = get_device(config.device)
        self._model = None
        self._transform = None

    def _load(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        # Always use timm's beit3_base_patch16_224 for Microsoft's official BEiT-3
        model_name = 'beit3_base_patch16_224'
        log.info(f"Loading BEiT-3 model from timm: {model_name}")
        
        # Load model and set num_classes=0 to extract pooled features
        self._model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self._model = self._model.eval().to(self.device)
        
        data_config = resolve_data_config({}, model=self._model)
        self._transform = create_transform(**data_config)
        
        log.info(f"BEiT-3 loaded on {self.device}")

    @property
    def embedding_dim(self) -> int:
        """Return the BEiT-3 embedding dimension."""
        # BEiT-3 base is 768
        return 768

    def encode_images(
        self,
        image_paths: list[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode images to L2-normalized embeddings.

        Args:
            image_paths: List of absolute paths to image files.
            batch_size:  Override config batch_size if provided.

        Returns:
            np.ndarray shape (N, dim), float32, L2-normalized.
        """
        self._load()
        bs = batch_size or self.config.batch_size
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(image_paths), bs):
            batch_paths = image_paths[i : i + bs]
            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(self._transform(img))

            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                embeddings = self._model(batch_tensor)
                # L2 normalize
                embs = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
                all_embeddings.append(embs)
            
            log.debug(f"Encoded image batch {i // bs + 1} | size={len(batch_paths)}")

        return np.vstack(all_embeddings)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode text strings to L2-normalized embeddings.
        Note: BEiT-3 loaded via timm only supports image encoding.
        """
        raise NotImplementedError("BEiT-3 via timm only supports image embeddings for deduplication.")


