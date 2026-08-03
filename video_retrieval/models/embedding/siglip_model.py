"""
SigLIP2 image-text embedding model wrapper.

Architecture:
    - Vision Transformer (ViT-SO400M) + Text Transformer
    - Patch size: 14×14, Resolution: 384×384
    - Embedding dim: 1152
    - Training: Sigmoid loss (SigLIP) on Web-scale image-text pairs
      (vs. softmax InfoNCE in CLIP)

Why SigLIP2 over OpenCLIP/CLIP:
    - Better multilingual zero-shot: trained on multilingual captions
    - Sigmoid loss = independent positives/negatives → better at
      fine-grained retrieval
    - 27% relative improvement on zero-shot ImageNet vs original CLIP
    - Native support for Vietnamese text through multilingual pretraining

Fine-tuning strategy:
    - Freeze vision encoder, fine-tune text encoder on Vietnamese captions
    - Or: LoRA adapters on both encoders for memory efficiency
    - Loss: InfoNCE or SigLIP loss on (frame, Vietnamese caption) pairs

Input:
    - Images: PIL RGB, resized to 384×384, normalized
    - Texts: tokenized up to 64 tokens

Output:
    - L2-normalized float32 vectors of dim=1152
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from config.embedding_config import EmbeddingConfig
from interfaces.base_interfaces import BaseEmbedder
from utils.gpu_utils import get_device
from utils.logging_utils import get_logger

log = get_logger(__name__)


class SigLIP2Embedder(BaseEmbedder):
    """
    SigLIP2 image and text encoder.

    Uses HuggingFace transformers AutoModel/AutoProcessor.
    Thread-safe after initialization (model weights are read-only).

    Attributes:
        config:    EmbeddingConfig instance.
        model:     Loaded SigLIP2 model.
        processor: AutoProcessor for image/text preprocessing.
        device:    torch device string.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """
        Initialize SigLIP2Embedder.

        Args:
            config: EmbeddingConfig with model name, device, batch_size, etc.
        """
        self.config = config
        self.device = get_device(config.device)
        self._model: Optional[AutoModel] = None
        self._processor: Optional[AutoProcessor] = None

    def _load(self) -> None:
        """Lazy-load the model on first use (avoid loading at import time)."""
        if self._model is not None:
            return

        log.info(f"Loading SigLIP2 model: {self.config.siglip_model_name}")
        kwargs: dict = {
            "torch_dtype": torch.float16 if self.config.use_fp16 else torch.float32,
        }
        if self.config.cache_dir:
            kwargs["cache_dir"] = self.config.cache_dir

        self._model = AutoModel.from_pretrained(
            self.config.siglip_model_name, **kwargs
        ).to(self.device).eval()

        from transformers import AutoImageProcessor
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.config.siglip_model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            log.warning(f"AutoProcessor failed to load ({e}). Text encoding may not work.")
            self._processor = None
            
        try:
            self._image_processor = AutoImageProcessor.from_pretrained(
                self.config.siglip_model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            log.warning(f"Failed to load AutoImageProcessor ({e}), fallback to processor.image_processor")
            if self._processor is not None:
                self._image_processor = getattr(self._processor, "image_processor", getattr(self._processor, "feature_extractor", self._processor))
            else:
                log.error("CRITICAL: Both AutoProcessor and AutoImageProcessor failed. SigLIP will not work.")
                self._image_processor = None
            
            
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        log.info(f"SigLIP2 loaded on {self.device} | fp16={self.config.use_fp16}")

    @property
    def embedding_dim(self) -> int:
        """Return the SigLIP2-so400m embedding dimension."""
        return self.config.embedding_dim  # 1152

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
            np.ndarray shape (N, 1152), float32, L2-normalized.
        """
        self._load()
        bs = batch_size or self.config.batch_size
        all_embeddings: list[np.ndarray] = []

        if self._image_processor is None:
            log.error("Image processor is None, returning empty embeddings")
            return np.array([])
            
        for i in range(0, len(image_paths), bs):
            batch_paths = image_paths[i : i + bs]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = self._image_processor(images=images, return_tensors="pt")
            model_dtype = next(self._model.parameters()).dtype
            inputs = {
                k: (v.to(dtype=model_dtype) if torch.is_floating_point(v) else v).to(self.device)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)

            if hasattr(outputs, "pooler_output"):
                embs = outputs.pooler_output.detach().cpu().float().numpy()
            elif isinstance(outputs, torch.Tensor):
                embs = outputs.detach().cpu().float().numpy()
            else:
                embs = outputs[0].detach().cpu().float().numpy()

            if self.config.normalize:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / np.maximum(norms, 1e-8)

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

        Args:
            texts:      List of text strings (Vietnamese or English).
            batch_size: Override config batch_size if provided.

        Returns:
            np.ndarray shape (N, 1152), float32, L2-normalized.
        """
        self._load()
        bs = batch_size or self.config.batch_size
        all_embeddings: list[np.ndarray] = []

        if self._processor is None:
            log.error("Text processor is None, returning empty embeddings")
            return np.array([])
            
        for i in range(0, len(texts), bs):
            batch_texts = texts[i : i + bs]

            inputs = self._processor(
                text=batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            )
            model_dtype = next(self._model.parameters()).dtype
            inputs = {
                k: (v.to(dtype=model_dtype) if torch.is_floating_point(v) else v).to(self.device)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = self._model.get_text_features(**inputs)

            if hasattr(outputs, "pooler_output"):
                embs = outputs.pooler_output.detach().cpu().float().numpy()
            elif hasattr(outputs, "text_embeds"):
                embs = outputs.text_embeds.detach().cpu().float().numpy()
            elif isinstance(outputs, torch.Tensor):
                embs = outputs.detach().cpu().float().numpy()
            else:
                embs = outputs[0].detach().cpu().float().numpy()

            if self.config.normalize:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / np.maximum(norms, 1e-8)

            all_embeddings.append(embs)

        return np.vstack(all_embeddings)
