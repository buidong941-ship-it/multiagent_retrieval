"""
Jina CLIP v2 embedding model wrapper.
"""
from __future__ import annotations

from typing import Optional
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from config.embedding_config import EmbeddingConfig
from interfaces.base_interfaces import BaseEmbedder
from utils.gpu_utils import get_device
from utils.logging_utils import get_logger

log = get_logger(__name__)

class JinaClipEmbedder(BaseEmbedder):
    """
    Jina CLIP v2 image and text encoder.
    Uses transformers AutoModel with trust_remote_code=True.
    """
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.device = get_device(config.device)
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return

        # Monkey-patch clip_loss into transformers if missing (for jina-clip-v2 compatibility with transformers 4.44+)
        try:
            import torch
            import transformers.models.clip.modeling_clip as clip_module
            if not hasattr(clip_module, "clip_loss"):
                def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
                    caption_loss = torch.nn.functional.cross_entropy(similarity, torch.arange(len(similarity), device=similarity.device))
                    image_loss = torch.nn.functional.cross_entropy(similarity.t(), torch.arange(len(similarity), device=similarity.device))
                    return (caption_loss + image_loss) / 2.0
                clip_module.clip_loss = clip_loss
        except Exception:
            pass

        model_name = getattr(self.config, "jina_model_name", "jinaai/jina-clip-v2")
        log.info(f"Loading Jina CLIP v2 model: {model_name}")
        
        # trust_remote_code=True is required for jina-clip-v2
        # low_cpu_mem_usage=False prevents meta tensor errors in transformers
        import transformers.utils.import_utils
        import transformers.modeling_utils
        
        old_is_accelerate = transformers.utils.import_utils.is_accelerate_available
        old_is_accelerate_mu = getattr(transformers.modeling_utils, "is_accelerate_available", old_is_accelerate)
        
        transformers.utils.import_utils.is_accelerate_available = lambda *args, **kwargs: False
        transformers.modeling_utils.is_accelerate_available = lambda *args, **kwargs: False
        
        device_str = "cuda" if "cuda" in str(self.device) else "cpu"
        try:
            self._model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                _fast_init=False
            )
            self._model = self._model.to(self.device)
        except Exception as exc:
            log.warning(f"Failed to load directly ({exc}). Trying with device_map...")
            transformers.utils.import_utils.is_accelerate_available = old_is_accelerate
            transformers.modeling_utils.is_accelerate_available = old_is_accelerate_mu
            self._model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=device_str
            )
        finally:
            transformers.utils.import_utils.is_accelerate_available = old_is_accelerate
            transformers.modeling_utils.is_accelerate_available = old_is_accelerate_mu
            
        self._model = self._model.eval()
        
        # Free CPU memory spike caused by loading model before moving to device
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        log.info(f"{model_name} loaded on {self.device}")

    @property
    def embedding_dim(self) -> int:
        return 1024 if "v2" in getattr(self.config, "jina_model_name", "") else 768

    def encode_images(self, image_paths: list[str], batch_size: Optional[int] = None) -> np.ndarray:
        self._load()
        bs = batch_size or self.config.batch_size
        all_embeddings = []

        for i in range(0, len(image_paths), bs):
            batch_paths = image_paths[i : i + bs]
            
            valid_paths = [p for p in batch_paths if Path(p).exists()]
            if not valid_paths:
                all_embeddings.append(np.zeros((len(batch_paths), self.embedding_dim), dtype=np.float32))
                continue

            try:
                with torch.no_grad():
                    # Pass string file paths directly to jina-clip-v2 encode_image
                    embs = self._model.encode_image(valid_paths)
            except Exception as exc:
                log.warning(f"Direct path encode_image failed ({exc}). Retrying with PIL Images.")
                images = []
                for p in valid_paths:
                    try:
                        images.append(Image.open(p).convert("RGB"))
                    except Exception:
                        images.append(Image.new("RGB", (224, 224)))
                try:
                    with torch.no_grad():
                        embs = self._model.encode_image(images)
                except Exception as exc2:
                    log.error(f"Fallback encode_image failed ({exc2}). Using zero vectors.")
                    embs = np.zeros((len(valid_paths), self.embedding_dim), dtype=np.float32)

            if isinstance(embs, torch.Tensor):
                embs = embs.detach().cpu().float().numpy()

            if self.config.normalize:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / np.maximum(norms, 1e-8)

            all_embeddings.append(embs)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def encode_texts(self, texts: list[str], batch_size: Optional[int] = None) -> np.ndarray:
        self._load()
        bs = batch_size or self.config.batch_size
        all_embeddings = []

        for i in range(0, len(texts), bs):
            batch_texts = texts[i : i + bs]

            with torch.no_grad():
                # encode_text expects list of strings
                try:
                    embs = self._model.encode_text(batch_texts, max_length=256, truncation=True)
                except TypeError:
                    embs = self._model.encode_text(batch_texts)

            if isinstance(embs, torch.Tensor):
                embs = embs.detach().cpu().float().numpy()

            if self.config.normalize:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / np.maximum(norms, 1e-8)

            all_embeddings.append(embs)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])
