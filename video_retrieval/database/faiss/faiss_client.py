"""
FAISS Vector Database client.

Design Decision:
    - Implements BaseVectorDatabase using FAISS (Facebook AI Similarity Search).
    - Uses Inner Product (IP) index for L2-normalized embeddings (Cosine similarity).
    - Maintains an in-memory & persisted metadata store mapping internal numeric IDs to payload dicts.
    - Maintains a frame_id to numeric ID index for fast string-based vector lookups.
    - Saves index files (.faiss) and metadata files (.pkl) to disk.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np

from config.database_config import FaissConfig
from interfaces.base_interfaces import BaseVectorDatabase
from utils.logging_utils import get_logger

log = get_logger(__name__)


class FaissVectorDatabase(BaseVectorDatabase):
    """
    FAISS-backed vector database implementation.

    Attributes:
        config: FaissConfig instance.
        indices: Dict mapping collection_name -> faiss.IndexIDMap2
        metadata_store: Dict mapping collection_name -> dict[int, dict[str, Any]]
        frame_id_to_id: Dict mapping collection_name -> dict[str, int]
    """

    def __init__(self, config: FaissConfig) -> None:
        self.config = config
        self.index_dir = Path(config.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.indices: Dict[str, faiss.IndexIDMap2] = {}
        self.metadata_store: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.frame_id_to_id: Dict[str, Dict[str, int]] = {}
        self.vectors_store: Dict[str, Dict[int, np.ndarray]] = {}

        # Auto-load existing collections in index_dir
        self._load_all_existing()

    def _get_collection_paths(self, collection_name: str) -> tuple[Path, Path]:
        """Return paths for the index file and metadata file."""
        index_path = self.index_dir / f"{collection_name}.faiss"
        meta_path = self.index_dir / f"{collection_name}.pkl"
        return index_path, meta_path

    def _load_all_existing(self) -> None:
        """Discover and load all .faiss files in index_dir."""
        for faiss_file in self.index_dir.glob("*.faiss"):
            collection_name = faiss_file.stem
            self.load_collection(collection_name)

    def load_collection(self, collection_name: str) -> None:
        """Load collection index and metadata from disk if present."""
        index_path, meta_path = self._get_collection_paths(collection_name)

        if index_path.exists() and meta_path.exists():
            log.info(f"Loading FAISS index for '{collection_name}' from {index_path}")
            try:
                index = faiss.read_index(str(index_path))
                with open(meta_path, "rb") as f:
                    meta_data = pickle.load(f)

                self.indices[collection_name] = index
                self.metadata_store[collection_name] = meta_data.get("metadata", {})
                self.frame_id_to_id[collection_name] = meta_data.get("frame_id_to_id", {})
                # DO NOT keep vectors in RAM to save memory! Use index.reconstruct(id) instead.
                # self.vectors_store[collection_name] = meta_data.get("vectors", {})
                self.vectors_store[collection_name] = {}
                log.info(
                    f"Successfully loaded '{collection_name}' with {index.ntotal} vectors."
                )
            except Exception as e:
                log.error(f"Failed to load FAISS collection '{collection_name}': {e}")
        else:
            log.debug(f"No existing FAISS index found for '{collection_name}' at {index_path}")

    def create_collection_if_not_exists(
        self, collection_name: str, dim: int = 1152
    ) -> None:
        """Create an empty FAISS index for collection_name if not loaded."""
        if collection_name in self.indices:
            return

        index_path, _ = self._get_collection_paths(collection_name)
        if index_path.exists():
            self.load_collection(collection_name)
            return

        log.info(f"Creating new FAISS index for '{collection_name}' (dim={dim})")
        # Base index: Flat Inner Product (for normalized cosine similarity)
        if self.config.metric_type.upper() == "L2":
            base_index = faiss.IndexFlatL2(dim)
        else:
            base_index = faiss.IndexFlatIP(dim)

        # Wrap with IDMap2 to support explicit int64 IDs
        index = faiss.IndexIDMap2(base_index)

        self.indices[collection_name] = index
        self.metadata_store[collection_name] = {}
        self.frame_id_to_id[collection_name] = {}
        self.vectors_store[collection_name] = {}

    def insert(
        self,
        collection_name: str,
        ids: list[int],
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """
        Insert embeddings with metadata into FAISS collection.

        Args:
            collection_name: Target collection.
            ids: Integer primary keys.
            embeddings: np.ndarray shape (N, dim), float32.
            metadata: List of dicts containing metadata fields.
        """
        if len(ids) != len(embeddings) != len(metadata):
            raise ValueError("ids, embeddings, and metadata must have the same length")

        if len(ids) == 0:
            return

        dim = embeddings.shape[1]
        self.create_collection_if_not_exists(collection_name, dim=dim)

        index = self.indices[collection_name]
        meta_dict = self.metadata_store[collection_name]
        frame_idx_dict = self.frame_id_to_id[collection_name]
        vec_dict = self.vectors_store[collection_name]

        # Convert to float32 contiguous array
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        ids_array = np.array(ids, dtype=np.int64)

        # Add to FAISS index
        index.add_with_ids(embeddings_f32, ids_array)

        # Store metadata
        for idx, pk, meta, emb in zip(range(len(ids)), ids, metadata, embeddings_f32):
            meta_dict[pk] = meta
            vec_dict[pk] = emb
            frame_id = meta.get("frame_id")
            if frame_id:
                frame_idx_dict[frame_id] = pk

        log.debug(f"Inserted {len(ids)} vectors into FAISS collection '{collection_name}'")

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int,
        filter_expr: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for top-k nearest neighbors.

        Args:
            collection_name: Collection name.
            query_vector: 1-D float32 numpy array.
            top_k: Top K results to return.
            filter_expr: Optional expression string (Basic support for filtering).

        Returns:
            List of dicts: {id, score, frame_id, video_id, frame_idx, timestamp, ...}
        """
        if collection_name not in self.indices:
            self.load_collection(collection_name)

        if collection_name not in self.indices or self.indices[collection_name].ntotal == 0:
            log.warning(f"Collection '{collection_name}' is empty or does not exist.")
            return []

        index = self.indices[collection_name]
        meta_dict = self.metadata_store[collection_name]

        query_f32 = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)

        # Request extra if filter_expr is supplied to handle filtered post-processing
        search_k = min(top_k * 5 if filter_expr else top_k, index.ntotal)

        distances, indices = index.search(query_f32, search_k)

        hits = []
        for dist, pk in zip(distances[0], indices[0]):
            if pk == -1:
                continue

            meta = meta_dict.get(int(pk), {})

            similarity = float(dist)
            distance = 1.0 - similarity

            hit = {
                "id": int(pk),
                "distance": distance,
                "score": distance,  # Contract matching BaseVectorDatabase (distance format, smaller is closer)
                "similarity": similarity,
                "frame_id": meta.get("frame_id", f"frame_{pk}"),
                "video_id": meta.get("video_id", ""),
                "frame_idx": meta.get("frame_idx", 0),
                "timestamp": meta.get("timestamp", 0.0),
                "entity": meta,
            }

            # Handle filter expression (e.g., frame_id in [...])
            if filter_expr and "frame_id in [" in filter_expr:
                allowed_ids_str = filter_expr.split("[")[1].split("]")[0]
                allowed_ids = {s.strip("'\" ") for s in allowed_ids_str.split(",")}
                if hit["frame_id"] not in allowed_ids:
                    continue

            hits.append(hit)
            if len(hits) >= top_k:
                break

        return hits

    def get_embeddings_by_ids(
        self,
        needed_list: list[Union[int, str]],
        collection_name: str = "clip_embeddings",
    ) -> dict[str, np.ndarray]:
        """
        Fetch vector embeddings by list of frame_ids or numeric IDs.

        Used extensively by NeighborReranker.
        """
        if collection_name not in self.indices:
            self.load_collection(collection_name)

        embeddings_map: dict[str, np.ndarray] = {}
        index = self.indices.get(collection_name)
        if not index:
            return embeddings_map

        frame_id_to_id = self.frame_id_to_id.get(collection_name, {})
        for needed_id in needed_list:
            if isinstance(needed_id, str) and needed_id in frame_id_to_id:
                pk = frame_id_to_id[needed_id]
            elif isinstance(needed_id, int):
                pk = needed_id
            else:
                pk = -1

            if pk != -1:
                try:
                    # Reconstruct embedding vector directly from FAISS index using ID
                    vec = index.reconstruct(int(pk))
                    embeddings_map[str(needed_id)] = np.array(vec, dtype=np.float32)
                except Exception as e:
                    # Ignore if ID not found in index
                    pass

        return embeddings_map

    def delete(self, collection_name: str, ids: list[int]) -> None:
        """Delete records by ID."""
        if collection_name not in self.indices:
            return

        index = self.indices[collection_name]
        ids_array = np.array(ids, dtype=np.int64)
        index.remove_ids(ids_array)

        meta_dict = self.metadata_store.get(collection_name, {})
        vec_dict = self.vectors_store.get(collection_name, {})
        frame_idx_dict = self.frame_id_to_id.get(collection_name, {})

        for pk in ids:
            if pk in meta_dict:
                fid = meta_dict[pk].get("frame_id")
                if fid in frame_idx_dict:
                    del frame_idx_dict[fid]
                del meta_dict[pk]
            if pk in vec_dict:
                del vec_dict[pk]

        log.info(f"Deleted {len(ids)} vectors from FAISS collection '{collection_name}'")

    def count(self, collection_name: str) -> int:
        """Return total vector count in collection."""
        if collection_name not in self.indices:
            self.load_collection(collection_name)
        if collection_name in self.indices:
            return self.indices[collection_name].ntotal
        return 0

    def flush(self, collection_name: Optional[str] = None) -> None:
        """Persist index and metadata to disk."""
        collections_to_save = (
            [collection_name] if collection_name else list(self.indices.keys())
        )

        for col in collections_to_save:
            if col in self.indices:
                index_path, meta_path = self._get_collection_paths(col)
                log.info(f"Flushing FAISS collection '{col}' to disk at {index_path}")

                faiss.write_index(self.indices[col], str(index_path))

                save_payload = {
                    "metadata": self.metadata_store.get(col, {}),
                    "frame_id_to_id": self.frame_id_to_id.get(col, {}),
                    "vectors": self.vectors_store.get(col, {}),
                }

                with open(meta_path, "wb") as f:
                    pickle.dump(save_payload, f)

    def close(self) -> None:
        """Flush to disk and release memory."""
        self.flush()
        self.indices.clear()
        self.metadata_store.clear()
        self.frame_id_to_id.clear()
        self.vectors_store.clear()
        log.info("FAISS database closed successfully.")
