"""
Action embedding service.

Computes mean-pooled embeddings over consecutive frames 
to represent actions, and stores them in Milvus.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from tqdm import tqdm

from config.database_config import MilvusConfig
from database.milvus.milvus_client import MilvusVectorDatabase
from interfaces.base_interfaces import FrameRecord
from utils.logging_utils import get_logger

log = get_logger(__name__)


class ActionEmbeddingService:
    """
    Computes moving averages of CLIP embeddings to capture actions.
    """
    def __init__(
        self,
        config: MilvusConfig,
        vector_db: MilvusVectorDatabase,
        window_size: int = 5,
    ) -> None:
        self.config = config
        self.vector_db = vector_db
        self.window_size = window_size
        self._clip_collection = config.clip_collection_name
        self._action_collection = config.action_collection_name

    def setup_collection(self) -> None:
        self.vector_db.create_collection_if_not_exists(self._action_collection)

    async def process(self, frames: list[FrameRecord], id_offset: int = 0) -> None:
        """
        Process frames to compute action embeddings.
        Args:
            frames: Flattened list of FrameRecords.
            id_offset: Starting ID for the action collection.
        """
        self.setup_collection()
        
        # Group frames by video_id and sort by frame_idx
        from collections import defaultdict
        video_groups: dict[str, list[FrameRecord]] = defaultdict(list)
        for f in frames:
            video_groups[f.video_id].append(f)
                
        for vid in video_groups:
            video_groups[vid].sort(key=lambda x: x.frame_idx)

        total_inserted = 0
        current_id = id_offset

        for video_id, vid_frames in tqdm(video_groups.items(), desc="Computing Action Embeddings"):
            if len(vid_frames) < self.window_size:
                log.debug(f"Video {video_id} has fewer than {self.window_size} frames. Skipping.")
                continue
                
            frame_ids = [f.frame_id for f in vid_frames]
            
            # Query Milvus for all original CLIP embeddings for this video
            emb_map = {}
            filter_expr = f'video_id == "{video_id}"'
            try:
                results = self.vector_db.query(
                    collection_name=self._clip_collection,
                    filter_expr=filter_expr,
                    output_fields=["frame_id", "embedding"]
                )
                for res in results:
                    emb_map[res["frame_id"]] = res["embedding"]
            except Exception as exc:
                log.error(f"Failed to fetch embeddings for video {video_id}: {exc}")
                    
            # Ordered embeddings
            ordered_embs = []
            for fid in frame_ids:
                if fid in emb_map:
                    ordered_embs.append(np.array(emb_map[fid], dtype=np.float32))
                else:
                    ordered_embs.append(np.zeros(self.config.clip_dim, dtype=np.float32))
                    
            # Compute moving average
            action_embs = []
            action_metadata = []
            action_ids = []
            
            half_w = self.window_size // 2
            for i in range(half_w, len(ordered_embs) - half_w):
                window = ordered_embs[i - half_w : i + half_w + 1]
                mean_emb = np.mean(window, axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb = mean_emb / norm
                    
                center_frame = vid_frames[i]
                action_embs.append(mean_emb)
                action_ids.append(current_id)
                action_metadata.append({
                    "frame_id": center_frame.frame_id,
                    "video_id": center_frame.video_id,
                    "frame_idx": center_frame.frame_idx,
                    "timestamp": center_frame.timestamp,
                })
                current_id += 1
                
            if action_embs:
                try:
                    self.vector_db.insert(
                        collection_name=self._action_collection,
                        ids=action_ids,
                        embeddings=np.array(action_embs, dtype=np.float32),
                        metadata=action_metadata,
                    )
                    total_inserted += len(action_embs)
                except Exception as exc:
                    log.error(f"Failed to insert action embeddings for video {video_id}: {exc}")

        self.vector_db.flush(self._action_collection)
        log.info(f"Action embeddings generation complete. Inserted {total_inserted} vectors.")
