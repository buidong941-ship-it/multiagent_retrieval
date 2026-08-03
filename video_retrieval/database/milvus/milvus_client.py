"""
Milvus vector database client.

Design Decision:
    - Wraps pymilvus MilvusClient (newer SDK, simpler API).
    - Provides create_collection_if_not_exists with HNSW index.
    - All insert/search operations go through this single client.
    - Connection is established lazily on first operation.

HNSW vs IVF trade-offs:
    ┌─────────────┬──────────┬─────────┬──────────┐
    │ Index       │ Recall   │ Speed   │ Memory   │
    ├─────────────┼──────────┼─────────┼──────────┤
    │ HNSW        │ High     │ Fast    │ High     │
    │ IVF_FLAT    │ High     │ Medium  │ Low      │
    │ IVF_PQ      │ Medium   │ Fast    │ Very Low │
    │ SCANN       │ High     │ Fast    │ Medium   │
    └─────────────┴──────────┴─────────┴──────────┘
    → For competition: HNSW (max recall, latency < 100ms for 1M vectors)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pymilvus import MilvusClient as _MilvusClient
from pymilvus import CollectionSchema, DataType

from config.database_config import MilvusConfig
from database.milvus.milvus_schema import COLLECTION_SCHEMAS
from interfaces.base_interfaces import BaseVectorDatabase
from utils.logging_utils import get_logger

log = get_logger(__name__)


class MilvusVectorDatabase(BaseVectorDatabase):
    """
    Milvus vector database client implementing BaseVectorDatabase.

    Wraps pymilvus.MilvusClient for simplified operations.

    Attributes:
        config:  MilvusConfig instance.
        _client: pymilvus MilvusClient (lazy-connected).
    """

    def __init__(self, config: MilvusConfig) -> None:
        """
        Initialize MilvusVectorDatabase.

        Args:
            config: MilvusConfig with host, port, collection names, etc.
        """
        self.config = config
        self._client: Optional[_MilvusClient] = None

    def _connect(self) -> None:
        """Establish connection to Milvus if not already connected."""
        if self._client is not None:
            return

        uri = self.config.uri
        log.info(f"Connecting to Milvus at {uri}")

        connect_kwargs: dict[str, Any] = {"uri": uri, "db_name": self.config.db_name}
        if self.config.user:
            connect_kwargs["user"] = self.config.user
            connect_kwargs["password"] = self.config.password

        self._client = _MilvusClient(**connect_kwargs)
        log.info("Milvus connection established")

    def create_collection_if_not_exists(self, collection_name: str) -> None:
        """
        Create a Milvus collection with HNSW index if it doesn't exist.

        Args:
            collection_name: Name of the collection to create.

        Raises:
            KeyError: If collection_name has no schema defined.
        """
        self._connect()

        if self._client.has_collection(collection_name):
            log.info(f"Collection '{collection_name}' already exists")
            return

        if collection_name not in COLLECTION_SCHEMAS:
            raise KeyError(f"No schema defined for collection: {collection_name}")

        schema = COLLECTION_SCHEMAS[collection_name]

        # HNSW index builder in Milvus Lite currently has a bug where it deletes the 
        # parquet file but fails to update manifest.json, corrupting the database on zip/transfer.
        # We use FLAT index (brute-force) which is perfectly fast enough for competition scale
        # and avoids the index-builder corruption bug entirely.
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type=self.config.metric_type,
        )

        self._client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        log.info(f"Created collection '{collection_name}' with FLAT index (bug workaround)")

    def insert(
        self,
        collection_name: str,
        ids: list[int],
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """
        Insert embeddings with metadata into a collection.

        Args:
            collection_name: Target collection.
            ids:             Integer primary keys.
            embeddings:      np.ndarray shape (N, dim).
            metadata:        List of dicts with scalar field values.

        Raises:
            ValueError: If lengths don't match.
        """
        self._connect()

        if len(ids) != len(embeddings) != len(metadata):
            raise ValueError("ids, embeddings and metadata must have same length")

        # Build data rows
        data = []
        for i, (pk, emb, meta) in enumerate(zip(ids, embeddings, metadata)):
            row: dict[str, Any] = {"milvus_id": pk, "embedding": emb.tolist()}
            row.update(meta)
            data.append(row)

        result = self._client.insert(collection_name=collection_name, data=data)
        log.debug(
            f"Inserted {len(data)} vectors into '{collection_name}' | "
            f"insert_count={result.get('insert_count', '?')}"
        )

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int,
        filter_expr: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for top-k nearest neighbours.

        Args:
            collection_name: Collection to search.
            query_vector:    1-D float32 numpy array.
            top_k:           Number of results to return.
            filter_expr:     Optional Milvus boolean expression filter
                             (e.g., 'video_id == "vid_001"').

        Returns:
            List of dicts: {id, distance, frame_id, video_id, ...}
        """
        self._connect()
        self._client.load_collection(collection_name)
        # In HNSW, 'ef' must be >= top_k. If top_k is requested to be very large,
        # dynamically boost ef to match it.
        ef_search = max(self.config.hnsw_ef_search, top_k)
        search_params = {"ef": ef_search}

        results = self._client.search(
            collection_name=collection_name,
            data=[query_vector.tolist()],
            anns_field="embedding",
            search_params={"metric_type": self.config.metric_type, "params": search_params},
            limit=top_k,
            filter=filter_expr,
            output_fields=["frame_id", "video_id", "frame_idx", "timestamp"],
        )

        hits = []
        for hit in results[0]:
            try:
                hits.append(
                    {
                        "id": hit.get("id", hit.get("milvus_id", 0)),
                        "score": hit.get("distance", 0.0),
                        **hit.get("entity", hit),
                    }
                )
            except Exception as e:
                log.error(f"Error parsing hit {hit}: {e}")
                
        return hits

    def query(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Query records by filter expression (e.g. 'milvus_id in [1, 2, 3]').

        Args:
            collection_name: Collection to query.
            filter_expr:     Milvus boolean expression filter.
            output_fields:   List of fields to return.

        Returns:
            List of dicts representing the queried records.
        """
        self._connect()
        self._client.load_collection(collection_name)
        results = self._client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=output_fields or ["milvus_id"],
        )
        return results

    def delete(self, collection_name: str, ids: list[int]) -> None:
        """Delete records by primary key IDs."""
        self._connect()
        self._client.delete(collection_name=collection_name, ids=ids)
        log.info(f"Deleted {len(ids)} records from '{collection_name}'")

    def get_embeddings_by_ids(
        self,
        needed_list: list[Union[int, str]],
        collection_name: str = "clip_embeddings",
    ) -> dict[str, np.ndarray]:
        """
        Fetch vector embeddings by list of frame_ids or numeric IDs.
        """
        self._connect()
        self._client.load_collection(collection_name)
        embeddings_map = {}
        chunk_size = 1000
        for i in range(0, len(needed_list), chunk_size):
            chunk = needed_list[i : i + chunk_size]
            in_list = ", ".join(f"'{fid}'" for fid in chunk)
            filter_expr = f"frame_id in [{in_list}]"
            try:
                res = self._client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=["frame_id", "embedding"]
                )
                for r in res:
                    emb = r.get("embedding")
                    if emb:
                        embeddings_map[r["frame_id"]] = np.array(emb, dtype=np.float32)
            except Exception as e:
                log.error(f"Milvus batch query error: {e}")
        return embeddings_map

    def count(self, collection_name: str) -> int:
        """Return the number of vectors in a collection."""
        self._connect()
        stats = self._client.get_collection_stats(collection_name)
        return int(stats.get("row_count", 0))

    def flush(self, collection_name: str) -> None:
        """Flush a collection to ensure all inserts are persisted."""
        self._connect()
        self._client.flush(collection_name)
        log.debug(f"Flushed collection '{collection_name}'")

    def close(self) -> None:
        """Close the Milvus client and flush to disk (crucial for milvus-lite)."""
        if self._client is not None:
            self._client.close()
            log.info("Milvus connection closed")
            self._client = None
