"""
SQLite metadata database using SQLAlchemy async.

Design Decision:
    - SQLite is sufficient for competition scale (< 1M frames).
    - Async SQLAlchemy with aiosqlite for non-blocking I/O in FastAPI.
    - Session-per-request pattern via async context manager.
    - PostgreSQL drop-in replacement: change the URL only.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from sqlalchemy import and_, select, func, desc, delete
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from config.base_config import BASE_DIR
from database.metadata.schema import Base, Detection, Frame, Video
from interfaces.base_interfaces import BaseMetadataDB
from utils.logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_DB_PATH = BASE_DIR / "data" / "indexes" / "metadata.db"
DEFAULT_DB_URL = f"sqlite+aiosqlite:///{DEFAULT_DB_PATH}"


class MetadataDatabase(BaseMetadataDB):
    """
    Async SQLAlchemy metadata database.

    Manages video, frame, and detection metadata.

    Attributes:
        db_url:  SQLAlchemy async connection URL.
        _engine: Async SQLAlchemy engine.
        _session_factory: Async session factory.
    """

    def __init__(self, db_url: str = DEFAULT_DB_URL) -> None:
        """
        Initialize MetadataDatabase.

        Args:
            db_url: SQLAlchemy async DB URL.
                    Default: sqlite+aiosqlite:///data/indexes/metadata.db
        """
        self.db_url = db_url
        Path(DEFAULT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_async_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False},  # SQLite-specific
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init_db(self) -> None:
        """Create all tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("Database tables initialized")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide an async database session context manager."""
        async with self._session_factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise

    # ── Video operations ─────────────────────────────────────────────────

    async def upsert_video(self, video_data: dict[str, Any]) -> None:
        """
        Insert or update a Video record.

        Args:
            video_data: Dict with video_id, video_path, duration, fps, etc.
        """
        async with self.session() as sess:
            stmt = select(Video).where(Video.video_id == video_data.get("video_id"))
            result = await sess.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing:
                for k, v in video_data.items():
                    setattr(existing, k, v)
            else:
                sess.add(Video(**video_data))

    async def get_video(self, video_id: str) -> Optional[dict[str, Any]]:
        """Return video metadata by video_id."""
        async with self.session() as sess:
            stmt = select(Video).where(Video.video_id == video_id)
            result = await sess.execute(stmt)
            video = result.scalar_one_or_none()
            if video is None:
                return None
                
            path_str = video.video_path
            if path_str:
                if "data/videos/" in path_str.replace("\\", "/"):
                    idx = path_str.replace("\\", "/").find("data/videos/")
                    path_str = path_str[idx:]
                elif "datasets/" in path_str.replace("\\", "/"):
                    # For /kaggle/input/datasets/....
                    idx = path_str.replace("\\", "/").find("datasets/")
                    # Just keep the filename if we can't find data/videos
                    path_str = f"data/videos/{Path(path_str).name}"
                    
            return {
                "video_id": video.video_id,
                "video_path": path_str,
                "duration": video.duration,
                "fps": video.fps,
                "total_frames": video.total_frames,
            }

    async def get_video_fps_async(self, video_id: str) -> Optional[float]:
        """Return the FPS of a video by video_id, or None if not found.

        Used by _TemporalCache in agent_tools to convert window_seconds
        into the correct number of frames for neighbour lookups.
        """
        async with self.session() as sess:
            stmt = select(Video.fps).where(Video.video_id == video_id)
            result = await sess.execute(stmt)
            return result.scalar_one_or_none()

    # ── Frame operations ─────────────────────────────────────────────────

    async def upsert_frame(self, frame_data: dict[str, Any]) -> None:
        """
        Insert or update a Frame record.

        Args:
            frame_data: Dict matching Frame column names.
        """
        async with self.session() as sess:
            stmt = select(Frame).where(Frame.frame_id == frame_data["frame_id"])
            result = await sess.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing:
                for k, v in frame_data.items():
                    setattr(existing, k, v)
            else:
                sess.add(Frame(**frame_data))

    async def delete_frame_async(self, frame_id: str) -> None:
        """
        Delete a frame record from the database.
        
        Args:
            frame_id: Globally unique frame string ID.
        """
        async with self.session() as sess:
            stmt = delete(Frame).where(Frame.frame_id == frame_id)
            await sess.execute(stmt)

    async def upsert_frames_bulk(self, frames_data: list[dict[str, Any]]) -> None:
        """
        Bulk insert frames for performance.

        Args:
            frames_data: List of frame dicts.
        """
        if not frames_data:
            return
        async with self.session() as sess:
            stmt = insert(Frame).values(frames_data)
            stmt = stmt.on_conflict_do_nothing(index_elements=['frame_id'])
            await sess.execute(stmt)
        log.debug(f"Bulk upserted {len(frames_data)} frames")

    def get_frame(self, frame_id: str) -> Optional[dict[str, Any]]:
        """Sync-compatible: Return frame metadata by frame_id (placeholder)."""
        raise NotImplementedError("Use async get_frame_async instead")

    async def get_frame_async(self, frame_id: str) -> Optional[dict[str, Any]]:
        """
        Return frame metadata dict by frame_id.

        Args:
            frame_id: Globally unique frame string ID.

        Returns:
            Dict with frame metadata or None if not found.
        """
        async with self.session() as sess:
            stmt = select(Frame).where(Frame.frame_id == frame_id)
            result = await sess.execute(stmt)
            frame = result.scalar_one_or_none()
            if frame is None:
                return None
            return self._frame_to_dict(frame)

    def get_frames_by_video(self, video_id: str) -> list[dict[str, Any]]:
        """Placeholder — use async version."""
        raise NotImplementedError("Use get_frames_by_video_async")

    async def get_frames_by_video_async(
        self, video_id: str
    ) -> list[dict[str, Any]]:
        """Return all frames for a video, ordered by frame_idx."""
        async with self.session() as sess:
            stmt = (
                select(Frame)
                .where(Frame.video_id == video_id)
                .order_by(Frame.frame_idx)
            )
            result = await sess.execute(stmt)
            frames = result.scalars().all()
            return [self._frame_to_dict(f) for f in frames]

    async def get_all_frames_async(self) -> list[dict[str, Any]]:
        """Return all frames across all videos, ordered by frame_id."""
        async with self.session() as sess:
            stmt = select(Frame).order_by(Frame.frame_id)
            result = await sess.execute(stmt)
            frames = result.scalars().all()
            return [self._frame_to_dict(f) for f in frames]

    def get_neighbouring_frames(
        self, video_id: str, frame_idx: int, window: int
    ) -> list[dict[str, Any]]:
        """Placeholder — use async version."""
        raise NotImplementedError("Use get_neighbouring_frames_async")

    async def get_neighbouring_frames_async(
        self, video_id: str, frame_idx: int, window: int
    ) -> list[dict[str, Any]]:
        """
        Return frames within ±window of frame_idx.

        Used by temporal refinement module.

        Args:
            video_id:  Video identifier.
            frame_idx: Center frame index.
            window:    Number of frames to each side.

        Returns:
            List of frame metadata dicts.
        """
        lo = max(0, frame_idx - window)
        hi = frame_idx + window

        async with self.session() as sess:
            stmt = (
                select(Frame)
                .where(
                    and_(
                        Frame.video_id == video_id,
                        Frame.frame_idx >= lo,
                        Frame.frame_idx <= hi,
                    )
                )
                .order_by(Frame.frame_idx)
            )
            result = await sess.execute(stmt)
            frames = result.scalars().all()
            return [self._frame_to_dict(f) for f in frames]

    def get_frames_by_objects(
        self, class_names: list[str], min_confidence: float
    ) -> list[str]:
        """Placeholder — use async version."""
        raise NotImplementedError("Use get_frames_by_objects_async")

    async def get_frames_by_objects_async(
        self, class_names: list[str], min_confidence: float = 0.3
    ) -> list[tuple[str, float]]:
        """
        Return frames that contain ALL specified object classes, sorted by confidence.

        Args:
            class_names:    List of YOLO class name strings.
            min_confidence: Minimum detection confidence.

        Returns:
            List of tuples (frame_id, total_confidence).
        """
        async with self.session() as sess:
            if not class_names:
                return []

            stmt = (
                select(Detection.frame_id, func.avg(Detection.confidence).label("avg_conf"))
                .where(
                    and_(
                        Detection.class_name.in_(class_names),
                        Detection.confidence >= min_confidence,
                    )
                )
                .group_by(Detection.frame_id)
                .having(func.count(func.distinct(Detection.class_name)) == len(class_names))
                .order_by(desc("avg_conf"))
            )
            result = await sess.execute(stmt)
            return [(r[0], float(r[1])) for r in result.fetchall()]

    # ── Detection operations ─────────────────────────────────────────────

    async def upsert_detections_bulk(
        self, detections_data: list[dict[str, Any]]
    ) -> None:
        """
        Bulk insert detection records.

        Args:
            detections_data: List of dicts matching Detection columns.
        """
        async with self.session() as sess:
            sess.add_all([Detection(**d) for d in detections_data])
        log.debug(f"Bulk inserted {len(detections_data)} detections")

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _frame_to_dict(frame: Frame) -> dict[str, Any]:
        """Convert ORM Frame to plain dict."""
        path_str = frame.frame_path
        if path_str:
            # Safely resolve Kaggle or absolute paths to local relative paths
            if "data/frames/" in path_str.replace("\\", "/"):
                idx = path_str.replace("\\", "/").find("data/frames/")
                path_str = path_str[idx:]  # e.g., 'data/frames/...'
                
        return {
            "frame_id": frame.frame_id,
            "video_id": frame.video_id,
            "frame_idx": frame.frame_idx,
            "timestamp": frame.timestamp,
            "frame_path": path_str,
            "milvus_clip_id": frame.milvus_clip_id,
            "milvus_ocr_id": frame.milvus_ocr_id,
            "ocr_text": frame.ocr_text,
            "caption": frame.caption,
        }
