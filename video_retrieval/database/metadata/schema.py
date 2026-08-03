"""
SQLAlchemy ORM schema for the metadata database.

Design Decision:
    - SQLite for metadata storage during competition (zero-ops, single file).
    - PostgreSQL-compatible via SQLAlchemy ORM (easy migration if needed).
    - Three tables:
        1. videos        — video-level metadata
        2. frames        — frame-level metadata (FK → videos)
        3. detections    — YOLO objects per frame (FK → frames)
    - OCR text stored as JSON in frames.ocr_results.
    - Indexes on (video_id, frame_idx) for fast temporal queries.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Video(Base):
    """
    Stores video-level metadata.

    Columns:
        id          — auto-increment PK
        video_id    — unique string identifier (filename stem)
        video_path  — absolute path to video file
        duration    — video duration in seconds
        fps         — original video FPS
        width       — video width in pixels
        height      — video height in pixels
        total_frames — total number of frames in video
        indexed_at  — timestamp when indexing completed
    """

    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    video_path: Mapped[str] = mapped_column(Text, nullable=False)
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    frames: Mapped[list["Frame"]] = relationship(
        "Frame", back_populates="video", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Video id={self.id} video_id={self.video_id!r}>"


class Frame(Base):
    """
    Stores frame-level metadata.

    Columns:
        id              — auto-increment PK
        frame_id        — globally unique string: "{video_id}_frame_{frame_idx:06d}"
        video_id        — FK → videos.video_id
        frame_idx       — 0-based frame index within video
        timestamp       — time offset in seconds
        frame_path      — path to saved JPEG/PNG
        milvus_clip_id  — corresponding ID in Milvus clip_embeddings collection
        milvus_ocr_id   — corresponding ID in Milvus ocr_embeddings collection (nullable)
        ocr_results     — JSON: list of {text, confidence, bbox}
        ocr_text        — concatenated OCR text (for BM25 search)
        caption         — optional image caption
    """

    __tablename__ = "frames"
    __table_args__ = (
        UniqueConstraint("video_id", "frame_idx", name="uq_video_frame"),
        Index("ix_frames_video_timestamp", "video_id", "timestamp"),
        Index("ix_frames_frame_id", "frame_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    frame_id: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    video_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("videos.video_id", ondelete="CASCADE"), nullable=False
    )
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    frame_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Milvus IDs (int64 primary keys in Milvus collections)
    milvus_clip_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    milvus_beit3_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    milvus_ocr_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # OCR data
    ocr_results: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    ocr_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Enrichment
    caption: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    video: Mapped["Video"] = relationship("Video", back_populates="frames")
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="frame", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Frame frame_id={self.frame_id!r} ts={self.timestamp:.2f}s>"


class Detection(Base):
    """
    Stores YOLO detection results per frame.

    One row per detected object instance.

    Columns:
        id          — auto-increment PK
        frame_id    — FK → frames.frame_id
        class_id    — COCO class ID
        class_name  — COCO class label string
        confidence  — detection confidence [0, 1]
        x1, y1      — top-left bounding box (normalized [0, 1])
        x2, y2      — bottom-right bounding box (normalized [0, 1])
    """

    __tablename__ = "detections"
    __table_args__ = (
        Index("ix_detections_frame_class", "frame_id", "class_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    frame_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("frames.frame_id", ondelete="CASCADE"), nullable=False
    )
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationship
    frame: Mapped["Frame"] = relationship("Frame", back_populates="detections")

    def __repr__(self) -> str:
        return f"<Detection frame={self.frame_id!r} class={self.class_name!r} conf={self.confidence:.2f}>"
