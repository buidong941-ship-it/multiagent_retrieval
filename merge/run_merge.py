"""Merge partitioned retrieval artifacts without losing colliding SQLite IDs.

Each first-level directory below ``merge/input`` is treated as one source part
(for example ``head`` or ``tail``).  Logical video/frame IDs are namespaced by
that part.  This is deliberately different from the old ``INSERT OR IGNORE``
approach: every input row gets a new SQLite integer PK, while every reference
to its logical ID is remapped consistently.
"""

import gc
import pickle
import re
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from sqlalchemy import create_engine

# Add project root to PYTHONPATH so we can import from video_retrieval
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "video_retrieval"))

from config.settings import get_settings
from database.bm25.bm25_index import BM25OcrIndex
from database.faiss.faiss_client import FaissVectorDatabase
from database.metadata.schema import Base


VIDEO_COLUMNS = (
    "video_id", "video_path", "duration", "fps", "width", "height",
    "total_frames", "indexed_at",
)
FRAME_COLUMNS = (
    "frame_id", "video_id", "frame_idx", "timestamp", "frame_path",
    "milvus_clip_id", "milvus_beit3_id", "milvus_ocr_id", "ocr_results",
    "ocr_text", "caption",
)
DETECTION_COLUMNS = (
    "frame_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2",
)
FAISS_BATCH_SIZE = 4_096
_missing_frame_files: Counter[str] = Counter()


def init_master_metadata(db_path: Path) -> sqlite3.Connection:
    """Create a clean master DB and enable referential-integrity checks."""
    Base.metadata.create_all(create_engine(f"sqlite:///{db_path}"))
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return tag or "part"


def _source_key(input_dir: Path, artifact: Path) -> str:
    """Return the first input-directory component owning an artifact."""
    relative = artifact.resolve().relative_to(input_dir.resolve())
    return relative.parts[0] if len(relative.parts) > 1 else "root"


def _build_source_tags(input_dir: Path, db_files: list[Path]) -> dict[str, str]:
    """Assign stable, unique namespace tags to source parts."""
    keys = sorted({_source_key(input_dir, db_file) for db_file in db_files})
    used: Counter[str] = Counter()
    tags: dict[str, str] = {}
    for key in keys:
        base = _safe_tag(key)
        used[base] += 1
        tags[key] = base if used[base] == 1 else f"{base}_{used[base]}"
    return tags


def _namespaced_id(source_tag: str, original_id: str) -> str:
    return f"{source_tag}__{original_id}"


def _source_root(input_dir: Path, artifact: Path) -> Path:
    key = _source_key(input_dir, artifact)
    return input_dir if key == "root" else input_dir / key


def _locate_frame(source_db: Path, source_root: Path, frame_path: str) -> Path | None:
    """Find a frame even when an exported DB contains an old absolute path."""
    raw = Path(frame_path)
    candidates = [raw, source_db.parent / raw]
    normal = frame_path.replace("\\", "/")
    if "data/frames/" in normal:
        suffix = normal.split("data/frames/", 1)[1]
        candidates.extend((source_root / "data" / "frames" / suffix, source_root / "frames" / suffix))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _stage_frame(
    source_db: Path, source_root: Path, output_dir: Path, source_tag: str, frame_path: str
) -> str:
    """Copy available images into a collision-free output tree and return its path.

    If the source image is not available locally, retain the original path and
    report it; metadata is still preserved rather than silently dropping a row.
    """
    source = _locate_frame(source_db, source_root, frame_path)
    if source is None:
        _missing_frame_files[source_tag] += 1
        return frame_path

    try:
        relative = source.relative_to(source_root / "data" / "frames")
    except ValueError:
        relative = Path(source.name)
    destination = output_dir / "frames" / source_tag / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    return str(destination.resolve())


def _select_rows(connection: sqlite3.Connection, alias: str, table: str, columns: tuple[str, ...]):
    column_sql = ", ".join(columns)
    return connection.execute(f"SELECT {column_sql} FROM {alias}.{table}")


def merge_sqlite(
    input_dir: Path, output_dir: Path, db_files: list[Path], source_tags: dict[str, str]
) -> tuple[sqlite3.Connection, set[str]]:
    """Merge DB rows with fresh surrogate IDs and remapped logical IDs."""
    master_path = output_dir / "metadata.db"
    if master_path.exists():
        master_path.unlink()
    master = init_master_metadata(master_path)
    known_frame_ids: set[str] = set()

    for number, db_file in enumerate(db_files):
        key = _source_key(input_dir, db_file)
        tag = source_tags[key]
        root = _source_root(input_dir, db_file)
        alias = f"part_db_{number}"
        print(f"  - Merging {db_file} as namespace '{tag}'...")
        master.execute(f"ATTACH DATABASE ? AS {alias}", (str(db_file),))
        try:
            # Do not select source `id`: SQLite allocates a fresh PK in master.
            video_rows = []
            for row in _select_rows(master, alias, "videos", VIDEO_COLUMNS):
                values = dict(zip(VIDEO_COLUMNS, row))
                values["video_id"] = _namespaced_id(tag, values["video_id"])
                video_rows.append(tuple(values[column] for column in VIDEO_COLUMNS))
            master.executemany(
                f"INSERT INTO videos ({', '.join(VIDEO_COLUMNS)}) VALUES ({', '.join('?' for _ in VIDEO_COLUMNS)})",
                video_rows,
            )

            frame_rows = []
            for row in _select_rows(master, alias, "frames", FRAME_COLUMNS):
                values = dict(zip(FRAME_COLUMNS, row))
                values["frame_id"] = _namespaced_id(tag, values["frame_id"])
                values["video_id"] = _namespaced_id(tag, values["video_id"])
                values["frame_path"] = _stage_frame(db_file, root, output_dir, tag, values["frame_path"])
                frame_rows.append(tuple(values[column] for column in FRAME_COLUMNS))
                known_frame_ids.add(values["frame_id"])
            master.executemany(
                f"INSERT INTO frames ({', '.join(FRAME_COLUMNS)}) VALUES ({', '.join('?' for _ in FRAME_COLUMNS)})",
                frame_rows,
            )

            detection_rows = []
            for row in _select_rows(master, alias, "detections", DETECTION_COLUMNS):
                values = dict(zip(DETECTION_COLUMNS, row))
                values["frame_id"] = _namespaced_id(tag, values["frame_id"])
                detection_rows.append(tuple(values[column] for column in DETECTION_COLUMNS))
            master.executemany(
                f"INSERT INTO detections ({', '.join(DETECTION_COLUMNS)}) VALUES ({', '.join('?' for _ in DETECTION_COLUMNS)})",
                detection_rows,
            )
            master.commit()
        finally:
            master.execute(f"DETACH DATABASE {alias}")

    return master, known_frame_ids


def _source_vector_records(pkl_file: Path):
    """Yield ``(old_id, vector, metadata)`` without requiring vectors in output PKLs.

    Older source PKLs contain a ``vectors`` dictionary.  New merged artifacts
    intentionally persist only metadata in their PKL and retain vectors in the
    paired FAISS file; this fallback makes the new format mergeable as well.
    """
    with pkl_file.open("rb") as stream:
        payload = pickle.load(stream)

    metadata_by_id = payload.get("metadata", {})
    vectors_by_id = payload.get("vectors", {})
    if vectors_by_id:
        for old_id, vector in vectors_by_id.items():
            yield old_id, vector, metadata_by_id.get(old_id, {})
        return

    faiss_file = pkl_file.with_suffix(".faiss")
    if not metadata_by_id:
        return
    if not faiss_file.exists():
        raise FileNotFoundError(
            f"{pkl_file} has no vectors and its paired FAISS index is missing: {faiss_file}"
        )
    source_index = faiss.read_index(str(faiss_file))
    try:
        for old_id, item in metadata_by_id.items():
            yield old_id, source_index.reconstruct(int(old_id)), item
    finally:
        del source_index


def _release_faiss_collection(database: FaissVectorDatabase, collection: str) -> None:
    """Release RAM for a flushed collection before starting the next model."""
    database.indices.pop(collection, None)
    database.metadata_store.pop(collection, None)
    database.frame_id_to_id.pop(collection, None)
    database.vectors_store.pop(collection, None)
    gc.collect()


def merge_faiss(input_dir: Path, output_dir: Path, source_tags: dict[str, str], known_frame_ids: set[str]) -> None:
    """Merge vectors in bounded batches with the same ID remapping as SQLite."""
    settings = get_settings()
    settings.faiss.index_dir = str(output_dir / "faiss")
    faiss_output = Path(settings.faiss.index_dir)
    # A merge is a fresh snapshot.  Do not leave collections from an older run.
    if faiss_output.exists():
        for old_file in list(faiss_output.glob("*.faiss")) + list(faiss_output.glob("*.pkl")):
            old_file.unlink()
    collections: dict[str, list[Path]] = defaultdict(list)
    for pkl_file in input_dir.rglob("*.pkl"):
        if "embeddings" in pkl_file.stem:
            collections[pkl_file.stem].append(pkl_file)

    database = FaissVectorDatabase(settings.faiss)
    for collection, files in collections.items():
        seen_ids: set[str] = set()
        next_id = 0
        print(f"  - Merging FAISS '{collection}' from {len(files)} part(s)...")
        for pkl_file in sorted(files):
            tag = source_tags.get(_source_key(input_dir, pkl_file), _safe_tag(_source_key(input_dir, pkl_file)))
            batch_vectors: list[np.ndarray] = []
            batch_metadata: list[dict[str, Any]] = []
            for old_id, vector, source_metadata in _source_vector_records(pkl_file):
                item = dict(source_metadata)
                original_frame_id = item.get("frame_id")
                if not original_frame_id:
                    print(f"    ! Skipping vector {old_id}: metadata has no frame_id")
                    continue
                item["frame_id"] = _namespaced_id(tag, original_frame_id)
                if item.get("video_id"):
                    item["video_id"] = _namespaced_id(tag, item["video_id"])
                if item["frame_id"] not in known_frame_ids:
                    print(f"    ! Skipping dangling vector for {item['frame_id']}")
                    continue
                if item["frame_id"] in seen_ids:
                    print(f"    ! Skipping duplicate vector for {item['frame_id']} in {collection}")
                    continue
                seen_ids.add(item["frame_id"])
                batch_vectors.append(np.asarray(vector, dtype=np.float32))
                batch_metadata.append(item)

                if len(batch_vectors) == FAISS_BATCH_SIZE:
                    ids = list(range(next_id, next_id + len(batch_vectors)))
                    database.insert(collection, ids, np.vstack(batch_vectors), batch_metadata)
                    # Retrieval reconstructs from FAISS; retaining these copies is
                    # unnecessary and is the main source of peak-memory pressure.
                    database.vectors_store[collection].clear()
                    next_id += len(batch_vectors)
                    batch_vectors.clear()
                    batch_metadata.clear()

            if batch_vectors:
                ids = list(range(next_id, next_id + len(batch_vectors)))
                database.insert(collection, ids, np.vstack(batch_vectors), batch_metadata)
                database.vectors_store[collection].clear()
                next_id += len(batch_vectors)
            del batch_vectors, batch_metadata
            gc.collect()

        if next_id:
            # The persisted PKL deliberately contains metadata/mapping only.
            # The paired .faiss file remains the single source of vector data.
            database.flush(collection)
            print(f"    -> saved {next_id} unique, metadata-backed vectors (batch={FAISS_BATCH_SIZE})")
            _release_faiss_collection(database, collection)


def rebuild_bm25(master: sqlite3.Connection, output_dir: Path) -> None:
    settings = get_settings()
    settings.ocr.bm25_index_path = str(output_dir / "bm25_ocr.pkl")
    rows = master.execute(
        "SELECT frame_id, ocr_text FROM frames WHERE ocr_text IS NOT NULL AND ocr_text != ''"
    ).fetchall()
    if not rows:
        print("No OCR text found; skipping BM25 rebuild.")
        return
    bm25 = BM25OcrIndex(settings.ocr)
    bm25.build([row[1] for row in rows], [row[0] for row in rows])
    bm25.save()
    print(f"  - Rebuilt BM25 with {len(rows)} metadata-backed OCR documents")


def run_merge() -> None:
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    _missing_frame_files.clear()
    if not input_dir.exists():
        print(f"Error: input directory not found at {input_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    db_files = sorted(input_dir.rglob("*.db"))
    if not db_files:
        print("No source metadata.db files found; aborting so vectors cannot become orphaned.")
        return

    source_tags = _build_source_tags(input_dir, db_files)
    print("Source namespaces:", source_tags)
    print("Merging SQLite metadata with fresh primary keys...")
    master, known_frame_ids = merge_sqlite(input_dir, output_dir, db_files, source_tags)
    try:
        db_count = master.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        assert db_count == len(known_frame_ids), "duplicate remapped frame IDs were produced"
        print(f"  -> master metadata has {db_count} frames")
        print("Merging FAISS vectors with remapped frame IDs...")
        merge_faiss(input_dir, output_dir, source_tags, known_frame_ids)
        print("Rebuilding BM25 from master metadata...")
        rebuild_bm25(master, output_dir)
    finally:
        master.close()
    if _missing_frame_files:
        summary = ", ".join(f"{tag}: {count}" for tag, count in sorted(_missing_frame_files.items()))
        print(f"Warning: {sum(_missing_frame_files.values())} source image(s) were not found locally ({summary}). Their original frame_path values were retained.")
    print(f"Merge complete: {output_dir}")


if __name__ == "__main__":
    run_merge()
