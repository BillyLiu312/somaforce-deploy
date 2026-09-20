"""Crash-recoverable chunked recording with atomic NPZ finalization."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "somaforce_chunked_recording_v1"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_savez(path: Path, arrays: Mapping[str, Any], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as file:
        if compressed:
            np.savez_compressed(file, **arrays)
        else:
            np.savez(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completed_frame_count(
    records: Mapping[str, Sequence[Any]], field_names: Sequence[str]
) -> int:
    if not field_names:
        return 0
    return min(len(records[name]) for name in field_names)


class ChunkedNpzRecorder:
    """Write complete policy rows into atomic, independently readable chunks."""

    def __init__(
        self,
        output: str | Path,
        field_names: Sequence[str],
        *,
        base_metadata: Mapping[str, Any] | None = None,
        chunk_size: int = 25,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.output = Path(output)
        self.directory = self.output.with_suffix(".recording")
        self.manifest_path = self.directory / "manifest.json"
        self.field_names = tuple(field_names)
        if len(set(self.field_names)) != len(self.field_names):
            raise ValueError("recording field names must be unique")
        self.chunk_size = int(chunk_size)
        self.base_metadata = dict(base_metadata or {})
        self._scheduled_frames = 0
        self._chunks: list[dict[str, Any]] = []
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=2)
        self._error: BaseException | None = None
        self._closed = False
        if self.directory.exists():
            raise FileExistsError(f"recording directory already exists: {self.directory}")
        self.directory.mkdir(parents=True)
        self._write_manifest(status="recording")
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="suitcase-record-writer",
            # atexit closes and joins this thread; non-daemon would be joined by
            # Python before atexit and deadlock while waiting for our sentinel.
            daemon=True,
        )
        self._thread.start()

    def _write_manifest(self, *, status: str, reason: str | None = None) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "schema": SCHEMA,
                "status": status,
                "reason": reason,
                "output": str(self.output.resolve()),
                "field_names": list(self.field_names),
                "chunk_size": self.chunk_size,
                "committed_frames": sum(item["frames"] for item in self._chunks),
                "chunks": self._chunks,
                "base_metadata": self.base_metadata,
            },
        )

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    start, end, arrays = item
                    name = f"chunk_{start:06d}_{end - 1:06d}.npz"
                    path = self.directory / name
                    atomic_savez(path, arrays, compressed=False)
                    self._chunks.append(
                        {
                            "file": name,
                            "start": start,
                            "end": end,
                            "frames": end - start,
                            "size_bytes": path.stat().st_size,
                            "sha256": _sha256(path),
                        }
                    )
                    self._write_manifest(status="recording")
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._error = exc
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()

    def _raise_writer_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("recording writer failed") from self._error

    def capture(self, records: Mapping[str, Sequence[Any]], *, force: bool = False) -> None:
        if self._closed:
            return
        self._raise_writer_error()
        complete = _completed_frame_count(records, self.field_names)
        while complete - self._scheduled_frames >= self.chunk_size or (
            force and complete > self._scheduled_frames
        ):
            end = (
                complete
                if force
                else min(complete, self._scheduled_frames + self.chunk_size)
            )
            start = self._scheduled_frames
            arrays = {
                name: np.asarray(records[name][start:end])
                for name in self.field_names
            }
            self._queue.put((start, end, arrays))
            self._scheduled_frames = end

    def close(
        self,
        records: Mapping[str, Sequence[Any]],
        *,
        status: str,
        reason: str | None = None,
    ) -> int:
        if self._closed:
            return self._scheduled_frames
        self.capture(records, force=True)
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._raise_writer_error()
        self._closed = True
        committed = sum(item["frames"] for item in self._chunks)
        if committed != self._scheduled_frames:
            raise RuntimeError(
                f"recording committed {committed} of {self._scheduled_frames} scheduled frames"
            )
        self._write_manifest(status=status, reason=reason)
        return committed


def finalize_chunked_recording(
    output: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    complete: bool,
    reason: str | None = None,
) -> int:
    output_path = Path(output)
    directory = output_path.with_suffix(".recording")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported recording manifest: {manifest.get('schema')!r}")
    field_names = tuple(manifest["field_names"])
    chunks = sorted(manifest["chunks"], key=lambda item: int(item["start"]))
    arrays_by_name: dict[str, list[np.ndarray]] = {name: [] for name in field_names}
    expected_start = 0
    for chunk in chunks:
        if int(chunk["start"]) != expected_start:
            raise ValueError(
                f"recording chunk gap: expected {expected_start}, got {chunk['start']}"
            )
        path = directory / chunk["file"]
        if _sha256(path) != chunk["sha256"]:
            raise ValueError(f"recording chunk hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            for name in field_names:
                arrays_by_name[name].append(np.asarray(data[name]).copy())
        expected_start = int(chunk["end"])
    arrays = {
        name: (
            np.concatenate(values, axis=0)
            if values
            else np.empty((0,), dtype=np.float32)
        )
        for name, values in arrays_by_name.items()
    }
    final_metadata = dict(manifest.get("base_metadata", {}))
    final_metadata.update(dict(metadata or {}))
    final_metadata.update(
        {
            "complete": bool(complete),
            "termination_reason": reason,
            "steps": expected_start,
            "recording_manifest": str(manifest_path.resolve()),
        }
    )
    arrays["metadata"] = np.asarray(json.dumps(final_metadata, sort_keys=True))
    atomic_savez(output_path, arrays, compressed=True)
    with np.load(output_path, allow_pickle=False) as verification:
        if int(verification[field_names[0]].shape[0]) != expected_start:
            raise RuntimeError("finalized recording frame count verification failed")
        if any(name not in verification for name in (*field_names, "metadata")):
            raise RuntimeError("finalized recording is missing fields")
    manifest["status"] = "finalized" if complete else "partial_finalized"
    manifest["reason"] = reason
    manifest["finalized_frames"] = expected_start
    manifest["finalized_output_sha256"] = _sha256(output_path)
    atomic_write_json(manifest_path, manifest)
    return expected_start
