"""Artifact manifest loading and SHA-256 verification for deployment bundles."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactEntry:
    name: str
    role: str
    sha256: str
    required: bool = True


@dataclass(frozen=True)
class ArtifactManifest:
    root: Path
    entries: tuple[ArtifactEntry, ...]
    schema_version: int = 1

    @classmethod
    def load(cls, path: str | Path) -> "ArtifactManifest":
        manifest_path = Path(path).expanduser().resolve()
        payload: dict[str, Any] = json.loads(manifest_path.read_text())
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported artifact manifest schema_version")
        raw_entries = payload.get("artifacts")
        if not isinstance(raw_entries, list):
            raise ValueError("manifest.artifacts must be a list")
        entries = tuple(
            ArtifactEntry(
                name=str(item["name"]),
                role=str(item["role"]),
                sha256=str(item["sha256"]).lower(),
                required=bool(item.get("required", True)),
            )
            for item in raw_entries
        )
        return cls(root=manifest_path.parent, entries=entries)

    def verify(self) -> None:
        for entry in self.entries:
            candidate = (self.root / entry.name).resolve()
            try:
                candidate.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(f"artifact escapes manifest root: {entry.name}") from exc
            if not candidate.exists():
                if entry.required:
                    raise FileNotFoundError(candidate)
                continue
            digest = sha256_file(candidate)
            if digest != entry.sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {entry.name}: expected {entry.sha256}, got {digest}"
                )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
