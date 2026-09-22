#!/usr/bin/env python3
"""Finalize any crash-recoverable policy recording from committed chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from somaforce_deploy.chunked_recording import (
    atomic_write_json,
    finalize_chunked_recording,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reason", default="runner_exit")
    args = parser.parse_args()

    manifest_path = args.output.with_suffix(".recording") / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    base_metadata = dict(manifest.get("base_metadata", {}))
    schema = str(base_metadata.get("schema", "somaforce_policy_hardware_record"))
    frames = finalize_chunked_recording(
        args.output,
        metadata={"schema": schema},
        complete=False,
        reason=args.reason,
    )
    report = {
        "schema": schema,
        "complete": False,
        "termination_reason": args.reason,
        "steps": frames,
        "output": str(args.output.resolve()),
        "recording_directory": str(args.output.with_suffix(".recording").resolve()),
    }
    atomic_write_json(args.output.with_suffix(".partial.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
