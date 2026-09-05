#!/usr/bin/env python3
"""Verify a task artifact manifest without starting policy, MuJoCo, or hardware."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from somaforce_deploy.artifacts import ArtifactManifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    ArtifactManifest.load(args.manifest).verify()
    print(f"artifact manifest: PASS ({args.manifest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
