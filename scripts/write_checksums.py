#!/usr/bin/env python3
"""Write a deterministic SHA256SUMS manifest for an experiment directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    return parser.parse_args()


def main() -> None:
    root = Path(parse_args().directory).resolve()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    destination = root / "SHA256SUMS"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != destination)
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(lines)} checksums to {destination}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
