"""Create SHA-256 inventory for the packaged nozzle PINN results."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def main() -> None:
    package = Path(__file__).resolve().parents[1]
    manifest = package / "MANIFEST.csv"
    rows = []
    for path in sorted(package.rglob("*")):
        if (not path.is_file() or path == manifest or
                "__pycache__" in path.parts or path.suffix == ".pyc"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        rows.append((path.relative_to(package).as_posix(), path.stat().st_size,
                     digest.hexdigest()))
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "bytes", "sha256"))
        writer.writerows(rows)
    print(f"Indexed {len(rows)} files in {manifest}")


if __name__ == "__main__":
    main()
