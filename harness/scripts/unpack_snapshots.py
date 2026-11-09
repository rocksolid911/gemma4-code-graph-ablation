"""Extract only snapshots/*.tgz from the full competition zip into data/competition/snapshots/.

    python -I harness/scripts/unpack_snapshots.py <path-to-gemma-4-developer-agent.zip>

Skips files already present with the right size, and rejects any member path that would escape the target.
"""

import sys
import zipfile
from pathlib import Path

DEST = Path("data/competition")


def main(zip_path: str) -> None:
    root = DEST.resolve()
    done = skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if m.filename.startswith("snapshots/") and m.filename.endswith(".tgz")]
        print(f"{len(members)} snapshots, {sum(m.file_size for m in members) / 1e9:.1f} GB")
        for m in members:
            out = (root / m.filename).resolve()
            if root not in out.parents:
                raise ValueError(f"unsafe path in zip: {m.filename}")
            if out.exists() and out.stat().st_size == m.file_size:
                skipped += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            done += 1
            print(f"[{done + skipped}/{len(members)}] {m.filename}", flush=True)
    print(f"extracted {done}, already present {skipped}")


if __name__ == "__main__":
    main(sys.argv[1])
