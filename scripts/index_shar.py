"""Make an indexed copy of plain Shar directories, so a frozen set can read their audio by position.

    python scripts/index_shar.py SRC_ROOT DST_ROOT REL_DIR [REL_DIR ...]

    # e.g. VoxPopuli test from the azeros tree, on JUPITER (CPU only)
    python scripts/index_shar.py $AZEROS_DIR $PROJECT/$USER/melt-data/azeros-indexed \\
        voxpopuli/{en,de,fr,es,pl,it,ro,hu,cs,nl,fi,hr,sk,sl,et,lt}/test

melteval's Shar reader needs random access (``readers/shar.py``), which lhotse
only gives over uncompressed shards with ``.idx`` sidecars; ``cuts.*.jsonl.gz``
has neither. For each ``SRC_ROOT/REL_DIR`` this writes ``DST_ROOT/REL_DIR``:

* ``*.jsonl.gz`` decompressed to ``*.jsonl`` (manifests only, small);
* everything else symlinked, not copied (the audio tars stay where they are);
* ``.idx`` sidecars built with lhotse's ``create_shar_index``.

Audio must be plain ``*.tar`` (``*.tar.gz`` is an error). Safe to rerun.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path


def index_dir(src: Path, dst: Path) -> int:
    """Build the indexed copy of one Shar directory; return its number of cuts."""
    from lhotse import CutSet # type: ignore
    from lhotse.indexing import create_shar_index # type: ignore

    if not src.is_dir():
        raise SystemExit(f"{src} is not a directory.")
    compressed_tars = sorted(p.name for p in src.iterdir() if p.name.endswith((".tar.gz", ".tgz")))
    if compressed_tars:
        raise SystemExit(f"{src}: compressed audio shards cannot be indexed: {', '.join(compressed_tars)}")

    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.iterdir()):
        if path.name.endswith(".jsonl.gz"):
            out = dst / path.name.removesuffix(".gz")
            if not out.exists():
                tmp = out.with_suffix(".jsonl.tmp")
                with gzip.open(path, "rb") as fin, tmp.open("wb") as fout:
                    shutil.copyfileobj(fin, fout)
                tmp.replace(out)
        elif not path.name.endswith(".idx"):
            link = dst / path.name
            if not link.exists():
                link.symlink_to(path.resolve())

    if not all((dst / (p.name + ".idx")).exists() for p in dst.iterdir() if p.suffix in (".jsonl", ".tar")):
        create_shar_index(dst)

    cuts = CutSet.from_shar(in_dir=str(dst), shuffle_shards=False, split_for_dataloading=False)
    access = getattr(cuts.data, "has_constant_time_access", False)
    if not (access() if callable(access) else access):
        raise SystemExit(f"{dst}: lhotse still does not see an indexed Shar directory.")
    return len(cuts)


def main(argv: list[str] | None = None) -> None:
    """Index every REL_DIR given."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_root", type=Path, help="Root of the plain Shar tree (e.g. $AZEROS_DIR).")
    parser.add_argument("dst_root", type=Path, help="Root of the indexed copy; mirrors src_root's layout.")
    parser.add_argument("rel_dirs", nargs="+", help="Shar directories, relative to both roots.")
    args = parser.parse_args(argv)

    for rel in args.rel_dirs:
        n = index_dir(args.src_root / rel, args.dst_root / rel)
        print(f"{rel}: {n} cuts, indexed at {args.dst_root / rel}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
