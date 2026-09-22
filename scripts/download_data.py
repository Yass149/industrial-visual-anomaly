#!/usr/bin/env python3
"""Fetch and verify the MVTec AD dataset.

The official distribution sits behind a licence-acceptance form on mvtec.com with no stable
direct URL, so this pulls a mirror of the same archive and verifies it against a SHA-256
digest committed in the config. That digest is self-generated, not published by MVTec: it
guarantees everyone gets byte-identical data to what the reported numbers came from, and
nothing more. MVTec AD is licensed for non-commercial research use only.

Usage:
    python scripts/download_data.py [--config configs/default.yaml] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

# Make `import anomaly` work when this is run as a script rather than via the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly.config import Config, load_config  # noqa: E402

CHUNK = 1 << 20  # 1 MiB: large enough that per-chunk overhead is irrelevant on a 5GB file

# Every MVTec category tree has these, so their presence is a cheap structural check that the
# archive extracted into the shape the rest of the pipeline assumes.
REQUIRED_SUBDIRS = ("train/good", "test", "ground_truth")


def sha256_file(path: Path, desc: str = "hashing") -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    total = path.stat().st_size
    with (
        path.open("rb") as fh,
        tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as bar,
    ):
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download `url` to `dest`, resuming from a partial file if one is present.

    A 5GB download over a flaky connection will fail eventually; resuming from the byte
    offset already on disk is the difference between a retry and starting over.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if existing and response.status != 206:
            # Server ignored the range request, so the bytes we would append are the start of
            # the file again. Restart cleanly rather than corrupting the archive.
            print(f"server does not support resume (HTTP {response.status}); restarting download")
            existing = 0
            mode = "wb"
        else:
            mode = "ab" if existing else "wb"

        remaining = int(response.headers.get("Content-Length", 0))
        total = existing + remaining
        with (
            dest.open(mode) as fh,
            tqdm(total=total, initial=existing, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            while chunk := response.read(CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))


def extract(archive: Path, target: Path) -> None:
    """Extract the archive into `target`, stripping its single top-level directory.

    The mirror wraps everything in one folder whose name is a typo of the dataset's; stripping
    it keeps paths in the config short and independent of whichever mirror supplied the bytes.
    """
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        roots = {Path(m.filename).parts[0] for m in members}
        if len(roots) != 1:
            raise RuntimeError(f"expected a single top-level directory in the archive, got {roots}")
        strip = roots.pop()

        for member in tqdm(members, desc="extracting", unit="file"):
            relative = Path(member.filename).relative_to(strip)
            if not relative.parts:
                continue
            out = target / relative
            # Reject entries that would escape the target directory (zip-slip).
            if not out.resolve().is_relative_to(target.resolve()):
                raise RuntimeError(f"archive entry escapes target directory: {member.filename}")
            if member.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)


def verify_layout(extract_dir: Path, categories: list[str]) -> dict[str, int]:
    """Check every configured category is present and well-formed; return train image counts."""
    counts: dict[str, int] = {}
    for category in categories:
        root = extract_dir / category
        if not root.is_dir():
            raise RuntimeError(f"category {category!r} missing from {extract_dir}")
        for subdir in REQUIRED_SUBDIRS:
            if not (root / subdir).is_dir():
                raise RuntimeError(f"{category}: expected {subdir}/ under {root}")
        n_train = len(list((root / "train" / "good").glob("*.png")))
        if n_train == 0:
            raise RuntimeError(f"{category}: no training images in train/good")
        counts[category] = n_train
    return counts


def already_present(cfg: Config) -> bool:
    """True when every configured category is already extracted, so the work can be skipped."""
    try:
        verify_layout(cfg.data.extract_dir, cfg.data.categories)
    except RuntimeError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    archive = cfg.data.archive_path

    if not args.force and already_present(cfg):
        print(f"dataset already present at {cfg.data.extract_dir}; nothing to do (--force to redo)")
        return 0

    if args.force and archive.exists():
        archive.unlink()

    if not archive.exists():
        print(f"downloading {cfg.data.url}")
        download(cfg.data.url, archive)

    digest = sha256_file(archive, desc="verifying")
    if cfg.data.sha256 is None:
        # First run against a new mirror: report the digest so it can be pinned in the config.
        print(f"\nno checksum pinned. Archive SHA-256:\n  {digest}\nSet data.sha256 in the config.")
    elif digest != cfg.data.sha256:
        print(
            f"checksum mismatch for {archive}\n  expected {cfg.data.sha256}\n  got      {digest}\n"
            "The mirror may have changed. Delete the archive and retry, or verify the source.",
            file=sys.stderr,
        )
        return 1
    else:
        print(f"checksum ok ({digest[:16]}...)")

    print(f"extracting to {cfg.data.extract_dir}")
    extract(archive, cfg.data.extract_dir)

    counts = verify_layout(cfg.data.extract_dir, cfg.data.categories)
    for category, n in counts.items():
        print(f"  {category:<12} {n:>4} normal training images")

    if not cfg.data.keep_archive:
        archive.unlink()
        print(f"removed {archive} (data.keep_archive is false)")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
