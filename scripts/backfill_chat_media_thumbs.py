"""Backfill bubble thumbnails for chat media uploaded before thumbnails existed.

Idempotent: skips files that already have a `{stem}_t.jpg` sibling. Original
files are left byte-identical — only missing thumbnails are generated, so the
script is safe to re-run at any time.

Run inside the server container (CHAT_MEDIA_DIR must point at the data disk):

    docker exec companion-server python scripts/backfill_chat_media_thumbs.py

Add --dry-run to only report what would be generated.
"""

from __future__ import annotations

import argparse
import sys

from app.services.chat_media import storage

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only")
    args = parser.parse_args()

    media_dir = storage._MEDIA_DIR
    if not media_dir.exists():
        print(f"media dir {media_dir} does not exist; nothing to do")
        return 0

    generated = skipped = failed = 0
    for path in sorted(media_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if path.name.endswith(storage._THUMB_KEY_SUFFIX):
            continue
        thumb_path = storage.storage_path(storage.thumb_storage_key(path.name))
        if thumb_path.exists():
            skipped += 1
            continue
        if args.dry_run:
            print(f"would generate {thumb_path.name} <- {path.name}")
            generated += 1
            continue
        thumb = storage.generate_thumbnail_blob(path.read_bytes())
        if thumb is None:
            print(f"FAILED to decode {path.name}", file=sys.stderr)
            failed += 1
            continue
        thumb_path.write_bytes(thumb)
        print(
            f"generated {thumb_path.name} "
            f"({path.stat().st_size // 1024}KB -> {len(thumb) // 1024}KB)"
        )
        generated += 1

    print(
        f"done: generated={generated} already-had-thumb={skipped} failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
