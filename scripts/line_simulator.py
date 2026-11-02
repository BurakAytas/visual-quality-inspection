"""Simulate a production line: post every new image in a folder to the inspection API.

The folder plays the camera: copy images into it and each one is inspected once, oldest
first. A real camera writes to a temporary name and renames when done; when copying by
hand, an image caught half-written is answered with HTTP 400, printed, and skipped.

Example (API running on port 8000):
    python scripts/line_simulator.py --folder incoming
"""

import argparse
import time
from pathlib import Path

import httpx

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def new_images(folder: Path, seen: set[Path]) -> list[Path]:
    """Images in ``folder`` not posted yet, oldest first (modification time, then name)."""
    images = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p not in seen]
    return sorted(images, key=lambda p: (p.stat().st_mtime, p.name))


def post_new_images(client: httpx.Client, url: str, folder: Path, seen: set[Path]) -> int:
    """Post each new image once and print the decision; returns how many were posted."""
    images = new_images(folder, seen)
    for path in images:
        seen.add(path)
        response = client.post(f"{url}/inspect", files={"file": (path.name, path.read_bytes())})
        if response.status_code != 200:
            print(f"{path.name}: HTTP {response.status_code} {response.text}")
            continue
        result = response.json()
        review = "  needs review" if result["needs_review"] else ""
        print(
            f"{path.name}: {result['decision']} score {result['score']:.3f} "
            f"(threshold {result['threshold']:.3f}) {result['latency_ms']:.0f} ms{review}"
        )
    return len(images)


def main() -> None:
    """Parse arguments and poll the folder until stopped (or once with --once)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between polls")
    parser.add_argument("--once", action="store_true", help="post what is there, then exit")
    args = parser.parse_args()

    seen: set[Path] = set()
    with httpx.Client(timeout=60) as client:
        while True:
            post_new_images(client, args.url.rstrip("/"), args.folder, seen)
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
