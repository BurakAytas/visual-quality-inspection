"""Put one dataset category in place and check its layout.

Examples:
    # MVTec AD 2: download it yourself from the MVTec website, then point here
    python scripts/download_data.py --config configs/fabric.yaml --source /path/to/fabric.tar.xz

    # MVTec AD: download from the official MVTec host (needs the [train] extra)
    python scripts/download_data.py --config configs/leather.yaml
"""

import argparse
import sys
from pathlib import Path

from vqi.config import load_config
from vqi.dataset_files import download_mvtec_ad, install_from_local, missing_dirs


def main() -> int:
    """Install the dataset named in the config and report missing folders."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--source", type=Path, help="Downloaded folder or .tar(.xz/.gz) archive (required for AD 2)"
    )
    args = parser.parse_args()

    dataset = load_config(args.config).dataset
    if dataset.category_dir.exists():
        print(f"Found {dataset.category_dir}, checking layout only.")
    elif args.source is not None:
        install_from_local(args.source, dataset)
    elif dataset.name == "mvtec_ad":
        download_mvtec_ad(dataset)
    else:
        print(
            "MVTec AD 2 needs a manual download (web form on the MVTec website). "
            "Pass the downloaded file with --source.",
            file=sys.stderr,
        )
        return 2

    missing = missing_dirs(dataset.category_dir, dataset.name)
    if missing:
        print(f"Layout check FAILED in {dataset.category_dir}: missing {missing}", file=sys.stderr)
        return 1
    print(f"Layout OK: {dataset.category_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
