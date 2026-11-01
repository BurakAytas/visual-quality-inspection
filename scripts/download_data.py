"""Put one dataset category in place and check its layout.

Download the data yourself through the MVTec web forms, then point here.
--source can be the archive, the unpacked folder, or a folder holding <category>.tar*.

Examples:
    python scripts/download_data.py --config configs/fabric.yaml --source downloads/fabric.tar.gz
    python scripts/download_data.py --config configs/leather.yaml --source downloads/
"""

import argparse
import sys
from pathlib import Path

from vqi.config import load_config
from vqi.dataset_files import install_from_local, missing_dirs


def main() -> int:
    """Install the dataset named in the config and report missing folders."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--source", type=Path, help="Downloaded archive, unpacked folder, or folder of archives"
    )
    args = parser.parse_args()

    dataset = load_config(args.config).dataset
    if dataset.category_dir.exists():
        print(f"Found {dataset.category_dir}, checking layout only.")
    elif args.source is not None:
        install_from_local(args.source, dataset)
    else:
        print(
            "Download the data through the web form on the MVTec website, "
            "then pass it with --source.",
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
