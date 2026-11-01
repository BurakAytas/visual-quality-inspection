"""Check that every result number in the README comes from a JSON file in results/.

A claimed number is written with a marker that names its source::

    Image AUROC 0.873<!-- claim: results/metrics.json#fabric.image_auroc -->
    Miss rate 4.2%<!-- claim: results/metrics.json#fabric.miss_rate -->

The README value must equal the JSON value rounded to the same number of decimals
(a trailing % means the JSON holds a fraction). Inside a block between
``<!-- results:start -->`` and ``<!-- results:end -->`` every number needs a marker,
so a number cannot be typed in by hand.
"""

import argparse
import json
import re
import sys
from pathlib import Path

CLAIM = re.compile(r"(-?\d+(?:\.\d+)?)(%?)\s*<!--\s*claim:\s*([^#\s]+)#(\S+)\s*-->")
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")
RESULTS_BLOCK = re.compile(r"<!--\s*results:start\s*-->(.*?)<!--\s*results:end\s*-->", re.S)


def lookup(data: object, dotted_key: str) -> float:
    """Follow ``a.b.c`` through nested dicts and return a number.

    JSON keys may contain dots themselves (``pixel_aupro_0.3``), so at each level the
    longest key that matches the start of the remaining path is used. A number selects
    a list item.
    """
    if dotted_key:
        if isinstance(data, list):  # e.g. a confidence interval: ci95.0 / ci95.1
            index, _, rest = dotted_key.partition(".")
            if not index.isdigit() or int(index) >= len(data):
                raise KeyError(dotted_key)
            return lookup(data[int(index)], rest)
        if not isinstance(data, dict):
            raise KeyError(dotted_key)
        for key in sorted(data, key=len, reverse=True):
            if dotted_key == key or dotted_key.startswith(key + "."):
                return lookup(data[key], dotted_key[len(key) + 1 :])
        raise KeyError(dotted_key)
    if isinstance(data, bool) or not isinstance(data, int | float):
        raise TypeError("value is not a number")
    return float(data)


def check(readme: Path, repo_root: Path) -> list[str]:
    """Return a list of human-readable errors; empty means every claim checks out."""
    text = readme.read_text(encoding="utf-8")
    errors = []

    for match in CLAIM.finditer(text):
        shown, percent, rel_path, key = match.groups()
        source = repo_root / rel_path
        try:
            value = lookup(json.loads(source.read_text(encoding="utf-8")), key)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"{rel_path}#{key}: cannot read ({exc!r})")
            continue
        if percent:
            value *= 100
        decimals = len(shown.split(".")[1]) if "." in shown else 0
        expected = f"{value:.{decimals}f}"
        if expected != shown:
            errors.append(f"{rel_path}#{key}: README says {shown}, JSON gives {expected}")

    for block in RESULTS_BLOCK.findall(text):
        unmarked = NUMBER.findall(CLAIM.sub("", block))
        if unmarked:
            errors.append(f"Numbers without a claim marker in a results block: {unmarked}")
    return errors


def main() -> int:
    """Run the check and print a short report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    errors = check(args.readme, args.root)
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
    claims = len(CLAIM.findall(args.readme.read_text(encoding="utf-8")))
    print(f"{claims} claims checked, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
