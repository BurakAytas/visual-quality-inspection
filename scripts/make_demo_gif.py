"""Record the README demo GIF from the running operator panel, without a screen recorder.

Starts the inspection API on a fresh, temporary decision log, replays one shift (every
leather test image, through the API like the line simulator), starts the panel, and drives
it with a headless browser: three parts are inspected, then the review queue and the
daily report are shown. Each step is a screenshot; the screenshots become the GIF. Rerun
it after any change to the panel or the models.

Needs the exported models, the leather data and the ``demo`` extra plus a browser:
    pip install -e ".[serve,demo]" && playwright install chromium
    python scripts/make_demo_gif.py
"""

import argparse
import io
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
from PIL import Image
from playwright.sync_api import Page, expect, sync_playwright

from vqi.provenance import warn_if_dirty

TEST_DIR = Path("data/MVTec_AD/leather/test")
# Inspected on camera: a good part, then two defect types whose marks are easy to see at GIF
# size. Picked by eye; good/000 was not used because it is a false alarm (score above the
# threshold), which the review queue shows instead.
SHOWN = ["good/005.png", "cut/003.png", "color/000.png"]
VIEWPORT = {"width": 1400, "height": 1100}
GIF_WIDTH = 900
# Seconds each screenshot stays on screen.
HOLD_S = {"start": 1.5, "result": 3.0, "tab": 3.0}


@contextmanager
def serve(command: list[str], env: dict[str, str], ready_url: str):  # noqa: ANN201
    """Run a server in the background until ``ready_url`` answers; stop it afterwards."""
    process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(120):
            try:
                if httpx.get(ready_url, timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"{command[2]} did not start: {process.stderr.read1().decode()}")
        yield
    finally:
        process.terminate()
        process.wait(timeout=30)


def replay_shift(api: str) -> int:
    """Post every leather test image once, as the line simulator would."""
    paths = sorted(
        p for p in TEST_DIR.rglob("*.png") if p.relative_to(TEST_DIR).as_posix() not in SHOWN
    )
    with httpx.Client(base_url=api, timeout=30) as client:
        for path in paths:
            name = path.relative_to(TEST_DIR).as_posix()
            client.post("/inspect", files={"file": (name, path.read_bytes())}).raise_for_status()
    return len(paths)


def shot(page: Page) -> Image.Image:
    """Screenshot of the visible panel."""
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(800)  # let Streamlit finish its fade-in
    return Image.open(io.BytesIO(page.screenshot())).convert("RGB")


def record(panel: str, frames: list[tuple[Image.Image, float]]) -> None:
    """Drive the panel and append (screenshot, seconds) pairs."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(panel)
        page.get_by_text("Serving").wait_for(timeout=60_000)
        frames.append((shot(page), HOLD_S["start"]))
        for name in SHOWN:
            page.locator("input[type=file]").set_input_files(
                files=[
                    {
                        "name": name.replace("/", "_"),
                        "mimeType": "image/png",
                        "buffer": (TEST_DIR / name).read_bytes(),
                    }
                ]
            )
            # The file name shows twice once the answer is in: upload chip and image caption.
            uploaded = page.get_by_text(name.replace("/", "_"), exact=True)
            expect(uploaded).to_have_count(2, timeout=60_000)
            frames.append((shot(page), HOLD_S["result"]))
        for tab in ["Review queue", "Daily report"]:
            page.get_by_role("tab", name=tab).click()
            frames.append((shot(page), HOLD_S["tab"]))
        browser.close()


def save_gif(frames: list[tuple[Image.Image, float]], out: Path) -> None:
    """Scale every frame to the GIF width and write a looping GIF."""
    scaled = []
    for image, _ in frames:
        height = round(image.height * GIF_WIDTH / image.width)
        resized = image.resize((GIF_WIDTH, height), Image.Resampling.LANCZOS)
        scaled.append(resized.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    durations = [round(seconds * 1000) for _, seconds in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    scaled[0].save(out, save_all=True, append_images=scaled[1:], duration=durations, loop=0)


def main() -> None:
    """Start API and panel, record, write the GIF."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("docs/demo.gif"))
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--panel-port", type=int, default=8766)
    args = parser.parse_args()
    warn_if_dirty()
    api = f"http://localhost:{args.api_port}"
    panel = f"http://localhost:{args.panel_port}"
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "VQI_DB_PATH": str(Path(tmp) / "decisions.sqlite"), "VQI_API_URL": api}
        uvicorn = [
            sys.executable,
            "-m",
            "uvicorn",
            "--factory",
            "vqi.api:app_from_env",
            "--port",
            str(args.api_port),
        ]
        streamlit = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/vqi/panel.py",
            "--server.port",
            str(args.panel_port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
            "--client.toolbarMode",
            "minimal",
        ]
        frames: list[tuple[Image.Image, float]] = []
        with serve(uvicorn, env, f"{api}/health"):
            print(f"Replayed {replay_shift(api)} parts", flush=True)
            with serve(streamlit, env, f"{panel}/_stcore/health"):
                record(panel, frames)
    save_gif(frames, args.out)
    print(f"Wrote {args.out}: {len(frames)} frames, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
