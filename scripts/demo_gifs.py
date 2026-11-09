"""Cut the demo video into the two README GIFs: swiping, and chatting with a paper.

    uv run --with playwright --with pillow python scripts/demo_gifs.py docs/demo.webm

Chrome plays the video and each frame is captured at a set time (Playwright's own ffmpeg build
can't export frames), then Pillow writes a GIF with one shared palette. Re-check SEGMENTS
after re-recording the video with scripts/record_demo.py.
"""

import sys
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
SEGMENTS = {"demo-swipe.gif": (5, 27), "demo-chat.gif": (28, 50)}  # seconds in docs/demo.webm
FPS = 8
WIDTH, HEIGHT = 800, 500  # the video is 960 x 600
HOLD_LAST_MS = 3000


def frames(video: Path, start: float, end: float, work: Path) -> list[Image.Image]:
    player = work / "player.html"
    player.write_text(f'<body style="margin:0"><video id=v src="{video.as_uri()}" muted preload=auto '
                      f'style="width:{WIDTH}px;height:{HEIGHT}px;display:block"></video></body>')
    shots = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--allow-file-access-from-files"])
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(player.as_uri())
        page.wait_for_function("document.getElementById('v').readyState >= 2")
        n = 0
        while (t := start + n / FPS) <= end:
            page.evaluate(f"document.getElementById('v').currentTime = {t}")
            page.wait_for_function("!document.getElementById('v').seeking && document.getElementById('v').readyState >= 2")
            path = work / f"{n:04d}.png"
            page.screenshot(path=str(path))
            shots.append(Image.open(path).convert("RGB"))
            n += 1
        browser.close()
    return shots


def write_gif(shots: list[Image.Image], out: Path) -> None:
    palette = shots[len(shots) // 2].quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    quantized = [shot.quantize(palette=palette, dither=Image.Dither.NONE) for shot in shots]
    durations = [round(1000 / FPS)] * (len(quantized) - 1) + [HOLD_LAST_MS]
    quantized[0].save(out, save_all=True, append_images=quantized[1:], duration=durations, loop=0,
                      optimize=True, disposal=1)
    print(f"-> {out} ({len(quantized)} frames, {out.stat().st_size / 1e6:.1f} MB)")


def main(video: Path) -> None:
    for name, (start, end) in SEGMENTS.items():
        with tempfile.TemporaryDirectory() as work:
            write_gif(frames(video.resolve(), start, end, Path(work)), OUT_DIR / name)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
