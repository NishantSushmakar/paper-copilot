"""Record the Research Guru demo video with Playwright.

    uv run --with streamlit streamlit run ui/demo.py --server.port 8601 --server.headless true
    uv run --with playwright python scripts/record_demo.py docs/demo.webm

Drives the demo app in Chrome along a fixed script: the welcome screen, swiping through the
10 papers (3 kept), then two chats with typed questions whose answers were pre-computed by
ui/build_demo_answers.py. Videos don't show a mouse pointer, so the page gets an overlay
cursor that moves to each click, and a keycap that shows the swipe keys. Needs Playwright's
ffmpeg once: `uv run --with playwright playwright install ffmpeg`.
"""

import shutil
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8601"
SIZE = {"width": 1280, "height": 800}
OVERLAY = """
() => {
  if (document.getElementById('demo-cursor')) return;
  const style = document.createElement('style');
  style.textContent = `
    #demo-cursor { position: fixed; z-index: 99999; left: 640px; top: 420px; width: 22px; height: 22px;
      margin: -11px 0 0 -11px; border-radius: 50%; background: rgba(2,132,199,.35); border: 2px solid #0369a1;
      pointer-events: none; transition: left .55s cubic-bezier(.4,0,.2,1), top .55s cubic-bezier(.4,0,.2,1), transform .15s; }
    #demo-cursor.press { transform: scale(.7); background: rgba(2,132,199,.6); }
    #demo-key { position: fixed; z-index: 99999; left: 50%; bottom: 36px; transform: translateX(-50%) translateY(20px);
      min-width: 58px; padding: 10px 16px; border-radius: 12px; background: #0b2540; color: #fff; text-align: center;
      font: 600 24px -apple-system, sans-serif; box-shadow: 0 8px 24px rgba(0,0,0,.25); opacity: 0;
      transition: opacity .15s, transform .15s; pointer-events: none; }
    #demo-key.show { opacity: .92; transform: translateX(-50%); }`;
  document.head.appendChild(style);
  const c = document.createElement('div'); c.id = 'demo-cursor'; document.body.appendChild(c);
  const k = document.createElement('div'); k.id = 'demo-key'; document.body.appendChild(k);
}
"""


def overlay(page):
    page.evaluate(OVERLAY)


def click(page, locator, pause=0.5):
    locator.scroll_into_view_if_needed()
    box = locator.bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.evaluate("([x, y]) => { const c = document.getElementById('demo-cursor'); c.style.left = x + 'px'; c.style.top = y + 'px'; }", [x, y])
    time.sleep(0.7)
    page.evaluate("() => document.getElementById('demo-cursor').classList.add('press')")
    time.sleep(0.15)
    locator.click()
    page.evaluate("() => document.getElementById('demo-cursor').classList.remove('press')")
    time.sleep(pause)


def key(page, name, label, pause=1.0):
    page.evaluate("(l) => { const k = document.getElementById('demo-key'); k.textContent = l; k.classList.add('show'); }", label)
    page.keyboard.press(name)
    time.sleep(0.45)
    page.evaluate("() => document.getElementById('demo-key').classList.remove('show')")
    time.sleep(pause)


def hover(page, locator, pause=1.0):
    locator.scroll_into_view_if_needed()
    box = locator.bounding_box()
    page.evaluate("([x, y]) => { const c = document.getElementById('demo-cursor'); c.style.left = x + 'px'; c.style.top = y + 'px'; }",
                  [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2])
    locator.hover()
    time.sleep(0.7 + pause)


def type_question(page, text, pause=4.0):
    box = page.locator('[data-testid="stChatInputTextArea"]')
    click(page, box, pause=0.3)
    box.press_sequentially(text, delay=65)
    time.sleep(0.5)
    box.press("Enter")
    time.sleep(pause)


def main(out: Path) -> None:
    video_dir = out.parent / "_video"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        context = browser.new_context(viewport=SIZE, record_video_dir=str(video_dir), record_video_size={"width": 960, "height": 600})
        page = context.new_page()
        page.goto(URL, wait_until="networkidle")
        time.sleep(2)
        overlay(page)
        time.sleep(2.5)  # read the welcome screen

        click(page, page.get_by_role("button", name="Start swiping"), pause=3)  # read the first card
        for direction in "RLLRRLLLLL":  # keep Heads of Security, MERLIN, Metacognitive Decoding
            key(page, "ArrowRight" if direction == "R" else "ArrowLeft", "→  Save" if direction == "R" else "←  Skip")
        time.sleep(2)  # all caught up

        click(page, page.get_by_role("button", name="Open chats"), pause=3.5)  # the greeting types itself
        hover(page, page.get_by_text("Read on arXiv ↗"), pause=1.5)  # the paper itself is one click away
        type_question(page, "How does it compare with SecAlign on adaptive attacks?", pause=6)
        click(page, page.locator("details.gs-sources summary").first, pause=4)
        page.mouse.wheel(0, 500)
        time.sleep(1.5)
        hover(page, page.get_by_text("Check them in the full paper ↗"), pause=1.5)

        click(page, page.get_by_role("button", name="Multi-State Linear Attention"), pause=3.5)
        type_question(page, "Does it make inference slower?", pause=3.5)

        video = page.video.path()
        context.close()
        browser.close()
    shutil.move(video, out)
    shutil.rmtree(video_dir, ignore_errors=True)
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
