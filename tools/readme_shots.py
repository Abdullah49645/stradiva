"""
Screenshots for the README (docs/screenshots/readme_*.jpg), cropped to one
section each, from the static UI build (frontend/out).

  cd frontend && npm run build && cd .. && python3 tools/readme_shots.py
"""
import functools
import http.server
import io
import os
import threading

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "screenshots")
os.makedirs(OUT, exist_ok=True)


def serve(port=8766):
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.join(ROOT, "frontend", "out"))
    h.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), h)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def save(png, name, width=1200):
    im = Image.open(io.BytesIO(png)).convert("RGB")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    im.save(os.path.join(OUT, f"readme_{name}.jpg"), quality=82)


def section(pg, heading):
    """The <section> that contains the h2 with this text."""
    return pg.locator("section", has=pg.get_by_role("heading", name=heading, exact=False)).first


def main():
    httpd = serve()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        pg.goto("http://127.0.0.1:8766/", wait_until="networkidle")
        pg.wait_for_timeout(1200)

        def pick(title):
            pg.locator("button.pick", has_text=title).first.click()
            pg.wait_for_timeout(1800)

        pick("Real dashcam")
        save(pg.locator("article section").first.screenshot(), "hero_dashcam")
        save(section(pg, "Vehicle ahead").screenshot(), "lead_vs_radar")
        pick("Rear-end, roadside camera")
        pg.locator("video").first.evaluate("v => { v.currentTime = 1.6; }")
        pg.wait_for_timeout(1500)
        save(pg.locator("article section").first.screenshot(), "rear_end")
        save(section(pg, "Which assumption decides").screenshot(), "envelope")
        save(section(pg, "How close they came").screenshot(), "near_miss")
        pick("Camera knocked mid-clip")
        save(section(pg, "What Odolite tried").screenshot(), "investigator")
        pg.get_by_role("tab", name="How it was tested").click()
        pg.wait_for_timeout(1500)
        save(section(pg, "v2.2 with the investigator").screenshot(), "test3")
        b.close()
    httpd.shutdown()
    print("README screenshots in", OUT)


if __name__ == "__main__":
    main()
