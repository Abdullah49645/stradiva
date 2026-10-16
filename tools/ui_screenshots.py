"""
Screenshots of the static UI build (frontend/out) for review and for the
submission write-up. Serves out/ locally, then captures each tab and each
bundled case at desktop and phone widths.

  cd frontend && npm run build && cd .. && python3 tools/ui_screenshots.py OUT_DIR
"""
import functools
import http.server
import os
import sys
import threading

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "screenshots")
os.makedirs(OUT, exist_ok=True)


def serve(port=8765):
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.join(ROOT, "frontend", "out"))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), h)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    httpd = serve()
    url = "http://127.0.0.1:8765/"
    with sync_playwright() as p:
        b = p.chromium.launch()
        for label, vp in (("desktop", {"width": 1440, "height": 900}), ("phone", {"width": 390, "height": 844})):
            pg = b.new_page(viewport=vp)
            pg.goto(url, wait_until="networkidle")
            pg.wait_for_timeout(1200)
            picks = pg.locator("button.pick")
            for i in range(picks.count()):
                picks.nth(i).click()
                pg.wait_for_timeout(1500)
                pg.screenshot(path=os.path.join(OUT, f"{label}_case{i}.png"), full_page=True)
            for name in ("Real dashcam footage", "How it was tested"):
                pg.get_by_role("tab", name=name).click()
                pg.wait_for_timeout(1500)
                pg.screenshot(path=os.path.join(OUT, f"{label}_{name.split()[0].lower()}.png"), full_page=True)
            pg.close()
        b.close()
    httpd.shutdown()
    print("screenshots in", OUT)


if __name__ == "__main__":
    main()
