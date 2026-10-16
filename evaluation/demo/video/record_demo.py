"""
Records the 5-minute demo video from the real UI (static build + local API), scene by scene, then cuts
and encodes it with ffmpeg. Voiceover script: evaluation/demo/video/voiceover.md (timings match SCENES).

  # 1. a UI build wired to the local API, copied out of frontend/out:
  cd frontend && NEXT_PUBLIC_ODOLITE_API_BASE=http://127.0.0.1:8000 npm run build && cp -r out /tmp/rec_out && cd ..
  # 2. the local API (same job contract as AWS):
  ODOLITE_CORS_ORIGINS=http://127.0.0.1:8770 uvicorn backend.api.server:app --port 8000 &
  # 3. record:
  python3 evaluation/demo/video/record_demo.py /tmp/rec_out OUT_DIR

Every number on screen comes from the committed case and evaluation files; the upload scene is a live
analysis by the local server. Nothing is staged.
"""
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SITE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "frontend", "out")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "evaluation", "_sim_cache", "demo_video")
PORT = 8770
VIEW = {"width": 1600, "height": 900}      # recorded at this size, scaled to 1080p by ffmpeg (larger UI text)
VIDEO = {"width": 1600, "height": 900}
CLIP = os.path.join(ROOT, "evaluation", "_sim_cache", "demo", "demo_rear_end.mp4")

CURSOR_JS = """
(() => {
  if (document.getElementById('demo-cursor')) return;
  const d = document.createElement('div'); d.id = 'demo-cursor';
  Object.assign(d.style, {position: 'fixed', left: '-40px', top: '-40px', width: '22px', height: '22px', borderRadius: '50%',
    background: 'rgba(217,164,0,0.55)', border: '2px solid #1e2a33', zIndex: 99999, pointerEvents: 'none',
    transform: 'translate(-50%,-50%)', transition: 'width .12s, height .12s'});
  document.documentElement.appendChild(d);
  addEventListener('mousemove', e => { d.style.left = e.clientX + 'px'; d.style.top = e.clientY + 'px'; }, true);
  addEventListener('mousedown', () => { d.style.width = '14px'; d.style.height = '14px'; }, true);
  addEventListener('mouseup', () => { d.style.width = '22px'; d.style.height = '22px'; }, true);
  window.__smoothScroll = (y, ms) => new Promise(res => {
    const y0 = scrollY, t0 = performance.now();
    const step = t => { const k = Math.min(1, (t - t0) / ms), e = k < .5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2) / 2;
      scrollTo(0, y0 + (y - y0) * e); k < 1 ? requestAnimationFrame(step) : res(); };
    requestAnimationFrame(step);
  });
})();
"""


def serve():
    root = os.path.join(OUT, "_site")
    if os.path.exists(root):
        shutil.rmtree(root)
    shutil.copytree(SITE, root)
    cards = os.path.join(root, "demo")
    os.makedirs(os.path.join(cards, "fonts"), exist_ok=True)
    for f in ("title.html", "arch.html", "close.html", "cards.css"):
        shutil.copy(os.path.join(HERE, f), cards)
    fs = os.path.join(ROOT, "frontend", "node_modules", "@fontsource")
    for fam, w in (("overpass", (400, 700, 800)), ("ibm-plex-sans", (400, 600))):
        for x in w:
            shutil.copy(os.path.join(fs, fam, "files", f"{fam}-latin-{x}-normal.woff2"), os.path.join(cards, "fonts"))
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
    h.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), h)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


class Scene:
    def __init__(self, page):
        self.p = page

    def wait(self, s):
        self.p.wait_for_timeout(int(s * 1000))

    def cursor(self):
        self.p.evaluate(CURSOR_JS)

    def move_to(self, locator, steps=30):
        b = locator.bounding_box()
        self.p.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, steps=steps)

    def click(self, locator):
        locator.scroll_into_view_if_needed()
        self.move_to(locator)
        self.wait(0.25)
        locator.click()

    def scroll_to(self, locator, offset=90, ms=1400):
        y = self.p.evaluate("([el, off]) => el.getBoundingClientRect().top + scrollY - off", [locator.element_handle(), offset])
        self.p.evaluate(f"window.__smoothScroll({y}, {ms})")
        self.wait(ms / 1000 + 0.2)

    def scroll_by(self, dy, ms=1400):
        self.p.evaluate(f"window.__smoothScroll(scrollY + {dy}, {ms})")
        self.wait(ms / 1000 + 0.2)

    def section(self, heading):
        return self.p.locator("section", has=self.p.get_by_role("heading", name=heading, exact=False)).first

    def case(self, title):
        self.click(self.p.locator("button.pick", has_text=title).first)
        self.wait(1.0)
        self.p.evaluate("window.__smoothScroll(0, 600)")
        self.wait(0.7)


def app(sc):
    sc.p.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
    sc.cursor()
    sc.p.mouse.move(800, 450)
    sc.wait(0.8)


def card(name, seconds):
    def run(sc):
        sc.p.goto(f"http://127.0.0.1:{PORT}/demo/{name}.html", wait_until="networkidle")
        sc.wait(0.3)
        sc.mark()
        sc.wait(seconds + 0.5)
    return run


def s_dashcam_mode(sc):
    """The added mode: moving cameras. Vehicle ahead vs radar, then the own-speed refusal."""
    app(sc)
    sc.case("Real dashcam")
    sc.p.locator("video").first.evaluate("v => { v.muted = true; v.play(); }")
    sc.mark()
    sc.wait(5)
    sc.scroll_to(sc.section("Vehicle ahead"), offset=30, ms=1600)
    sc.wait(3)
    charts = sc.section("Vehicle ahead").locator(".recharts-wrapper")
    sc.move_to(charts.nth(0), steps=40); sc.wait(8)
    sc.move_to(charts.nth(1), steps=40); sc.wait(7)
    sc.scroll_to(sc.section("Own-vehicle speed"), offset=30, ms=1600)
    sc.wait(4)
    sc.scroll_by(330, ms=1400)
    sc.wait(9)
    return None


def s_rear_end(sc):
    app(sc)
    sc.case("Rear-end, roadside camera")
    v = sc.p.locator("video").first
    sc.mark()
    sc.move_to(v, steps=40)
    v.evaluate("v => { v.muted = true; v.currentTime = 0; v.play(); }")
    sc.wait(8)
    v.evaluate("v => { v.currentTime = 0; v.playbackRate = 0.6; v.play(); }")
    sc.wait(12)
    sc.move_to(sc.p.locator(".facts").first, steps=40)
    sc.wait(13)
    return None


def s_envelope(sc):
    app(sc)
    sc.case("Rear-end, roadside camera")
    sc.scroll_to(sc.section("Which assumption decides"), offset=30, ms=10)
    sc.wait(0.5)
    sc.mark()
    sc.move_to(sc.p.locator(".envelope").first, steps=40)
    sc.wait(13)
    sc.scroll_to(sc.section("How close they came"), offset=30)
    sc.wait(9)
    sc.scroll_to(sc.section("Checked against ground truth"), offset=30)
    sc.wait(10)
    return None


def s_investigator(sc):
    app(sc)
    sc.case("Camera knocked mid-clip")
    sc.p.locator("video").first.evaluate("v => { v.muted = true; v.currentTime = 2.2; v.play(); }")
    sc.mark()
    sc.wait(5)
    sc.scroll_to(sc.section("What Odolite tried"), offset=30)
    sc.wait(9)
    sc.move_to(sc.p.locator(".inv-step").nth(0), steps=40)
    sc.wait(7)
    sc.scroll_to(sc.p.locator(".inv-step").nth(1), offset=160, ms=1600)
    sc.move_to(sc.p.locator(".inv-step").nth(1), steps=40)
    sc.wait(7)
    sc.move_to(sc.section("What Odolite tried").get_by_text("What would let Odolite say more"), steps=40)
    sc.wait(4)
    return None


def s_upload(sc):
    app(sc)
    drop = sc.p.locator(".drop").first
    sc.scroll_to(drop, offset=200, ms=10)
    sc.wait(0.5)
    sc.mark()
    sc.wait(1.5)
    sc.p.locator("input[type=file]").set_input_files(CLIP)
    sc.move_to(sc.p.locator("input[type=file]"))
    sc.wait(1.2)
    sc.click(drop.locator("summary"))
    sc.wait(1.0)
    fps = drop.locator("input[inputmode=decimal]").first
    sc.click(fps)
    fps.type("30", delay=180)
    sc.wait(1.0)
    sc.click(drop.get_by_role("button", name="Analyse clip"))
    t_wait0 = time.time()
    sc.p.get_by_text("Analysed just now by the Odolite service").first.wait_for(timeout=300000)
    t_wait1 = time.time()
    sc.p.evaluate("window.__smoothScroll(0, 900)")
    sc.wait(1.2)
    sc.wait(4)
    sc.scroll_to(sc.section("What this video establishes"), offset=30, ms=1500)
    sc.wait(3)
    return {"cut": [t_wait0 + 2.5, t_wait1 - 1.0]}


def s_results(sc):
    app(sc)
    sc.click(sc.p.get_by_role("tab", name="How it was tested"))
    sc.wait(1.0)
    sc.scroll_to(sc.section("v2.2 with the investigator"), offset=30, ms=10)
    sc.wait(0.5)
    sc.mark()
    sc.wait(14)
    sc.scroll_to(sc.section("What the investigator changed"), offset=30)
    sc.wait(8)
    sc.click(sc.p.get_by_role("tab", name="Real dashcam footage"))
    sc.wait(1.0)
    sc.scroll_to(sc.section("Vehicle ahead"), offset=30, ms=10)
    sc.wait(11)
    return None


# (name, function, seconds kept) -- the voiceover script follows this order and these lengths
SCENES = [
    ("01_title", card("title", 13), 13),
    ("04_rear_end", s_rear_end, None),
    ("05_envelope", s_envelope, None),
    ("06_investigator", s_investigator, None),
    ("07_upload", s_upload, None),
    ("07b_dashcam_mode", s_dashcam_mode, None),
    ("08_architecture", card("arch", 28), 28),
    ("09_results", s_results, None),
    ("10_close", card("close", 10), 10),
]


def record_scene(browser, name, fn):
    d = os.path.join(OUT, "raw", name)
    shutil.rmtree(d, ignore_errors=True)
    ctx = browser.new_context(viewport=VIEW, record_video_dir=d, record_video_size=VIDEO)
    page = ctx.new_page()
    t_start = time.time()
    sc = Scene(page)
    ready = {}
    sc.mark = lambda: ready.setdefault("t", time.time())
    extra = fn(sc)
    t_end = time.time()
    path = page.video.path()
    ctx.close()
    return {"video": path, "t_start": t_start, "t_end": t_end, "ready": ready.get("t"), "extra": extra}


def main():
    os.makedirs(OUT, exist_ok=True)
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None   # re-record only these scenes
    httpd = serve()
    logf = os.path.join(OUT, "scenes.json")
    log = json.load(open(logf)) if os.path.exists(logf) else {}
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        for name, fn, keep in SCENES:
            if only is not None and name not in only and name in log:
                continue
            e = record_scene(b, name, fn)
            e["keep"] = keep
            log[name] = e
            json.dump(log, open(logf, "w"), indent=1)
            print(name, round(e["t_end"] - e["t_start"], 1), "s recorded", flush=True)
        b.close()
    httpd.shutdown()
    cut(log)


def cut(log):
    parts = []
    for name, _, _ in SCENES:
        e = log[name]
        start = (e["ready"] or e["t_start"]) - e["t_start"]
        dur = e["keep"] if e["keep"] else (e["t_end"] - (e["ready"] or e["t_start"]))
        src = e["video"]
        seg = os.path.join(OUT, f"{name}.mp4")
        vf = "fps=30,scale=1920:1080:flags=lanczos,format=yuv420p"
        if e["extra"] and e["extra"].get("cut"):
            a, z = [x - e["t_start"] for x in e["extra"]["cut"]]
            # keep [start, a] and [z, end]: the analysis wait is cut, and the cut is marked on screen in the voiceover
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-filter_complex",
                            f"[0:v]trim={start}:{a},setpts=PTS-STARTPTS[a];[0:v]trim={z},setpts=PTS-STARTPTS[b];"
                            f"[a][b]concat=n=2:v=1[c];[c]{vf}[v]", "-map", "[v]", "-c:v", "libx264", "-crf", "16",
                            "-preset", "slow", seg], check=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", src, "-t", f"{dur:.2f}",
                            "-vf", vf, "-c:v", "libx264", "-crf", "16", "-preset", "slow", seg], check=True)
        parts.append(seg)
    lst = os.path.join(OUT, "parts.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    final = os.path.join(OUT, "odolite_demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-shortest", "-movflags", "+faststart", final], check=True)
    for p in parts:
        d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                 capture_output=True, text=True).stdout)
        print(os.path.basename(p), f"{d:.1f} s")
    print("wrote", final)


if __name__ == "__main__":
    main()
