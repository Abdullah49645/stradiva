"""
Integration tests (slower: render a short synthetic clip, run real stages).

  * geometry: the homography recovered from rendered markings maps pixels to
    the simulator's true road coordinates (the simulator's exact homography
    is used only here, never by the pipeline);
  * pipeline: a clip with no road markings is refused (NOT_RECONSTRUCTABLE)
    rather than given a number;
  * local API: the job contract the UI uses (create, upload, poll).

Run:  python3 -m pytest -q tests
"""
import os
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.geometry.road_geometry import RoadPrior, estimate_road_geometry, image_to_road  # noqa: E402
from evaluation.sim.world import (CameraSpec, Degradation, RoadSpec, Scenario, Camera,  # noqa: E402
                                  render_scenario)


def _frames(path, n=None):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok or (n and len(out) >= n):
            break
        out.append(f)
    return out


@pytest.fixture(scope="module")
def marked_road(tmp_path_factory):
    d = tmp_path_factory.mktemp("geo")
    cam = CameraSpec(focal_px=1000.0, height_m=7.0, pitch_deg=14.0, roll_deg=1.5, y_m=-5.0)
    sc = Scenario(name="geo_test", seed=11, fps=30.0, duration_s=1.0, camera=cam,
                  road=RoadSpec(dash_len_m=3.0, dash_gap_m=9.0, dash_phase_m=2.0),
                  degradation=Degradation(noise_std=3.0, blur_px=0.6))
    video, _, gt = render_scenario(sc, str(d))
    return sc, video, gt


def test_geometry_recovers_true_road_coordinates(marked_road):
    sc, video, gt = marked_road
    geo = estimate_road_geometry(_frames(video), RoadPrior(edge_to_centre_m=sc.road.edge_half_width_m,
                                                           dash_len_m=sc.road.dash_len_m, dash_gap_m=sc.road.dash_gap_m))
    assert geo.ok, geo.reasons
    assert geo.roll_deg == pytest.approx(sc.camera.roll_deg, abs=0.3)
    # true road points on the calibrated stretch -> image (true camera) -> road (estimated H)
    cam = Camera(sc.camera)
    x0, x1 = geo.calibrated_x_range_m
    xs = np.linspace(x0 + 0.1 * (x1 - x0), x1 - 0.1 * (x1 - x0), 6)
    pts = np.array([[x, y, 0.0] for x in xs for y in (-1.5, 0.0, 1.5)])
    uv = cam.project(pts)[0]
    est = image_to_road(geo.H, uv)
    # the estimated X origin is arbitrary: compare distances along the road (the speed scale)
    true_d = np.diff(pts[::3, 0])
    est_d = np.diff(est[::3, 0])
    assert np.max(np.abs(est_d / true_d - 1)) < 0.03
    assert np.max(np.abs(est[:, 1] - est[:, 1].mean() - (pts[:, 1] - pts[:, 1].mean()))) < 0.25


def test_unmarked_road_is_refused(tmp_path):
    from backend.pipeline.analyze import analyze_video
    sc = Scenario(name="no_marks", seed=3, fps=30.0, duration_s=1.0,
                  road=RoadSpec(paint_level=72, asphalt_level=72))   # paint indistinguishable from asphalt
    video, _, _ = render_scenario(sc, str(tmp_path))
    case = analyze_video(video, seed=0)
    assert case["status"] == "NOT_RECONSTRUCTABLE"
    assert (case.get("verdict") or {}).get("value") in (None, "NOT_RECONSTRUCTABLE")


def test_local_api_job_contract(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    monkeypatch.setenv("STRADIVA_WORK_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from backend.api import server
    client = TestClient(server.app)
    assert client.get("/health").json()["ok"] is True
    assert client.post("/jobs", json={"filename": "x.exe"}).status_code == 400
    job = client.post("/jobs", json={"filename": "clip.mp4"}).json()
    assert job["job_id"] and job["upload_url"]
    path = job["upload_url"].split("://", 1)[-1].split("/", 1)[-1]
    r = client.put("/" + path, content=b"this is not a video")
    assert r.status_code in (200, 204)
    status = None
    for _ in range(100):
        status = client.get(f"/jobs/{job['job_id']}").json()["status"]
        if status not in ("PENDING", "PROCESSING"):
            break
        time.sleep(0.1)
    assert status in ("INVALID_INPUT", "ERROR")
    assert client.get("/jobs/does-not-exist").status_code == 404
