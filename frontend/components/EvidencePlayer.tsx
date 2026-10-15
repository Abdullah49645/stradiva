"use client";

import { useEffect, useRef, useState } from "react";
import { CaseFile } from "@/lib/types";
import { Units, speed } from "@/lib/format";

// Inverse of a 3x3 matrix (road -> image from the case's image -> road homography).
function inv3(m: number[][]): number[][] {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

function apply(m: number[][], x: number, y: number): [number, number] | null {
  const w = m[2][0] * x + m[2][1] * y + m[2][2];
  if (Math.abs(w) < 1e-9) return null;
  return [(m[0][0] * x + m[0][1] * y + m[0][2]) / w, (m[1][0] * x + m[1][1] * y + m[1][2]) / w];
}

const COLORS = ["#d9a400", "#7fb3e6", "#f08a7e", "#9fd3b4"];

export default function EvidencePlayer({ c, units }: { c: CaseFile; units: Units }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showGrid, setShowGrid] = useState(true);
  const [showTracks, setShowTracks] = useState(true);
  const [t, setT] = useState(0);

  const W = c.input?.width ?? 1280;
  const Hh = c.input?.height ?? 720;
  const Himg = c.geometry?.H ?? null;
  const G = Himg ? inv3(Himg) : null;

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const v = videoRef.current;
      if (v) setT(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    cv.width = rect.width * dpr;
    cv.height = rect.height * dpr;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform((rect.width * dpr) / W, 0, 0, (rect.height * dpr) / Hh, 0, 0);
    ctx.clearRect(0, 0, W, Hh);
    const px = W / rect.width; // one screen pixel in image units

    if (showGrid && G && c.geometry?.calibrated_x_range_m && c.prior) {
      const [x0, x1] = c.geometry.calibrated_x_range_m;
      const half = c.prior.edge_to_centre_m;
      ctx.strokeStyle = "rgba(127,179,230,0.85)";
      ctx.fillStyle = "rgba(127,179,230,0.95)";
      ctx.lineWidth = 1.5 * px;
      ctx.font = `${12 * px}px Overpass, sans-serif`;
      const start = Math.ceil(x0 / 10) * 10;
      for (let X = start; X <= x1 + 1e-6; X += 10) {
        const a = apply(G, X, -half), b = apply(G, X, half);
        if (!a || !b) continue;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        if (((X - start) / 10) % 2 === 0) ctx.fillText(`${(X - start).toFixed(0)} m`, b[0] + 6 * px, b[1] + 4 * px);
      }
      for (const Y of [-half, 0, half]) {
        ctx.beginPath();
        let first = true;
        for (let X = x0; X <= x1; X += 2) {
          const p = apply(G, X, Y);
          if (!p) continue;
          if (first) { ctx.moveTo(p[0], p[1]); first = false; } else ctx.lineTo(p[0], p[1]);
        }
        ctx.setLineDash(Y === 0 ? [6 * px, 6 * px] : []);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    if (showTracks && c.vehicles) {
      c.vehicles.forEach((veh, vi) => {
        const col = COLORS[vi % COLORS.length];
        const n = veh.t_s.length;
        let k = -1;
        for (let i = 0; i < n; i++) if (veh.t_s[i] <= t + 1e-3) k = i;
        if (k < 0 || t - veh.t_s[k] > 0.25) return;
        ctx.strokeStyle = col;
        ctx.lineWidth = 2 * px;
        ctx.beginPath();
        for (let i = Math.max(0, k - 45); i <= k; i++) {
          const [u, vv] = veh.uv[i];
          if (i === Math.max(0, k - 45)) ctx.moveTo(u, vv); else ctx.lineTo(u, vv);
        }
        ctx.stroke();
        const [u, vv] = veh.uv[k];
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(u, vv, 5 * px, 0, Math.PI * 2); ctx.fill();
        const label = `${speed(Math.abs(veh.v_mps[k]), units)} ±${speed(1.645 * veh.sv_mps[k], units).replace(/ .*/, "")}`;
        ctx.font = `700 ${14 * px}px Overpass, sans-serif`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(20,28,34,0.82)";
        ctx.fillRect(u + 9 * px, vv - 26 * px, tw + 12 * px, 20 * px);
        ctx.fillStyle = col;
        ctx.fillText(label, u + 15 * px, vv - 11 * px);
      });
    }
    const tc = c.collision?.t_contact_s;
    if (tc != null && Math.abs(t - tc) < 0.2) {
      ctx.fillStyle = "rgba(158,42,43,0.9)";
      ctx.font = `800 ${18 * px}px Overpass, sans-serif`;
      ctx.fillText("contact", 16 * px, 30 * px);
    }
  }, [t, showGrid, showTracks, c, G, W, Hh, units]);

  return (
    <div>
      <div className="player" style={{ aspectRatio: `${W} / ${Hh}` }}>
        {c.video ? (
          <video ref={videoRef} src={c.video} controls muted playsInline preload="metadata" />
        ) : (
          <div style={{ color: "#ccc", padding: 24 }}>No video attached to this case.</div>
        )}
        <canvas ref={canvasRef} aria-hidden="true" />
      </div>
      <div className="player-bar">
        <label className="toggle"><input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
          Road grid recovered from the markings (10 m)</label>
        <label className="toggle"><input type="checkbox" checked={showTracks} onChange={(e) => setShowTracks(e.target.checked)} />
          Tracked ground contacts</label>
        <span className="num">t = {t.toFixed(2)} s</span>
      </div>
    </div>
  );
}
