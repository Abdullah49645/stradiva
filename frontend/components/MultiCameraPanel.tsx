import { MultiCameraResult } from "@/lib/types";
import Eyebrow from "./Eyebrow";
import Panel from "./Panel";

function CameraCard({ label, cam }: { label: string; cam: MultiCameraResult["camera_A"] }) {
  const ok = cam.status === "RECONSTRUCTABLE";
  return (
    <div className="bg-graphite2 rounded p-4 border border-hairline">
      <div className="flex items-center justify-between mb-3">
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted">{label}</span>
        <span className={`font-mono text-[10px] uppercase tracking-wider ${ok ? "text-hud" : "text-hazard"}`}>
          {cam.status.replace(/_/g, " ")}
        </span>
      </div>
      {ok ? (
        <>
          <div className="font-display text-3xl font-semibold text-paper">
            {cam.speed_mps?.toFixed(1)} <span className="text-sm text-muted font-body">m/s</span>
          </div>
          <div className="font-mono text-[11px] text-muted mt-1">
            90% CI {cam.ci90_lo?.toFixed(1)}–{cam.ci90_hi?.toFixed(1)} m/s
          </div>
          {cam.used_color_blob_fallback && (
            <div className="mt-2 font-mono text-[10px] text-signal">
              ⚠ primary tracker degenerate — used color-blob fallback (see docs/multi_camera_architecture.md)
            </div>
          )}
          {cam.dominant_sources && cam.dominant_sources.length > 0 && (
            <div className="mt-2 font-mono text-[10px] text-muted/80">
              dominant uncertainty: {cam.dominant_sources.join(", ")}
            </div>
          )}
        </>
      ) : (
        <p className="font-mono text-[11px] text-muted">{cam.stop_reason}</p>
      )}
    </div>
  );
}

export default function MultiCameraPanel({ result }: { result?: MultiCameraResult }) {
  if (!result) return null;
  const crossExam = typeof result.cross_examination === "object" ? result.cross_examination : null;
  const disputed = crossExam ? !crossExam.intervals_overlap : false;

  return (
    <Panel>
      <Eyebrow>Multi-Camera Cross-Examination — Proof of Concept</Eyebrow>
      <p className="text-[13px] text-muted mt-3 mb-5 leading-relaxed">
        Two independently-positioned, unsynchronized cameras of the same event — no shared clock,
        no shared coordinate system, no ground truth fed into either reconstruction. Each camera
        detects its own sync anchor and speed estimate; only afterward are the two compared.
        See <code className="text-hud">docs/multi_camera_architecture.md</code> for exactly what
        is and isn&apos;t validated here.
      </p>

      <div className="grid sm:grid-cols-2 gap-4 mb-5">
        <CameraCard label="Camera A" cam={result.camera_A} />
        <CameraCard label="Camera B" cam={result.camera_B} />
      </div>

      {result.sync && (
        <div className="bg-graphite2 rounded p-4 border border-hairline mb-4 font-mono text-[12px]">
          <div className="text-muted uppercase text-[10px] tracking-wider mb-2">self-discovered sync (no ground truth used)</div>
          <div className="flex flex-wrap gap-x-6 gap-y-1">
            <span>estimated offset: <span className="text-paper">{result.sync.estimated_offset_b_to_a_s.toFixed(3)}s</span></span>
            <span className="text-muted">true offset (checked after, not fed in): {result.sync.true_offset_b_to_a_s.toFixed(3)}s</span>
            <span className="text-signal">error: {result.sync.sync_error_s.toFixed(3)}s</span>
          </div>
        </div>
      )}

      {crossExam && (
        <div className={`rounded border px-4 py-3 font-mono text-[12px] ${
          disputed ? "border-hazard/40 bg-hazard/10 text-hazard" : "border-hud/30 bg-hud/5 text-hud"
        }`}>
          {crossExam.verdict}
        </div>
      )}
    </Panel>
  );
}
