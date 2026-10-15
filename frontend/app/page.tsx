"use client";

import { useState, useEffect } from "react";
import UploadForm from "@/components/UploadForm";
import AvoidabilityPanel from "@/components/AvoidabilityPanel";
import ReconstructabilityPanel from "@/components/ReconstructabilityPanel";
import AgentTracePanel from "@/components/AgentTracePanel";
import TrajectoryChart from "@/components/TrajectoryChart";
import Eyebrow from "@/components/Eyebrow";
import Panel from "@/components/Panel";
import HudStatusBar from "@/components/HudStatusBar";
import MultiCameraPanel from "@/components/MultiCameraPanel";
import ContinuousFeedPanel from "@/components/ContinuousFeedPanel";
import { PipelineResult, MultiCameraResult, ErraticDrivingResult } from "@/lib/types";
import { pollJobUntilDone } from "@/lib/api";

type DemoResults = Record<string, PipelineResult>;
type ViewMode = "collision" | "multicam" | "continuous";

const VIEW_LABELS: Record<ViewMode, string> = {
  collision: "Collision Reconstruction",
  multicam: "Multi-Camera Cross-Examination",
  continuous: "Continuous Monitoring",
};

const SCENARIO_LABELS: Record<string, string> = {
  unavoidable: "Case 001 — Rear-End, High Closing Speed",
  avoidable_but_occurred: "Case 002 — Rear-End, Delayed Braking",
};

export default function Home() {
  const [demoResults, setDemoResults] = useState<DemoResults | null>(null);
  const [selectedDemo, setSelectedDemo] = useState<string | null>(null);
  const [liveJobId, setLiveJobId] = useState<string | null>(null);
  const [liveStatus, setLiveStatus] = useState<string | null>(null);
  const [liveResult, setLiveResult] = useState<PipelineResult | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("collision");
  const [multiCamResult, setMultiCamResult] = useState<MultiCameraResult | null>(null);
  const [erraticResult, setErraticResult] = useState<ErraticDrivingResult | null>(null);

  useEffect(() => {
    fetch("/demo-results.json")
      .then((r) => r.json())
      .then((data: DemoResults) => {
        setDemoResults(data);
        setSelectedDemo(Object.keys(data)[0] ?? null);
      })
      .catch(() => {});
    fetch("/multi-camera-results.json").then((r) => r.json()).then(setMultiCamResult).catch(() => {});
    fetch("/erratic-driving-results.json").then((r) => r.json()).then(setErraticResult).catch(() => {});
  }, []);

  useEffect(() => {
    if (!liveJobId) return;
    setLiveStatus("PENDING");
    pollJobUntilDone(liveJobId, (status) => {
      setLiveStatus(status.status);
      if (status.result) setLiveResult(status.result);
    }).catch((e) => setLiveStatus(`ERROR: ${e.message}`));
  }, [liveJobId]);

  const activeResult: PipelineResult | null =
    liveResult ?? (demoResults && selectedDemo ? demoResults[selectedDemo] : null);

  const collisionTimeS = activeResult?.collision_time_s ?? null;

  return (
    <main className="max-w-5xl mx-auto px-5 sm:px-8 py-8">
      <HudStatusBar status={liveStatus ?? activeResult?.status} />

      {/* Hero */}
      <header className="mb-12 fade-up">
        <Eyebrow>OpenCV AI Competition 2026 — Working Submission</Eyebrow>
        <h1 className="font-display text-5xl sm:text-7xl font-semibold mt-4 leading-[1.02] tracking-tight">
          Was it <span className="text-hud verdict-glow">avoidable?</span>
        </h1>
        <p className="text-muted mt-5 max-w-xl leading-relaxed text-[15px]">
          CARAI reconstructs vehicle motion from ordinary, uncalibrated accident footage —
          then determines, from the physics alone, whether the collision appears to have been
          physically avoidable. A second, adversarial agent — the Skeptic — cross-examines every
          conclusion before it ships. Never a fault claim. Never false precision.
        </p>
        <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-2 font-mono text-[10px] uppercase tracking-[0.1em] text-muted/80">
          <span className="flex items-center gap-1.5"><span className="w-1 h-1 rounded-full bg-hud" />reconstructor + skeptic</span>
          <span className="flex items-center gap-1.5"><span className="w-1 h-1 rounded-full bg-signal" />uncertainty-aware</span>
          <span className="flex items-center gap-1.5"><span className="w-1 h-1 rounded-full bg-muted" />no fault attribution</span>
        </div>
      </header>

      <div className="tick-rule mb-6" />

      {/* View switcher */}
      <div className="flex flex-wrap gap-2 mb-10 fade-up" style={{ animationDelay: "60ms" }}>
        {(Object.keys(VIEW_LABELS) as ViewMode[]).map((mode) => (
          <button
            key={mode}
            onClick={() => setViewMode(mode)}
            className={`px-3 py-1.5 rounded font-mono text-[11px] uppercase tracking-wider transition-colors border ${
              viewMode === mode
                ? "bg-hud text-asphalt border-hud"
                : "bg-graphite2 text-muted border-hairline hover:text-paper"
            }`}
          >
            {VIEW_LABELS[mode]}
          </button>
        ))}
      </div>

      {viewMode === "collision" && (
        <>
      {/* Intake */}
      <div className="grid sm:grid-cols-2 gap-5 mb-10 fade-up" style={{ animationDelay: "80ms" }}>
        <UploadForm onJobCreated={setLiveJobId} />

        <Panel>
          <Eyebrow>Case Log</Eyebrow>
          <p className="text-[13px] text-muted mt-3 mb-4 leading-relaxed">
            Real pipeline output, validated against independently-computed ground truth
            (see docs/week3_4_causality_assessment.md) — not live AWS data.
          </p>
          {demoResults ? (
            <div className="space-y-2">
              {Object.keys(demoResults).map((name) => (
                <button
                  key={name}
                  onClick={() => { setSelectedDemo(name); setLiveResult(null); setLiveJobId(null); }}
                  className={`w-full text-left px-3 py-2 rounded font-mono text-[12px] transition-colors ${
                    selectedDemo === name && !liveResult
                      ? "bg-hud text-asphalt"
                      : "bg-graphite2 text-muted hover:text-paper border border-hairline"
                  }`}
                >
                  {SCENARIO_LABELS[name] ?? name}
                </button>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted font-mono">loading case log…</p>
          )}
        </Panel>
      </div>

      {liveJobId && (
        <div className="mb-6 font-mono text-[12px] text-muted fade-up">
          job <span className="text-paper">{liveJobId}</span> — <span className="text-hud">{liveStatus}</span>
        </div>
      )}

      {activeResult && (
        <div className="space-y-5">
          {/* Ordered per docs/definition_of_done.md: what happened -> how
              vehicles moved -> avoidability -> confidence. Avoidability
              leads as the visual hero; speed/trajectory numbers support
              it rather than the other way around. */}
          <div className="fade-up" style={{ animationDelay: "140ms" }}>
            <AvoidabilityPanel causality={activeResult.causality} uncertainty={activeResult.uncertainty} />
          </div>

          {activeResult.trajectories && (
            <div className="fade-up" style={{ animationDelay: "200ms" }}>
              <Panel>
                <TrajectoryChart
                  vehicleA={activeResult.trajectories.vehicle_a}
                  vehicleB={activeResult.trajectories.vehicle_b}
                  brakingOnsetBeforeImpact={activeResult.causality?.braking_onset_time_before_impact_s}
                  collisionTimeS={collisionTimeS}
                />
              </Panel>
            </div>
          )}

          <div className="grid sm:grid-cols-2 gap-5 fade-up" style={{ animationDelay: "260ms" }}>
            <ReconstructabilityPanel recon={activeResult.reconstructability} geometry={activeResult.geometry} />
            {activeResult.tracking && (
              <Panel>
                <Eyebrow>Vehicle Tracking</Eyebrow>
                <div className="space-y-3 mt-4 font-mono text-[13px]">
                  <div className="flex justify-between">
                    <span className="text-muted">vehicle_a (lead)</span>
                    <span>{activeResult.tracking.vehicle_a_tracked} tracked / {activeResult.tracking.vehicle_a_lost} lost</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">vehicle_b (following)</span>
                    <span>{activeResult.tracking.vehicle_b_tracked} tracked / {activeResult.tracking.vehicle_b_lost} lost</span>
                  </div>
                </div>
              </Panel>
            )}
          </div>

          <div className="fade-up" style={{ animationDelay: "320ms" }}>
            <AgentTracePanel trace={activeResult.agent_trace} />
          </div>

          {activeResult.ground_truth_physics_avoidability && (
            <div className="rounded-md border border-hairline bg-graphite/50 p-4 font-mono text-[11px] text-muted fade-up"
                 style={{ animationDelay: "380ms" }}>
              <span className="text-hud">validation_note</span> (demo data only) — independently-computed
              ground truth for this scenario: {JSON.stringify(activeResult.ground_truth_physics_avoidability)}
            </div>
          )}
        </div>
      )}
        </>
      )}

      {viewMode === "multicam" && (
        <div className="fade-up" style={{ animationDelay: "140ms" }}>
          <MultiCameraPanel result={multiCamResult ?? undefined} />
        </div>
      )}

      {viewMode === "continuous" && (
        <div className="fade-up" style={{ animationDelay: "140ms" }}>
          <ContinuousFeedPanel result={erraticResult ?? undefined} />
        </div>
      )}

      <footer className="mt-16 pt-6 border-t border-hairline font-mono text-[11px] text-muted/70">
        CARAI is a decision-support tool, not certified forensic testimony.
      </footer>
    </main>
  );
}
