// Shapes of the JSON produced by backend/pipeline/analyze.py (a "case").
// Kept deliberately loose: the UI renders whatever subset a case carries.

export type Verdict =
  | "AVOIDABLE"
  | "NOT_AVOIDABLE"
  | "ASSUMPTION_DEPENDENT"
  | "INSUFFICIENT_EVIDENCE"
  | "NO_COLLISION_ANALYSED"
  | string;

export interface Check {
  check: string;
  status: "PASS" | "DISPUTE" | "NOT_APPLICABLE";
  statistic: unknown;
  threshold: unknown;
  evidence: string;
  catches: string;
}

export interface VehicleTrack {
  id: number;
  t_s: number[];
  x_m: number[];
  v_mps: number[];
  a_mps2: number[];
  y_m: number[];
  sv_mps: number[];
  uv: [number, number][];
}

export interface Counterfactual {
  verdict: Verdict;
  narrative: string;
  reaction_grid_s: number[];
  decel_grid_mps2: number[];
  p_avoid: number[][];
  critical_reaction_s: Record<string, { median_s: number; ci90_s: [number, number] }>;
  envelope: { reaction_s: [number, number]; decel_mps2: [number, number] };
  n_samples: number;
}

export interface Provenance {
  kind: "synthetic" | "real";
  title: string;
  summary: string;
  source: string;
  license?: string;
}

export interface Validation {
  note: string;
  rows: { label: string; truth: string; carai: string; ok?: boolean | null }[];
}

export interface CaseFile {
  id?: string;
  status: string;
  pipeline_version?: string;
  verdict?: { value: Verdict; reason: string };
  reasons?: string[];
  input?: { file: string; fps: number; width?: number; height?: number; sha256?: string; fps_note?: string | null };
  prior?: { name: string; dash_len_m: number; dash_gap_m: number; edge_to_centre_m: number; cycle_rel_sigma: number };
  geometry?: {
    ok: boolean;
    reasons?: string[];
    H?: number[][] | null;
    roll_deg?: number | null;
    calibrated_x_range_m?: [number, number] | null;
    diagnostics?: Record<string, unknown>;
  };
  vehicles?: VehicleTrack[];
  collision?: {
    detected: boolean;
    t_contact_s: number | null;
    t_contact_ci90_s?: [number, number] | null;
    source?: string;
    follower_id?: number;
    lead_id?: number;
    traffic_direction?: string;
  } | null;
  hazard_onset?: { source: string; t_s: number; t_lo_s?: number; t_hi_s?: number; description: string };
  follower_speed_at_hazard_onset?: { median_mps: number; ci90_mps: [number, number] } | null;
  speeds?: { vehicle_id: number; t_ref_s: number; speed_mps: number; ci90_mps: [number, number] | null;
             uncertainty_budget_rel_sigma?: Record<string, number | null> }[];
  counterfactual?: Counterfactual;
  skeptic?: { disputed: boolean; disputed_checks: string[]; checks: Check[]; untestable: string[] };
  identifiability?: { quantity: string; status: string; basis: string }[];
  stages?: { stage: string; status: string; detail: string; t_s: number }[];
  timings_s?: Record<string, number>;
  dashcam?: { fps: number; fps_source?: string; prior: { name: string; cycle_m: number }; windows: DashWindow[] };
  // added by evaluation/demo/build_cases.py for the bundled cases only
  provenance?: Provenance;
  validation?: Validation;
  video?: string;
}

export interface DashWindow {
  t_start_s: number;
  t_end_s: number;
  status: "OK" | "ABSTAIN";
  speed_mps: number | null;
  ci90_mps: [number, number] | null;
  reasons: string[];
  truth_mps?: number;
}

export interface JobStatusResponse {
  job_id: string;
  status: string;
  stage?: string;
  result?: CaseFile;
  error?: string;
}
