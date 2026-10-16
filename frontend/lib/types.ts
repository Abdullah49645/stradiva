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
  status: "PASS" | "DISPUTE" | "ADVISORY" | "NOT_APPLICABLE";
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
  speeds?: { vehicle_id: number; t_ref_s: number; speed_mps: number; ci90_mps: [number, number] | null; withheld?: boolean; withheld_reason?: string;
             uncertainty_budget_rel_sigma?: Record<string, number | null> }[];
  counterfactual?: Counterfactual;
  skeptic?: { disputed: boolean; disputed_checks: string[]; checks: Check[]; untestable: string[] };
  identifiability?: { quantity: string; status: string; basis: string }[];
  stages?: { stage: string; status: string; detail: string; t_s: number }[];
  timings_s?: Record<string, number>;
  dashcam?: { fps: number; fps_source?: string; prior: { name: string; cycle_m: number }; windows: DashWindow[] };
  near_miss?: NearMiss;
  investigation?: Investigation;
  dashcam_lead?: DashcamLead;
  marking_standard_candidates?: { standard: string; name: string; cycle_m: number; source: string; status: string; verdict: string | null }[];
  segment_analysis?: { camera_shift_frame: number; camera_shift_t_s: number; adopted_frames: [number, number];
                       segments: { frames: [number, number]; t_s?: [number, number]; status?: string; verdict?: string | null; skipped?: string }[] };
  quarantine?: Record<string, { frames: number[]; fraction: number }>;
  time_base?: string;
  // added by evaluation/demo/build_cases.py for the bundled cases only
  provenance?: Provenance;
  validation?: Validation;
  video?: string;
}

export interface Interval { median: number; ci90: [number, number] }

export interface NearMiss {
  withheld?: string;
  disputed?: boolean;
  observed_interval_s?: [number, number];
  n_samples?: number;
  ttc_cap_s?: number;
  p_conflict_ttc_below_1_5s?: number;
  t_min_ttc_s?: number | null;
  min_gap_m?: Interval;
  min_ttc_s?: Interval;
  max_drac_mps2?: Interval;
  definitions?: Record<string, string>;
}

export interface InvestigationStep {
  step: string;
  diagnosis?: string;
  action?: string;
  observation?: string | Record<string, unknown>;
  decision?: string;
  candidates?: { standard: string; name: string; cycle_m: number; status: string; verdict: string | null; sub_step?: InvestigationStep }[];
  segments?: { frames: [number, number]; t_s?: [number, number]; status?: string; verdict?: string | null; skipped?: string }[];
  re_examination?: { status: string; verdict: string | null; disputed_checks: string[] | null };
}

export interface Investigation {
  base: { status: string; verdict: string | null; disputed_checks: string[] | null };
  final: { status: string; verdict: string | null; disputed_checks: string[] | null };
  trace: InvestigationStep[];
  requests: { input: string; why: string; options?: string[] }[];
  changed: boolean;
  user_inputs?: Record<string, unknown>;
}

export interface LeadWindow {
  t_start_s: number;
  t_end_s: number;
  status: string;
  reasons?: string[];
  range_m?: number;
  range_ci90_m?: [number, number];
  rel_speed_mps?: number;
  rel_speed_ci90_mps?: [number, number];
  ttc_s?: number | null;
  ttc_ci90_s?: [number, number] | null;
  radar_range_m?: number | null;
  radar_rel_speed_mps?: number | null;
  radar_ttc_s?: number | null;
  part?: "development" | "test";
}

export interface DashcamLead {
  status: string;
  reason?: string;
  camera_height_m?: number;
  range_offset_m?: number;
  windows?: LeadWindow[];
  summary_test?: Record<string, unknown>;
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
