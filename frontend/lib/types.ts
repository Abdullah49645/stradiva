export interface ReconstructabilityComponents {
  geometry: number;
  scale: number;
  tracking: number;
  temporal: number;
}

export interface Reconstructability {
  overall: number;
  verdict: string;
  components: ReconstructabilityComponents;
  reasons?: string[];
}

export interface Geometry {
  roll_correction_applied_deg?: number | null;
  lane_width_self_consistency?: { max_abs_error_vs_assumed_m?: number } | null;
  world_x_span_m?: number | null;
  skeptic_disputed?: boolean;
  skeptic_findings?: { check: string; disputed: boolean; reason: string }[];
  [key: string]: unknown;
}

export interface ContributionFactors {
  following_vehicle_safe_speed_for_observed_gap_mps?: number | null;
  following_vehicle_speed_vs_safe_speed_pct?: number | null;
  lead_vehicle_deceleration_classification?: string;
  note?: string;
}

export interface CausalityAssessment {
  verdict: "AVOIDABLE" | "NOT_AVOIDABLE" | "INSUFFICIENT_EVIDENCE" | string;
  narrative: string;
  striking_vehicle_speed_at_earliest_observed_mps?: number | null;
  time_to_collision_at_earliest_observed_s?: number | null;
  braking_detected?: boolean;
  braking_onset_time_before_impact_s?: number | null;
  steering_detected?: boolean;
  steering_onset_time_before_impact_s?: number | null;
  steering_direction?: string | null;
  last_possible_avoidance_time_before_impact_s?: number | null;
  assumptions?: { max_decel_mps2?: number; reaction_time_s?: number };
  confidence?: string;
  evidence?: string[];
  contribution_factors?: ContributionFactors;
}

export interface UncertaintyInfo {
  striking_vehicle_speed_ci90_mps?: [number, number] | null;
  relative_uncertainty?: number;
  dominant_sources?: string[];
  sources?: Record<string, number>;
}

export interface AgentToolCall {
  tool: string;
  args: Record<string, unknown>;
  result: Record<string, unknown>;
  reasoning?: string;
}

export interface AgentTrace {
  success: boolean;
  chosen_frame_idx: number | null;
  n_iterations: number;
  stop_reason: string;
  calls: AgentToolCall[];
}

export interface TrackingInfo {
  vehicle_a_tracked: number;
  vehicle_a_lost: number;
  vehicle_b_tracked: number;
  vehicle_b_lost: number;
}

export interface Trajectory {
  t_s: number[];
  x_m: number[];
  speed_mps: number[];
}

export interface PipelineResult {
  status: string;
  geometry?: Geometry;
  agent_trace?: AgentTrace;
  tracking?: TrackingInfo;
  trajectories?: { vehicle_a: Trajectory; vehicle_b: Trajectory };
  reconstructability?: Reconstructability;
  causality?: CausalityAssessment;
  uncertainty?: UncertaintyInfo;
  collision_time_s?: number | null;
  ground_truth_physics_avoidability?: {
    required_stopping_distance_m: number;
    available_gap_m: number;
    physically_avoidable: boolean;
  };
  ground_truth_collided?: boolean;
}

export interface JobStatusResponse {
  job_id: string;
  status: string;
  result?: PipelineResult;
  report_markdown?: string;
  error?: string;
}

export interface CameraReconstruction {
  status: string;
  speed_mps?: number;
  ci90_lo?: number;
  ci90_hi?: number;
  event_t_local_s?: number;
  event_gap_m?: number;
  used_color_blob_fallback?: boolean;
  dominant_sources?: string[];
  stop_reason?: string;
}

export interface MultiCameraResult {
  scenario: string;
  camera_A: CameraReconstruction;
  camera_B: CameraReconstruction;
  sync?: { estimated_offset_b_to_a_s: number; true_offset_b_to_a_s: number; sync_error_s: number } | null;
  cross_examination?: {
    camera_a_speed_mps: number;
    camera_b_speed_mps: number;
    intervals_overlap: boolean;
    verdict: string;
  } | string;
}

export interface ErraticInterval {
  start_s: number;
  end_s: number;
  type: string;
  reason: string;
  threshold_m?: number;
  peak_decel_mps2?: number;
}

export interface ErraticGroundTruthSegment {
  type: string;
  t_start_s: number;
  t_end_s: number;
}

export interface ErraticDrivingResult {
  scenario: string;
  tracking_coverage: number;
  detected: ErraticInterval[];
  ground_truth_erratic_segments: ErraticGroundTruthSegment[];
  recall: number | null;
  precision: number | null;
}
