export type Units = "metric" | "imperial";

export function speed(mps: number | null | undefined, units: Units): string {
  if (mps == null || !isFinite(mps)) return "–";
  return units === "metric" ? `${(mps * 3.6).toFixed(0)} km/h` : `${(mps * 2.23694).toFixed(0)} mph`;
}

export function speedRange(r: [number, number] | null | undefined, units: Units): string {
  if (!r) return "–";
  const f = units === "metric" ? 3.6 : 2.23694;
  const u = units === "metric" ? "km/h" : "mph";
  return `${(r[0] * f).toFixed(0)}–${(r[1] * f).toFixed(0)} ${u}`;
}

export function secs(s: number | null | undefined, digits = 2): string {
  if (s == null || !isFinite(s)) return "–";
  return `${s.toFixed(digits)} s`;
}

export const VERDICT_TEXT: Record<string, { title: string; tone: string }> = {
  AVOIDABLE: { title: "Avoidable under every assumption tested", tone: "avoidable" },
  NOT_AVOIDABLE: { title: "Not avoidable under any assumption tested", tone: "unavoidable" },
  ASSUMPTION_DEPENDENT: { title: "Depends on what the video cannot show", tone: "depends" },
  INSUFFICIENT_EVIDENCE: { title: "No verdict: the reconstruction was disputed", tone: "withheld" },
  INSUFFICIENT_EVIDENCE_OTHER: { title: "No verdict: the footage does not support one", tone: "withheld" },
  NO_COLLISION_ANALYSED: { title: "No contact found while the vehicles were in view", tone: "neutral" },
  NO_PAIR_RECONSTRUCTED: { title: "No collision analysis: no two vehicles were tracked in the same lane", tone: "neutral" },
  CONTACT_UNDETERMINED: { title: "No verdict: the footage cannot settle whether they touched", tone: "withheld" },
  NOT_RECONSTRUCTABLE: { title: "This footage cannot support measurements", tone: "withheld" },
  STANDARD_DEPENDENT: { title: "No verdict: it depends on which marking standard applies", tone: "withheld" },
  MOVING_CAMERA: { title: "Dashcam footage: own-vehicle speed only", tone: "neutral" },
};

export function verdictKey(c: { status: string; verdict?: { value: string } }): string {
  if (c.status === "NOT_RECONSTRUCTABLE" || c.status === "INVALID_INPUT") return "NOT_RECONSTRUCTABLE";
  if (c.status === "MOVING_CAMERA") return "MOVING_CAMERA";
  if (c.status === "CONTACT_UNDETERMINED") return "CONTACT_UNDETERMINED";
  if (c.status === "STANDARD_DEPENDENT") return "STANDARD_DEPENDENT";
  if (c.verdict?.value === "INSUFFICIENT_EVIDENCE" && c.status !== "DISPUTED") return "INSUFFICIENT_EVIDENCE_OTHER";
  return c.verdict?.value ?? "NO_COLLISION_ANALYSED";
}

export const CHECK_NAMES: Record<string, string> = {
  held_out_markings: "Markings predicted from each other",
  temporal_stability: "Camera stayed still",
  marking_standard: "Marking proportions match the standard",
  vehicle_width: "Vehicle widths are physically plausible",
  kinematics: "Motion is physically possible",
};

export const STATUS_TEXT: Record<string, string> = {
  MEASURED: "Measured",
  MEASURED_GIVEN_PRIOR: "Measured, given a stated assumption",
  MEASURED_BY_EXTRAPOLATION: "Estimated by extrapolation",
  ASSUMED: "Assumed",
  NOT_OBSERVED: "Not observed",
  NOT_IDENTIFIABLE: "Cannot be known from this video",
};

export const INPUT_NAMES: Record<string, string> = {
  fps: "Frame rate",
  focal_px: "Focal length (pixels)",
  marking_standard: "Marking standard",
  footage: "Better footage",
};

export const STANDARD_NAMES: Record<string, string> = {
  carai_synthetic: "CARAI synthetic 3 m + 9 m (test roads)",
  us_mutcd: "US MUTCD 10 ft + 30 ft",
  caltrans_freeway: "Caltrans freeway 12 ft + 36 ft",
  fr_t1: "France T1 3 m + 10 m",
  at_motorway: "Austria motorway 6 m + 12 m",
  at_rural: "Austria rural 6 m + 9 m",
  at_urban: "Austria urban 3 m + 3 m",
};
