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
  NO_COLLISION_ANALYSED: { title: "No collision between tracked vehicles", tone: "neutral" },
  NOT_RECONSTRUCTABLE: { title: "This footage cannot support measurements", tone: "withheld" },
  MOVING_CAMERA: { title: "Dashcam footage: own-vehicle speed only", tone: "neutral" },
};

export function verdictKey(c: { status: string; verdict?: { value: string } }): string {
  if (c.status === "NOT_RECONSTRUCTABLE" || c.status === "INVALID_INPUT") return "NOT_RECONSTRUCTABLE";
  if (c.status === "MOVING_CAMERA") return "MOVING_CAMERA";
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
