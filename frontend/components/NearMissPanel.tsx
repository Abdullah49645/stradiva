import { CaseFile, Interval } from "@/lib/types";

function iv(x: Interval | undefined, unit: string, digits = 1, cap?: number) {
  if (!x) return { v: "–", sub: "" };
  const f = (n: number) => (cap != null && n >= cap ? `≥ ${cap.toFixed(0)}` : n.toFixed(digits));
  return { v: `${f(x.median)} ${unit}`, sub: `90 % interval ${f(x.ci90[0])}–${f(x.ci90[1])} ${unit}` };
}

export default function NearMissPanel({ c }: { c: CaseFile }) {
  const nm = c.near_miss;
  if (!nm || nm.disputed) return null;
  const collided = c.collision?.t_contact_s != null;
  if (nm.withheld)
    return (
      <section className="section">
        <div className="section-head"><h2>How close they came</h2></div>
        <p className="small muted">Not reported: {nm.withheld}.</p>
      </section>
    );
  const gap = iv(nm.min_gap_m, "m");
  const ttc = iv(nm.min_ttc_s, "s", 2, nm.ttc_cap_s);
  const drac = iv(nm.max_drac_mps2, "m/s²");
  const [t0, t1] = nm.observed_interval_s ?? [0, 0];
  return (
    <section className="section">
      <div className="section-head">
        <h2>How close they came</h2>
        <span className="small muted">while both vehicles were measured, {t0.toFixed(2)}–{t1.toFixed(2)} s{collided ? ", up to 0.3 s before contact" : ""}</span>
      </div>
      <div className="facts" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", marginTop: 0 }}>
        <div className="fact"><div className="k">Smallest bumper gap</div><div className="v">{gap.v}<small>{gap.sub}</small></div></div>
        <div className="fact"><div className="k">Shortest time to collision</div><div className="v">{ttc.v}<small>{ttc.sub}</small></div></div>
        <div className="fact"><div className="k">Braking needed to avoid contact, at worst</div><div className="v">{drac.v}<small>{drac.sub}</small></div></div>
      </div>
      <p className="small" style={{ marginTop: 16 }}>
        Standard traffic-conflict measures, computed only where both vehicles were seen, never extrapolated. Time to
        collision is the gap divided by the closing speed; braking needed is the constant deceleration that would have
        stopped the closing in the gap available.
        {nm.p_conflict_ttc_below_1_5s != null && ` In ${(100 * nm.p_conflict_ttc_below_1_5s).toFixed(0)} % of the reconstructions consistent with the video, the time to collision fell below 1.5 s.`}
      </p>
    </section>
  );
}
