"""
Library of broken-line (dash) marking standards the investigator can try
when the Skeptic's duty-cycle check says the assumed standard does not match
the road.

Only standards with a checkable source are listed. The duty cycle (dash /
cycle) can be MEASURED from the image; the cycle length cannot (a single
static view does not fix the speed scale). So the investigator keeps every
standard whose duty matches the measured one and reports a verdict only if
it holds across all of them; the speed interval is the union.

Lateral spacing (edge-to-centre) is not part of these standards: the case
prior's value is kept.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from backend.geometry.road_geometry import RoadPrior

FT = 0.3048


@dataclass(frozen=True)
class MarkingStandard:
    key: str
    name: str
    dash_len_m: float
    dash_gap_m: float
    used_on: str
    source: str

    @property
    def cycle_m(self):
        return self.dash_len_m + self.dash_gap_m

    @property
    def duty(self):
        return self.dash_len_m / self.cycle_m


STANDARDS = [
    MarkingStandard("carai_synthetic", "CARAI synthetic 3 m + 9 m", 3.0, 9.0, "the simulator's test roads only",
                    "evaluation/sim/world.py (not a real-world standard)"),
    MarkingStandard("us_mutcd", "US MUTCD broken line 10 ft + 30 ft", 10 * FT, 30 * FT,
                    "US roads (federal default)", "MUTCD 2009, Section 3A.06"),
    MarkingStandard("caltrans_freeway", "Caltrans freeway 12 ft + 36 ft", 12 * FT, 36 * FT,
                    "California freeways", "docs/assumptions.md"),
    MarkingStandard("fr_t1", "France T1 3 m + 10 m", 3.0, 10.0,
                    "French rural roads: centre and lane lines; also in towns",
                    "Instruction interministérielle sur la signalisation routière, 7e partie, art. 113-1.B"),
    MarkingStandard("at_motorway", "Austria motorway 6 m + 12 m", 6.0, 12.0, "Austrian Autobahnen and Autostraßen",
                    "Bodenmarkierungsverordnung (BoMaVO) § 5 (1)"),
    MarkingStandard("at_rural", "Austria rural 6 m + 9 m", 6.0, 9.0, "other Austrian roads outside built-up areas",
                    "Bodenmarkierungsverordnung (BoMaVO) § 5 (1)"),
    MarkingStandard("at_urban", "Austria urban 3 m + 3 m", 3.0, 3.0, "Austrian built-up areas, approaches to junctions",
                    "Bodenmarkierungsverordnung (BoMaVO) § 5 (1)"),
]
BY_KEY = {s.key: s for s in STANDARDS}


def prior_for(standard: MarkingStandard, base: RoadPrior) -> RoadPrior:
    return replace(base, name=standard.key, dash_len_m=standard.dash_len_m, dash_gap_m=standard.dash_gap_m)


# The measured duty cycle is biased UP: blur and a background median that still holds some vehicle pixels
# lengthen dashes. On synthetic development clips with true duty 0.25 we measured +0.04 over a 6 s clip and
# +0.08 over 3 s of it. A standard is consistent with a measurement d if its duty lies in
# [d - tol - DUTY_INFLATION_MAX, d + tol].
DUTY_INFLATION_MAX = 0.08


def consistent_with_duty(duty: float, duty_measured: float, tol: float) -> bool:
    return duty_measured - tol - DUTY_INFLATION_MAX <= duty <= duty_measured + tol


def candidates_for_duty(duty_measured: float, tol: float, exclude: str = None):
    """Standards consistent with the measured duty cycle (allowing for its upward bias), closest first."""
    c = [s for s in STANDARDS if consistent_with_duty(s.duty, duty_measured, tol) and s.key != exclude]
    return sorted(c, key=lambda s: abs(s.duty - duty_measured))
