"""Validation of the inputs a user may supply with a clip (the investigator's requests)."""
from backend.geometry.marking_standards import BY_KEY

LIMITS = {"fps": (1.0, 240.0), "focal_px": (100.0, 10000.0)}


def validate_inputs(raw):
    """Returns (clean dict, error message or None). Unknown keys are an error, not ignored."""
    if raw in (None, {}):
        return {}, None
    if not isinstance(raw, dict):
        return {}, "inputs must be an object"
    clean = {}
    for k, v in raw.items():
        if v in (None, ""):
            continue
        if k in LIMITS:
            try:
                x = float(v)
            except (TypeError, ValueError):
                return {}, f"{k} must be a number"
            lo, hi = LIMITS[k]
            if not lo <= x <= hi:
                return {}, f"{k} must be between {lo:g} and {hi:g}"
            clean[k] = x
        elif k == "marking_standard":
            if v not in BY_KEY:
                return {}, f"unknown marking_standard '{v}' (known: {', '.join(sorted(BY_KEY))})"
            clean[k] = v
        else:
            return {}, f"unknown input '{k}' (accepted: fps, focal_px, marking_standard)"
    return clean, None
