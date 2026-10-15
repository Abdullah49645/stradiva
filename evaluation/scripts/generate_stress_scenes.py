"""
Deliberate-failure dataset (brief section 26): scenarios that SHOULD be
rejected or downgraded, plus one non-baseline scenario that should still
succeed, to check the reconstructability gate isn't just tuned to pass
scene A specifically.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from scene_generator import generate_scene

if __name__ == "__main__":
    generate_scene("B_no_scale_reference", include_dashes=False)       # no dashed centerline -> no scale anchor at all
    generate_scene("C_camera_roll", roll_deg=8.0)                      # violates zero-roll assumption
    generate_scene("D_heavy_fog", fog=True)                            # severe visibility loss
    generate_scene("E_faster_vehicle", speed_mps=32.0)                 # sanity: different true speed, same geometry
