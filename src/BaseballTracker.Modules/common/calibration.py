"""
Camera calibration module for batting cage speed estimation.

Two calibration modes:
1. SIGN-SPEED (preferred): Uses the known pitch speed from the cage sign
   and measured pixel speed of the pitched ball to derive an empirical
   px/s → mph conversion factor. Accounts for perspective automatically.
2. MACHINE-BBOX (fallback): Uses the pitching machine's known dimensions
   to compute pixels-per-inch at the machine's depth. Less accurate for
   ball speed since the ball traverses varying depths.

The sign-speed approach is preferred because it inherently accounts for
camera perspective, lens distortion, and the fact that the ball's pixel
velocity changes as it approaches the camera.
"""
import json
import os
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MachineSpec:
    """Physical specs for a pitching machine model."""
    name: str
    height_inches: float
    width_inches: float
    length_inches: float
    recommended_distance_ft: float  # typical distance from home plate
    speed_range_mph: Tuple[float, float]


def load_machines_db(machines_path: str) -> Dict[str, MachineSpec]:
    """Load the pitching machine database from JSON."""
    with open(machines_path, "r") as f:
        data = json.load(f)

    machines = {}
    for key, m in data["machines"].items():
        dims = m["dimensions_inches"]
        dist = m["recommended_distance_ft"]
        speed = m["speed_range_mph"]
        machines[key] = MachineSpec(
            name=m["name"],
            height_inches=dims["height"],
            width_inches=dims["width"],
            length_inches=dims["length"],
            recommended_distance_ft=dist["typical"],
            speed_range_mph=(speed["min"], speed["max"]),
        )
    return machines


def load_machine_spec(machine_type: str, machines_path: str) -> MachineSpec:
    """Load a specific machine's specs from the database."""
    machines = load_machines_db(machines_path)
    if machine_type not in machines:
        available = ", ".join(machines.keys())
        raise ValueError(
            f"Unknown machine type '{machine_type}'. Available: {available}"
        )
    return machines[machine_type]


def compute_pixels_per_inch(machine_bbox_px: list, machine_spec: MachineSpec,
                            use_height: bool = True) -> float:
    """
    Compute pixels-per-inch from a bounding box of the machine in the frame.
    NOTE: This is the scale at the MACHINE'S depth only.
    """
    x1, y1, x2, y2 = machine_bbox_px
    box_width_px = abs(x2 - x1)
    box_height_px = abs(y2 - y1)

    if use_height:
        ppi = box_height_px / machine_spec.height_inches
    else:
        ppi = box_width_px / machine_spec.width_inches

    return ppi


@dataclass
class CageCalibration:
    """
    Complete calibration for a batting cage camera setup.

    Supports two calibration modes:
    - sign_speed_mph: known pitch speed from cage sign (preferred)
    - pixels_per_inch: from machine bbox measurement (fallback)

    When sign_speed_mph is set, the to_mph() method uses a first-pass
    approach: it returns the sign speed for the pitched ball, and uses
    the empirical conversion factor (derived from calibration pass) for
    other measurements.
    """
    machine_spec: MachineSpec
    machine_distance_ft: float
    pixels_per_inch: Optional[float] = None
    sign_speed_mph: Optional[float] = None
    # Empirical factor: set after first calibration pass
    # mph = px_per_sec * empirical_factor
    _empirical_factor: Optional[float] = None

    @property
    def is_calibrated(self) -> bool:
        return (self._empirical_factor is not None or
                (self.pixels_per_inch is not None and self.pixels_per_inch > 0))

    @property
    def calibration_mode(self) -> str:
        if self._empirical_factor is not None:
            return "SIGN-SPEED"
        elif self.pixels_per_inch is not None and self.pixels_per_inch > 0:
            return "MACHINE-BBOX"
        return "UNCALIBRATED"

    def calibrate_from_pitch(self, pitch_px_per_sec: float):
        """
        Set the empirical conversion factor from a measured pitch speed.

        Call this after the first pass identifies the pitched ball's
        average pixel speed. Combined with the sign speed, this gives us:
            factor = sign_speed_mph / pitch_px_per_sec
            mph = px_per_sec * factor

        This factor inherently accounts for camera perspective because
        it's derived from a real measurement at a known speed.
        """
        if self.sign_speed_mph and pitch_px_per_sec > 0:
            self._empirical_factor = self.sign_speed_mph / pitch_px_per_sec

    def to_mph(self, pixels_per_second: float) -> Optional[float]:
        """
        Convert pixel speed to mph.

        If empirical factor is set (sign-speed calibration), use it.
        Otherwise fall back to machine-bbox PPI method.
        Returns None if not calibrated.
        """
        if self._empirical_factor is not None:
            return pixels_per_second * self._empirical_factor

        if self.pixels_per_inch is not None and self.pixels_per_inch > 0:
            inches_per_sec = pixels_per_second / self.pixels_per_inch
            feet_per_sec = inches_per_sec / 12.0
            return feet_per_sec * 3600.0 / 5280.0

        return None

    def to_feet(self, pixel_distance: float) -> Optional[float]:
        """Convert pixel distance to feet. Uses PPI if available."""
        if self.pixels_per_inch is not None and self.pixels_per_inch > 0:
            return (pixel_distance / self.pixels_per_inch) / 12.0
        return None

    def get_info_lines(self) -> list:
        """Return calibration status lines for HUD display."""
        lines = [
            f"Machine: {self.machine_spec.name}",
            f"Mound: {self.machine_distance_ft} ft",
        ]
        if self._empirical_factor is not None:
            lines.append(f"Cal: SIGN-SPEED ({self.sign_speed_mph} mph ref)")
        elif self.pixels_per_inch is not None:
            lines.append(f"Cal: MACHINE-BBOX ({self.pixels_per_inch:.1f} px/in)")
        else:
            lines.append("Cal: NOT SET (pixel speed only)")
        return lines


def load_calibration(config_path: str, machines_path: str) -> CageCalibration:
    """
    Load cage calibration from config files.

    If known_pitch_speed_mph is set, uses sign-speed calibration mode.
    Otherwise falls back to machine-bbox PPI.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    machine_type = config["machine_type"]
    machine_distance = config["machine_distance_ft"]
    spec = load_machine_spec(machine_type, machines_path)

    cal = CageCalibration(
        machine_spec=spec,
        machine_distance_ft=machine_distance,
    )

    cal_config = config.get("calibration", {})

    # Sign speed calibration (preferred)
    sign_speed = cal_config.get("known_pitch_speed_mph")
    if sign_speed:
        cal.sign_speed_mph = float(sign_speed)

    # Machine bbox PPI (fallback / supplementary)
    machine_bbox = cal_config.get("machine_bbox_px")
    if machine_bbox and len(machine_bbox) == 4:
        cal.pixels_per_inch = compute_pixels_per_inch(machine_bbox, spec)

    return cal
