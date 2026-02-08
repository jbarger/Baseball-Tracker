"""
Camera calibration module using known pitching machine dimensions.

Uses the pitching machine visible in the frame as a real-world reference
to compute pixels-per-inch, enabling conversion from pixel measurements
to real-world distances and speeds.
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

    machine_bbox_px: [x1, y1, x2, y2] bounding box of the machine in pixels.
    machine_spec: known physical dimensions.
    use_height: if True, use machine height for calibration (more reliable
                since height is perpendicular to the ground plane).

    Returns pixels_per_inch at the machine's depth in the scene.
    """
    x1, y1, x2, y2 = machine_bbox_px
    box_width_px = abs(x2 - x1)
    box_height_px = abs(y2 - y1)

    if use_height:
        ppi = box_height_px / machine_spec.height_inches
    else:
        ppi = box_width_px / machine_spec.width_inches

    return ppi


def pixels_to_inches(pixel_distance: float, ppi: float) -> float:
    """Convert pixel distance to inches using calibration."""
    if ppi <= 0:
        return 0.0
    return pixel_distance / ppi


def pixels_to_feet(pixel_distance: float, ppi: float) -> float:
    """Convert pixel distance to feet using calibration."""
    return pixels_to_inches(pixel_distance, ppi) / 12.0


def pixel_speed_to_mph(pixels_per_second: float, ppi: float) -> float:
    """
    Convert pixel velocity to miles per hour.

    pixels/sec -> inches/sec -> feet/sec -> mph
    """
    if ppi <= 0:
        return 0.0
    inches_per_sec = pixels_per_second / ppi
    feet_per_sec = inches_per_sec / 12.0
    mph = feet_per_sec * 3600.0 / 5280.0
    return mph


def pixel_speed_to_fps(pixels_per_second: float, ppi: float) -> float:
    """Convert pixel velocity to feet per second."""
    if ppi <= 0:
        return 0.0
    inches_per_sec = pixels_per_second / ppi
    return inches_per_sec / 12.0


@dataclass
class CageCalibration:
    """
    Complete calibration for a batting cage camera setup.

    Holds the computed pixels-per-inch and machine reference info,
    plus convenience methods for converting measurements.
    """
    machine_spec: MachineSpec
    machine_distance_ft: float
    pixels_per_inch: Optional[float] = None

    @property
    def is_calibrated(self) -> bool:
        return self.pixels_per_inch is not None and self.pixels_per_inch > 0

    def to_mph(self, pixels_per_second: float) -> Optional[float]:
        """Convert pixel speed to mph. Returns None if not calibrated."""
        if not self.is_calibrated:
            return None
        return pixel_speed_to_mph(pixels_per_second, self.pixels_per_inch)

    def to_feet(self, pixel_distance: float) -> Optional[float]:
        """Convert pixel distance to feet. Returns None if not calibrated."""
        if not self.is_calibrated:
            return None
        return pixels_to_feet(pixel_distance, self.pixels_per_inch)

    def get_info_lines(self) -> list:
        """Return calibration status lines for HUD display."""
        lines = [
            f"Machine: {self.machine_spec.name}",
            f"Distance: {self.machine_distance_ft} ft",
        ]
        if self.is_calibrated:
            lines.append(f"Scale: {self.pixels_per_inch:.1f} px/in")
        else:
            lines.append("Calibration: NOT SET (pixel speed only)")
        return lines


def load_calibration(config_path: str, machines_path: str) -> CageCalibration:
    """
    Load cage calibration from config files.

    Returns a CageCalibration object. If machine_bbox_px is set in the config,
    computes pixels_per_inch automatically.
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

    # Auto-compute PPI if machine bounding box is configured
    cal_config = config.get("calibration", {})
    machine_bbox = cal_config.get("machine_bbox_px")
    if machine_bbox and len(machine_bbox) == 4:
        cal.pixels_per_inch = compute_pixels_per_inch(machine_bbox, spec)

    return cal
