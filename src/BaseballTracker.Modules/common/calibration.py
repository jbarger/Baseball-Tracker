"""
Camera calibration module for batting cage speed estimation.

Three calibration modes (in order of preference):
1. 3D-GEOMETRY (best): Full perspective model using known cage geometry
   to project pixel positions onto the pitch line and compute true 3D
   speed. Correctly handles depth-dominant trajectories where the ball
   moves mostly toward/away from the camera.
2. SIGN-SPEED (good): Uses the known pitch speed from the cage sign
   and measured pixel speed of the pitched ball to derive an empirical
   px/s -> mph conversion factor. Accounts for perspective on average.
3. MACHINE-BBOX (fallback): Uses the pitching machine's known dimensions
   to compute pixels-per-inch at the machine's depth.

The 3D geometry approach is strongly preferred because it correctly
handles the fundamental problem: a ball thrown at ~45 mph from 60ft
moves mostly in depth (Z-axis), producing only ~80px of lateral
motion in the image. The old flat-calibration factor amplifies any
tracking noise by ~200x when converting to mph.
"""
import json
import os
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field

from common.geometry3d import CageGeometry, CameraModel, PitchTrajectory


# Ball types with physical specs
BALL_TYPES = {
    "yellow_dimple": {
        "name": "Yellow Dimple Ball",
        "diameter_inches": 2.85,
        "notes": "Standard batting cage dimple ball (yellow). Slightly smaller than regulation.",
    },
    "white_regulation": {
        "name": "White Regulation Baseball",
        "diameter_inches": 2.9,
        "notes": "Standard MLB regulation baseball.",
    },
    "softball": {
        "name": "Softball",
        "diameter_inches": 3.82,
        "notes": "Standard 12-inch softball.",
    },
    "custom": {
        "name": "Custom",
        "diameter_inches": 2.9,
        "notes": "User-specified ball dimensions.",
    },
}


@dataclass
class MachineSpec:
    """Physical specs for a pitching machine model."""
    name: str
    height_inches: float
    width_inches: float
    length_inches: float
    recommended_distance_ft: float
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


def load_camera_model(camera_key: str, cameras_path: str) -> dict:
    """
    Load camera specs from the camera models database.

    Returns dict with sensor_size_mm, focal_length_mm, fov_degrees,
    distortion_coefficients, resolutions, etc.
    """
    with open(cameras_path, "r") as f:
        data = json.load(f)

    cameras = data.get("cameras", {})
    if camera_key not in cameras:
        available = ", ".join(cameras.keys())
        raise ValueError(
            f"Unknown camera model '{camera_key}'. Available: {available}"
        )
    return cameras[camera_key]


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

    Supports three calibration modes:
    - 3D geometry: full perspective model (preferred)
    - sign_speed_mph: empirical factor from measured pitch (good)
    - pixels_per_inch: from machine bbox measurement (fallback)
    """
    machine_spec: MachineSpec
    machine_distance_ft: float
    pixels_per_inch: Optional[float] = None
    sign_speed_mph: Optional[float] = None
    # Empirical factor: set after first calibration pass
    _empirical_factor: Optional[float] = None

    # 3D geometry model
    geometry: Optional[CageGeometry] = None

    # Ball type
    ball_type: str = "yellow_dimple"

    @property
    def ball_diameter_inches(self) -> float:
        """Diameter of the ball being tracked, in inches."""
        ball_info = BALL_TYPES.get(self.ball_type, BALL_TYPES["yellow_dimple"])
        return ball_info["diameter_inches"]

    @property
    def is_calibrated(self) -> bool:
        return (self.geometry is not None and self.geometry.is_calibrated or
                self._empirical_factor is not None or
                (self.pixels_per_inch is not None and self.pixels_per_inch > 0))

    @property
    def has_3d(self) -> bool:
        """Whether 3D geometry calibration is available."""
        return self.geometry is not None and self.geometry.is_calibrated

    @property
    def calibration_mode(self) -> str:
        if self.geometry is not None and self.geometry.is_calibrated:
            if self._empirical_factor is not None:
                return "3D-GEOMETRY + SIGN-SPEED"
            return "3D-GEOMETRY"
        if self._empirical_factor is not None:
            return "SIGN-SPEED"
        elif self.pixels_per_inch is not None and self.pixels_per_inch > 0:
            return "MACHINE-BBOX"
        return "UNCALIBRATED"

    def calibrate_from_pitch(self, pitch_px_per_sec: float):
        """
        Set the empirical conversion factor from a measured pitch speed.
        Still used as a fallback/comparison alongside 3D geometry.
        """
        if self.sign_speed_mph and pitch_px_per_sec > 0:
            self._empirical_factor = self.sign_speed_mph / pitch_px_per_sec

    def to_mph(self, pixels_per_second: float) -> Optional[float]:
        """
        Convert pixel speed to mph using the best available calibration.

        NOTE: This is the FLAT (2D) conversion. For accurate speed on
        depth-dominant trajectories, use to_mph_3d() with pixel positions.
        """
        if self._empirical_factor is not None:
            return pixels_per_second * self._empirical_factor

        if self.pixels_per_inch is not None and self.pixels_per_inch > 0:
            inches_per_sec = pixels_per_second / self.pixels_per_inch
            feet_per_sec = inches_per_sec / 12.0
            return feet_per_sec * 3600.0 / 5280.0

        return None

    def to_mph_3d(self, u1: float, v1: float, u2: float, v2: float,
                  dt_seconds: float) -> Optional[float]:
        """
        Convert two pixel positions to true 3D speed on the pitch line.

        This is the correct method for pitched balls that move mostly
        in depth. Falls back to flat calibration if 3D isn't available.
        """
        if self.has_3d and dt_seconds > 0:
            speed = self.geometry.pixel_speed_to_3d_speed_on_pitch_line(
                u1, v1, u2, v2, dt_seconds
            )
            if speed is not None:
                return speed

        # Fallback to flat calibration
        if dt_seconds > 0:
            dx = u2 - u1
            dy = v2 - v1
            px_per_sec = np.sqrt(dx * dx + dy * dy) / dt_seconds
            return self.to_mph(px_per_sec)

        return None

    def get_world_position(self, u: float, v: float
                           ) -> Optional[Tuple[float, float, float]]:
        """
        Get the 3D world position for a pixel point on the pitch line.
        Returns (x, y, z) in feet, or None if not calibrated.
        """
        if self.has_3d:
            return self.geometry.pixel_to_world_on_pitch_line(u, v)
        return None

    def get_depth_ft(self, u: float, v: float) -> Optional[float]:
        """Get depth (Z distance from home plate) for a pixel on the pitch line."""
        if self.has_3d:
            return self.geometry.pixel_to_depth_on_pitch_line(u, v)
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

        if self.has_3d:
            lines.append(f"Cal: 3D-GEOMETRY")
            geo_lines = self.geometry.get_info_lines()
            lines.extend(geo_lines)
        elif self._empirical_factor is not None:
            lines.append(f"Cal: SIGN-SPEED ({self.sign_speed_mph} mph ref)")
        elif self.pixels_per_inch is not None:
            lines.append(f"Cal: MACHINE-BBOX ({self.pixels_per_inch:.1f} px/in)")
        else:
            lines.append("Cal: NOT SET (pixel speed only)")

        return lines

    def export_trajectory_3d(self, pixel_positions: List[Tuple[float, float]],
                              frame_indices: List[int], fps: float) -> List[dict]:
        """
        Export a trajectory as 3D data for visualization.

        Returns JSON-serializable list of trajectory points with
        pixel, world, speed, and velocity data.
        """
        if self.has_3d:
            return self.geometry.export_trajectory_3d(
                pixel_positions, frame_indices, fps
            )
        # Fallback: pixel-only data
        trajectory = []
        for pos, frame_idx in zip(pixel_positions, frame_indices):
            trajectory.append({
                "frame": frame_idx,
                "time_s": round(frame_idx / fps, 4),
                "pixel": {"u": round(pos[0], 1), "v": round(pos[1], 1)},
                "world_ft": None,
                "depth_ft": None,
                "speed_mph": None,
                "velocity_ft_s": None,
            })
        return trajectory


def load_calibration(config_path: str, machines_path: str,
                     cameras_path: str = None) -> CageCalibration:
    """
    Load cage calibration from config files.

    Sets up 3D geometry model with camera specs (lens distortion, focal
    length from physical specs) in addition to the legacy calibration.

    If cameras_path is not provided, looks for camera_models.json in the
    same directory as config_path.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    config_dir = os.path.dirname(config_path)
    if cameras_path is None:
        cameras_path = os.path.join(config_dir, "camera_models.json")

    machine_type = config["machine_type"]
    machine_distance = config["machine_distance_ft"]
    spec = load_machine_spec(machine_type, machines_path)

    # Ball type
    ball_type = config.get("ball_type", "yellow_dimple")

    cal = CageCalibration(
        machine_spec=spec,
        machine_distance_ft=machine_distance,
        ball_type=ball_type,
    )

    cal_config = config.get("calibration", {})

    # Sign speed calibration
    sign_speed = cal_config.get("known_pitch_speed_mph")
    if sign_speed:
        cal.sign_speed_mph = float(sign_speed)

    # Machine bbox PPI (fallback / supplementary)
    machine_bbox = cal_config.get("machine_bbox_px")
    if machine_bbox and len(machine_bbox) == 4:
        cal.pixels_per_inch = compute_pixels_per_inch(machine_bbox, spec)

    # --- Camera Model ---
    camera_model_key = config.get("camera_model")
    camera_specs = None
    if camera_model_key and os.path.exists(cameras_path):
        try:
            camera_specs = load_camera_model(camera_model_key, cameras_path)
            print(f"\n--- Camera Model: {camera_specs['name']} ---")
        except (ValueError, FileNotFoundError) as e:
            print(f"  Warning: Could not load camera model: {e}")

    # --- 3D Geometry Model ---
    cage_geo_config = config.get("cage_geometry", {})

    # Camera parameters
    camera_height = cage_geo_config.get("camera_height_ft", 5.0)
    camera_behind_plate = cage_geo_config.get("camera_behind_plate_ft", 2.0)
    release_height = cage_geo_config.get("release_height_ft", 3.5)
    strike_zone_height = cage_geo_config.get("strike_zone_height_ft", 2.5)
    cage_length = cage_geo_config.get("cage_length_ft", 70.0)
    cage_width = cage_geo_config.get("cage_width_ft", 14.0)
    cage_height = cage_geo_config.get("cage_height_ft", 12.0)

    # Video dimensions
    video_width = cage_geo_config.get("video_width", 1920)
    video_height = cage_geo_config.get("video_height", 1080)

    # Check for auto-calibration overrides in cage_geometry
    # (saved by auto_calibrate.py after optimization)
    has_auto_cal = "focal_length_px" in cage_geo_config

    # Build camera model with specs if available
    camera = CameraModel(
        image_width=video_width,
        image_height=video_height,
    )

    if camera_specs:
        # Load distortion coefficients from camera specs as defaults
        dist = camera_specs.get("distortion_coefficients", {})
        camera.k1 = dist.get("k1", 0.0)
        camera.k2 = dist.get("k2", 0.0)
        camera.k3 = dist.get("k3", 0.0)
        camera.p1 = dist.get("p1", 0.0)
        camera.p2 = dist.get("p2", 0.0)

        # Compute focal length from physical specs
        sensor = camera_specs.get("sensor_size_mm")
        focal_mm = camera_specs.get("focal_length_mm")
        if sensor and focal_mm and sensor[0] > 0:
            camera.calibrate_from_camera_specs(
                sensor_width_mm=sensor[0],
                sensor_height_mm=sensor[1],
                focal_length_mm=focal_mm,
                image_width=video_width,
                image_height=video_height,
            )
            print(f"  Focal length from specs: {camera.focal_length_px:.1f} px "
                  f"({focal_mm}mm on {sensor[0]}x{sensor[1]}mm sensor)")

        if camera.has_distortion:
            print(f"  Distortion: k1={camera.k1}, k2={camera.k2}, "
                  f"k3={camera.k3}, p1={camera.p1}, p2={camera.p2}")

    # Apply auto-calibration overrides (these take priority over camera specs)
    if has_auto_cal:
        camera.focal_length_px = cage_geo_config["focal_length_px"]
        if "k1" in cage_geo_config:
            camera.k1 = cage_geo_config["k1"]
        if "k2" in cage_geo_config:
            camera.k2 = cage_geo_config["k2"]
        meta = cage_geo_config.get("calibration_meta", {})
        print(f"\n--- Auto-Calibration Override ---")
        print(f"  Focal length: {camera.focal_length_px:.1f} px (optimized)")
        if camera.has_distortion:
            print(f"  Distortion: k1={camera.k1:.5f}, k2={camera.k2:.5f}")
        if meta:
            print(f"  Method: {meta.get('method', 'unknown')}")
            print(f"  Error: {meta.get('mean_speed_error_pct', '?')}%")

    # Camera position from config (may be overridden by auto-calibration)
    camera_x = cage_geo_config.get("camera_x_ft", 0.0)

    if machine_bbox and len(machine_bbox) == 4:
        geometry = CageGeometry(
            camera=camera,
            cage_length_ft=cage_length,
            cage_width_ft=cage_width,
            cage_height_ft=cage_height,
        )

        if has_auto_cal:
            # Auto-calibrated: use optimized params directly
            geometry.camera.camera_x_ft = camera_x
            geometry.camera.camera_y_ft = camera_height
            geometry.camera.camera_z_ft = -camera_behind_plate

            # Use saved aim offsets if present
            if "aim_offset_x" in cage_geo_config:
                geometry.camera.aim_offset_x = cage_geo_config["aim_offset_x"]
            if "aim_offset_y" in cage_geo_config:
                geometry.camera.aim_offset_y = cage_geo_config["aim_offset_y"]

            # Set up pitch trajectory
            geometry.pitch = PitchTrajectory(
                release_x_ft=0.0,
                release_y_ft=release_height,
                release_z_ft=machine_distance,
                target_x_ft=0.0,
                target_y_ft=strike_zone_height,
                target_z_ft=0.0,
            )
            if cal.sign_speed_mph and cal.sign_speed_mph > 0:
                from common.geometry3d import FEET_PER_MILE, SECONDS_PER_HOUR
                geometry.pitch.speed_ft_per_sec = (
                    cal.sign_speed_mph * FEET_PER_MILE / SECONDS_PER_HOUR
                )
            geometry._calibrated = True

        elif camera.is_calibrated:
            # Camera specs gave us focal length — set up geometry without
            # re-deriving focal length from bbox
            geometry.camera.camera_x_ft = camera_x
            geometry.camera.camera_y_ft = camera_height
            geometry.camera.camera_z_ft = -camera_behind_plate

            # Derive aim offset from machine bbox position
            mx1, my1, mx2, my2 = machine_bbox
            machine_center_u = (mx1 + mx2) / 2.0
            machine_center_v = (my1 + my2) / 2.0

            projected = geometry.camera.world_to_pixel(
                0.0,
                (release_height + 0.0) / 2.0,
                machine_distance
            )
            if projected is not None:
                geometry.camera.aim_offset_x = machine_center_u - projected[0]
                geometry.camera.aim_offset_y = machine_center_v - projected[1]

            # Set up pitch trajectory
            geometry.pitch = PitchTrajectory(
                release_x_ft=0.0,
                release_y_ft=release_height,
                release_z_ft=machine_distance,
                target_x_ft=0.0,
                target_y_ft=strike_zone_height,
                target_z_ft=0.0,
            )
            if cal.sign_speed_mph and cal.sign_speed_mph > 0:
                from common.geometry3d import FEET_PER_MILE, SECONDS_PER_HOUR
                geometry.pitch.speed_ft_per_sec = (
                    cal.sign_speed_mph * FEET_PER_MILE / SECONDS_PER_HOUR
                )
            geometry._calibrated = True

            # Validate: compare bbox-derived focal length with specs
            bbox_focal = (abs(my2 - my1) * machine_distance * 12.0) / spec.height_inches
            print(f"  Focal length from bbox: {bbox_focal:.1f} px "
                  f"(validation, diff: {abs(camera.focal_length_px - bbox_focal):.1f}px)")
        else:
            # No camera specs — derive focal length from machine bbox (legacy)
            geometry.calibrate(
                machine_bbox_px=machine_bbox,
                machine_height_inches=spec.height_inches,
                machine_distance_ft=machine_distance,
                sign_speed_mph=cal.sign_speed_mph,
                camera_height_ft=camera_height,
                camera_behind_plate_ft=camera_behind_plate,
                release_height_ft=release_height,
                strike_zone_height_ft=strike_zone_height,
            )

        cal.geometry = geometry

        # Print 3D calibration diagnostics
        print(f"\n--- 3D Geometry Calibration ---")
        print(f"  Focal length: {geometry.camera.focal_length_px:.0f} px")
        if geometry.camera.has_distortion:
            print(f"  Lens distortion: k1={geometry.camera.k1:.3f}, k2={geometry.camera.k2:.3f}")
        print(f"  Camera pos: ({geometry.camera.camera_x_ft:.1f}, "
              f"{geometry.camera.camera_y_ft:.1f}, "
              f"{geometry.camera.camera_z_ft:.1f}) ft")
        print(f"  Aim offset: ({geometry.camera.aim_offset_x:.1f}, "
              f"{geometry.camera.aim_offset_y:.1f}) px")
        print(f"  Pitch line: ({geometry.pitch.release_x_ft:.1f}, "
              f"{geometry.pitch.release_y_ft:.1f}, "
              f"{geometry.pitch.release_z_ft:.1f}) -> "
              f"({geometry.pitch.target_x_ft:.1f}, "
              f"{geometry.pitch.target_y_ft:.1f}, "
              f"{geometry.pitch.target_z_ft:.1f})")
        print(f"  Ball type: {ball_type} ({cal.ball_diameter_inches}\" dia)")
        if geometry.pitch.speed_ft_per_sec > 0:
            print(f"  Pitch speed: {cal.sign_speed_mph} mph = "
                  f"{geometry.pitch.speed_ft_per_sec:.1f} ft/s")
            print(f"  Flight time: {geometry.pitch.flight_time_sec:.3f} s "
                  f"({geometry.pitch.flight_time_sec * 59.94:.1f} frames)")

        # Validation: expected pixel velocities at key depths
        expected_vel = geometry.get_expected_pixel_velocity(
            machine_distance, cal.sign_speed_mph or 47.5
        )
        if expected_vel:
            print(f"  Expected px speed at mound: "
                  f"{np.sqrt(expected_vel[0]**2 + expected_vel[1]**2):.0f} px/s")

        expected_vel_mid = geometry.get_expected_pixel_velocity(
            machine_distance / 2, cal.sign_speed_mph or 47.5
        )
        if expected_vel_mid:
            print(f"  Expected px speed at {machine_distance/2:.0f}ft: "
                  f"{np.sqrt(expected_vel_mid[0]**2 + expected_vel_mid[1]**2):.0f} px/s")

        expected_vel_near = geometry.get_expected_pixel_velocity(
            10.0, cal.sign_speed_mph or 47.5
        )
        if expected_vel_near:
            print(f"  Expected px speed at 10ft: "
                  f"{np.sqrt(expected_vel_near[0]**2 + expected_vel_near[1]**2):.0f} px/s")

    return cal
