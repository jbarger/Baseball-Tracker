"""
3D geometry model for batting cage camera setup.

Models the physical geometry of a batting cage to convert between
2D pixel coordinates and 3D world positions. This enables:
  - Accurate speed measurement for balls moving toward/away from camera
  - Correct trajectory reconstruction for any ball direction
  - Foundation for future 3D visualization

Coordinate system (right-handed, baseball convention):
  - Origin: home plate, ground level
  - X: toward first base (camera's right in typical setup)
  - Y: vertical (up)
  - Z: toward pitcher's mound / outfield (away from batter)

Camera model: pinhole projection with known intrinsics derived from
the cage geometry and a few measurable reference points (machine bbox,
cage dimensions, camera position).

Key insight: a single camera can't fully resolve 3D from a single
point detection. But we KNOW the ball travels along a constrained
path (pitched ball: roughly straight from machine to plate). By
combining:
  1. Pixel (u, v) position
  2. Known ball diameter (2.9 inches)
  3. Known pitch trajectory (machine → plate)
  4. Apparent ball size as depth cue
We can estimate depth (Z) and get full 3D position.
"""
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


# Physical constants
BASEBALL_DIAMETER_INCHES = 2.9
BASEBALL_RADIUS_INCHES = BASEBALL_DIAMETER_INCHES / 2.0
INCHES_PER_FOOT = 12.0
FEET_PER_MILE = 5280.0
SECONDS_PER_HOUR = 3600.0


@dataclass
class CameraModel:
    """
    Pinhole camera model with Brown-Conrady lens distortion.

    The camera is assumed to be mounted behind/above home plate,
    looking toward the pitching machine. We derive the focal length
    and camera position from known reference measurements.

    Intrinsic parameters:
      - focal_length_px: effective focal length in pixels
      - cx, cy: principal point (image center)
      - k1, k2, k3: radial distortion coefficients
      - p1, p2: tangential distortion coefficients

    Extrinsic parameters (camera position in world coords):
      - camera_pos: (x, y, z) in feet, world coordinates
      - The camera looks along +Z (toward the mound)
    """
    # Image dimensions
    image_width: int = 1920
    image_height: int = 1080

    # Intrinsic: focal length in pixels (derived from calibration)
    focal_length_px: float = 0.0

    # Principal point (usually image center)
    cx: float = 0.0
    cy: float = 0.0

    # Brown-Conrady distortion coefficients
    # k1, k2, k3: radial distortion (barrel when k1 < 0)
    # p1, p2: tangential distortion (decentering)
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    # Camera position in world coordinates (feet)
    # Default: behind home plate, ~5ft high, ~2ft behind plate
    camera_x_ft: float = 0.0     # centered laterally
    camera_y_ft: float = 5.0     # 5 feet high
    camera_z_ft: float = -2.0    # 2 feet behind home plate

    # Camera optical axis offset: where the camera points relative
    # to straight down the Z axis. In pixels from principal point.
    # (accounts for camera not perfectly centered on the pitch line)
    aim_offset_x: float = 0.0
    aim_offset_y: float = 0.0

    def __post_init__(self):
        if self.cx == 0:
            self.cx = self.image_width / 2.0
        if self.cy == 0:
            self.cy = self.image_height / 2.0

    @property
    def has_distortion(self) -> bool:
        return (self.k1 != 0.0 or self.k2 != 0.0 or self.k3 != 0.0
                or self.p1 != 0.0 or self.p2 != 0.0)

    def calibrate_from_camera_specs(self, sensor_width_mm: float,
                                     sensor_height_mm: float,
                                     focal_length_mm: float,
                                     image_width: int = 0,
                                     image_height: int = 0):
        """
        Compute focal_length_px from physical camera specifications.

        focal_length_px = focal_length_mm * image_width_px / sensor_width_mm

        This is more accurate than deriving focal length from a single
        reference object, especially for wide-angle lenses.
        """
        if image_width > 0:
            self.image_width = image_width
        if image_height > 0:
            self.image_height = image_height

        if sensor_width_mm > 0 and focal_length_mm > 0:
            self.focal_length_px = focal_length_mm * self.image_width / sensor_width_mm
            self.cx = self.image_width / 2.0
            self.cy = self.image_height / 2.0

    def undistort_point(self, u: float, v: float) -> Tuple[float, float]:
        """
        Remove lens distortion from raw pixel coordinates.

        Converts distorted pixel (u, v) to ideal undistorted pixel (u', v')
        using iterative Newton's method to invert the Brown-Conrady model.

        This is called before pixel_to_ray() so the ray computation
        uses corrected (rectilinear) coordinates.
        """
        if not self.has_distortion or self.focal_length_px <= 0:
            return (u, v)

        # Normalize to camera coordinates (centered, scaled by focal length)
        x_d = (u - self.cx) / self.focal_length_px
        y_d = (v - self.cy) / self.focal_length_px

        # Iterative undistortion: start from distorted point, solve for
        # undistorted point that maps back to it. Converges in 5-10 iters.
        x_u = x_d
        y_u = y_d

        for _ in range(20):
            r2 = x_u * x_u + y_u * y_u
            r4 = r2 * r2
            r6 = r4 * r2

            # Radial distortion factor
            radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6

            # Tangential distortion
            dx_t = 2.0 * self.p1 * x_u * y_u + self.p2 * (r2 + 2.0 * x_u * x_u)
            dy_t = self.p1 * (r2 + 2.0 * y_u * y_u) + 2.0 * self.p2 * x_u * y_u

            # The distorted point should equal: undistorted * radial + tangential
            # So: x_d = x_u * radial + dx_t  =>  x_u = (x_d - dx_t) / radial
            x_u = (x_d - dx_t) / radial
            y_u = (y_d - dy_t) / radial

        # Convert back to pixel coordinates
        u_out = x_u * self.focal_length_px + self.cx
        v_out = y_u * self.focal_length_px + self.cy

        return (u_out, v_out)

    def distort_point(self, u: float, v: float) -> Tuple[float, float]:
        """
        Apply lens distortion to ideal (undistorted) pixel coordinates.

        Converts undistorted pixel (u, v) to distorted pixel (u', v')
        using the forward Brown-Conrady model. Used when projecting
        world points back to raw image pixels (e.g., for overlay drawing).
        """
        if not self.has_distortion or self.focal_length_px <= 0:
            return (u, v)

        # Normalize to camera coordinates
        x_u = (u - self.cx) / self.focal_length_px
        y_u = (v - self.cy) / self.focal_length_px

        r2 = x_u * x_u + y_u * y_u
        r4 = r2 * r2
        r6 = r4 * r2

        # Radial distortion
        radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6

        # Tangential distortion
        dx_t = 2.0 * self.p1 * x_u * y_u + self.p2 * (r2 + 2.0 * x_u * x_u)
        dy_t = self.p1 * (r2 + 2.0 * y_u * y_u) + 2.0 * self.p2 * x_u * y_u

        # Apply distortion
        x_d = x_u * radial + dx_t
        y_d = y_u * radial + dy_t

        # Convert back to pixel coordinates
        u_out = x_d * self.focal_length_px + self.cx
        v_out = y_d * self.focal_length_px + self.cy

        return (u_out, v_out)

    @property
    def is_calibrated(self) -> bool:
        return self.focal_length_px > 0

    def calibrate_from_known_object(self, bbox_px: List[int],
                                     object_height_inches: float,
                                     object_distance_ft: float):
        """
        Derive focal length from a known object (e.g., pitching machine)
        at a known distance.

        focal_length = (pixel_height * real_distance) / real_height

        If distortion coefficients are set, undistorts the bbox corners
        first so the focal length is computed in undistorted (rectilinear)
        space. All measurements converted to consistent units (inches).
        """
        x1, y1, x2, y2 = bbox_px

        # Undistort bbox corners if we have distortion coefficients
        if self.has_distortion and self.focal_length_px > 0:
            _, y1_u = self.undistort_point(float(x1), float(y1))
            _, y2_u = self.undistort_point(float(x2), float(y2))
            pixel_height = abs(y2_u - y1_u)
        else:
            pixel_height = abs(y2 - y1)

        if pixel_height <= 0:
            return

        distance_inches = object_distance_ft * INCHES_PER_FOOT
        # focal_length_px = pixel_size * distance / real_size
        self.focal_length_px = (pixel_height * distance_inches) / object_height_inches

    def world_to_pixel(self, world_x_ft: float, world_y_ft: float,
                       world_z_ft: float) -> Optional[Tuple[float, float]]:
        """
        Project a 3D world point to 2D pixel coordinates.

        Uses pinhole model then applies lens distortion:
          1. Compute ideal (undistorted) pixel position
          2. Apply Brown-Conrady distortion to get raw pixel coords

        Returns (u, v) pixel coordinates or None if behind camera.
        """
        if not self.is_calibrated:
            return None

        # Convert to inches for projection (camera model in inches)
        dx = (world_x_ft - self.camera_x_ft) * INCHES_PER_FOOT
        dy = (world_y_ft - self.camera_y_ft) * INCHES_PER_FOOT
        dz = (world_z_ft - self.camera_z_ft) * INCHES_PER_FOOT

        if dz <= 0:
            return None  # Behind camera

        # Ideal (undistorted) pixel position
        u = self.focal_length_px * dx / dz + self.cx + self.aim_offset_x
        v = self.focal_length_px * (-dy) / dz + self.cy + self.aim_offset_y

        # Apply lens distortion to match raw image coordinates
        u, v = self.distort_point(u, v)

        return (u, v)

    def pixel_to_ray(self, u: float, v: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel coordinates to a 3D ray from the camera.

        First removes lens distortion from the raw pixel coordinates,
        then computes the ray in undistorted (rectilinear) space.

        Returns (origin, direction) where:
          - origin is camera position in feet
          - direction is unit vector in world coordinates
        """
        if not self.is_calibrated:
            origin = np.array([self.camera_x_ft, self.camera_y_ft, self.camera_z_ft])
            direction = np.array([0.0, 0.0, 1.0])
            return origin, direction

        # Remove lens distortion to get ideal pixel coordinates
        u_undist, v_undist = self.undistort_point(u, v)

        # Undistorted pixel to normalized camera coordinates
        nx = (u_undist - self.cx - self.aim_offset_x) / self.focal_length_px
        ny = -(v_undist - self.cy - self.aim_offset_y) / self.focal_length_px
        nz = 1.0  # forward

        direction = np.array([nx, ny, nz], dtype=np.float64)
        direction = direction / np.linalg.norm(direction)

        origin = np.array([self.camera_x_ft, self.camera_y_ft, self.camera_z_ft])
        return origin, direction

    def depth_from_apparent_size(self, apparent_diameter_px: float,
                                  real_diameter_inches: float = BASEBALL_DIAMETER_INCHES
                                  ) -> Optional[float]:
        """
        Estimate depth (distance from camera along Z) from apparent object size.

        depth = focal_length * real_size / pixel_size

        Returns depth in feet, or None if not calibrated.
        """
        if not self.is_calibrated or apparent_diameter_px <= 0:
            return None

        depth_inches = self.focal_length_px * real_diameter_inches / apparent_diameter_px
        return depth_inches / INCHES_PER_FOOT

    def apparent_size_at_depth(self, depth_ft: float,
                                real_diameter_inches: float = BASEBALL_DIAMETER_INCHES
                                ) -> Optional[float]:
        """
        Predict how large an object appears at a given depth.

        pixel_size = focal_length * real_size / depth

        Returns apparent diameter in pixels, or None if not calibrated.
        """
        if not self.is_calibrated or depth_ft <= 0:
            return None

        depth_inches = depth_ft * INCHES_PER_FOOT
        return self.focal_length_px * real_diameter_inches / depth_inches

    def pixel_to_world_on_plane(self, u: float, v: float,
                                 plane_z_ft: float) -> Optional[Tuple[float, float, float]]:
        """
        Intersect a pixel ray with a horizontal plane at a given Z depth.

        This gives the world (X, Y) at a known depth — useful for
        estimating position along the pitch trajectory.

        Returns (world_x_ft, world_y_ft, plane_z_ft) or None.
        """
        origin, direction = self.pixel_to_ray(u, v)

        if abs(direction[2]) < 1e-10:
            return None  # Ray parallel to plane

        # Solve: origin_z + t * dir_z = plane_z
        t = (plane_z_ft - origin[2]) / direction[2]
        if t < 0:
            return None  # Behind camera

        world_x = origin[0] + t * direction[0]
        world_y = origin[1] + t * direction[1]

        return (world_x, world_y, plane_z_ft)


@dataclass
class PitchTrajectory:
    """
    Models the expected path of a pitched ball.

    The ball leaves the machine at a known position (mound) and
    arrives at home plate. The trajectory is approximately linear
    at batting cage speeds (no significant curve for Iron Mike).

    Used to constrain depth estimation: if we know the ball is on
    the pitch line, we can resolve the depth ambiguity.
    """
    # Machine release point in world coordinates (feet)
    release_x_ft: float = 0.0
    release_y_ft: float = 3.5    # ~3.5ft above ground (machine exit height)
    release_z_ft: float = 60.0   # mound distance

    # Target point (home plate, strike zone center)
    target_x_ft: float = 0.0
    target_y_ft: float = 2.5     # ~2.5ft above ground (mid strike zone)
    target_z_ft: float = 0.0

    # Ball speed in ft/s (derived from sign speed)
    speed_ft_per_sec: float = 0.0

    @property
    def direction(self) -> np.ndarray:
        """Unit vector from release to target."""
        d = np.array([
            self.target_x_ft - self.release_x_ft,
            self.target_y_ft - self.release_y_ft,
            self.target_z_ft - self.release_z_ft,
        ], dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm > 0:
            return d / norm
        return np.array([0.0, 0.0, -1.0])

    @property
    def total_distance_ft(self) -> float:
        """Total distance from release to target."""
        d = np.array([
            self.target_x_ft - self.release_x_ft,
            self.target_y_ft - self.release_y_ft,
            self.target_z_ft - self.release_z_ft,
        ])
        return float(np.linalg.norm(d))

    @property
    def flight_time_sec(self) -> float:
        """Estimated total flight time."""
        if self.speed_ft_per_sec > 0:
            return self.total_distance_ft / self.speed_ft_per_sec
        return 0.0

    def position_at_z(self, z_ft: float) -> Tuple[float, float, float]:
        """
        Get the 3D position along the pitch line at a given Z depth.

        Linearly interpolates between release and target.
        """
        total_dz = self.target_z_ft - self.release_z_ft
        if abs(total_dz) < 1e-6:
            return (self.release_x_ft, self.release_y_ft, z_ft)

        t = (z_ft - self.release_z_ft) / total_dz
        t = max(0.0, min(1.0, t))

        x = self.release_x_ft + t * (self.target_x_ft - self.release_x_ft)
        y = self.release_y_ft + t * (self.target_y_ft - self.release_y_ft)

        return (x, y, z_ft)

    def position_at_time(self, t_sec: float) -> Tuple[float, float, float]:
        """
        Get 3D position at a given time after release.

        Includes simplified gravity model:
          y(t) = y0 + vy*t - 0.5*g*t^2
        """
        if self.speed_ft_per_sec <= 0:
            return (self.release_x_ft, self.release_y_ft, self.release_z_ft)

        direction = self.direction
        dist = self.speed_ft_per_sec * t_sec

        x = self.release_x_ft + direction[0] * dist
        z = self.release_z_ft + direction[2] * dist

        # Y with gravity (g = 32.2 ft/s^2)
        vy = direction[1] * self.speed_ft_per_sec
        y = self.release_y_ft + vy * t_sec - 0.5 * 32.2 * t_sec * t_sec

        return (x, y, z)


@dataclass
class CageGeometry:
    """
    Complete 3D model of a batting cage setup.

    Combines camera model, pitch trajectory, and cage dimensions
    to provide the full pixel <-> world conversion pipeline.
    """
    camera: CameraModel = field(default_factory=CameraModel)
    pitch: PitchTrajectory = field(default_factory=PitchTrajectory)

    # Cage dimensions (for visualization / bounds checking)
    cage_length_ft: float = 70.0   # along Z
    cage_width_ft: float = 14.0    # along X
    cage_height_ft: float = 12.0   # along Y

    # Calibration state
    _calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated and self.camera.is_calibrated

    def calibrate(self, machine_bbox_px: List[int],
                  machine_height_inches: float,
                  machine_distance_ft: float,
                  sign_speed_mph: Optional[float] = None,
                  camera_height_ft: float = 5.0,
                  camera_behind_plate_ft: float = 2.0,
                  release_height_ft: float = 3.5,
                  strike_zone_height_ft: float = 2.5):
        """
        Calibrate the full 3D model from known cage parameters.

        This is the main entry point for setting up the geometry.

        Parameters:
            machine_bbox_px: [x1, y1, x2, y2] of machine in frame
            machine_height_inches: physical height of the machine
            machine_distance_ft: distance from home plate to machine
            sign_speed_mph: posted speed on the cage sign
            camera_height_ft: estimated camera height above ground
            camera_behind_plate_ft: how far behind home plate
            release_height_ft: height of machine's ball release point
            strike_zone_height_ft: center of the strike zone
        """
        # Set up camera
        self.camera.camera_x_ft = 0.0
        self.camera.camera_y_ft = camera_height_ft
        self.camera.camera_z_ft = -camera_behind_plate_ft

        # Derive focal length from machine bbox
        self.camera.calibrate_from_known_object(
            machine_bbox_px, machine_height_inches, machine_distance_ft
        )

        # Derive camera aim offset: the machine center in the image
        # should project from the machine's known 3D position
        if self.camera.is_calibrated:
            mx1, my1, mx2, my2 = machine_bbox_px
            machine_center_u = (mx1 + mx2) / 2.0
            machine_center_v = (my1 + my2) / 2.0

            # Where SHOULD the machine center appear with no aim offset?
            projected = self.camera.world_to_pixel(
                0.0,  # centered laterally
                (release_height_ft + 0.0) / 2.0,  # mid-height of machine
                machine_distance_ft
            )
            if projected is not None:
                # The difference tells us the aim offset
                self.camera.aim_offset_x = machine_center_u - projected[0]
                self.camera.aim_offset_y = machine_center_v - projected[1]

        # Set up pitch trajectory
        self.pitch.release_x_ft = 0.0
        self.pitch.release_y_ft = release_height_ft
        self.pitch.release_z_ft = machine_distance_ft
        self.pitch.target_x_ft = 0.0
        self.pitch.target_y_ft = strike_zone_height_ft
        self.pitch.target_z_ft = 0.0

        if sign_speed_mph is not None and sign_speed_mph > 0:
            self.pitch.speed_ft_per_sec = sign_speed_mph * FEET_PER_MILE / SECONDS_PER_HOUR

        self._calibrated = True

    def pixel_to_depth_on_pitch_line(self, u: float, v: float) -> Optional[float]:
        """
        Estimate depth (Z) of a pixel point assuming it lies on the pitch line.

        Intersects the pixel ray with the 3D pitch line (closest point).
        Returns depth in feet from home plate, or None.
        """
        if not self.is_calibrated:
            return None

        # Get the camera ray for this pixel
        ray_origin, ray_dir = self.camera.pixel_to_ray(u, v)

        # Pitch line: from release point toward target
        line_origin = np.array([
            self.pitch.release_x_ft,
            self.pitch.release_y_ft,
            self.pitch.release_z_ft
        ])
        line_dir = self.pitch.direction

        # Find closest point between the two lines (ray and pitch line)
        # Using the standard closest-point-on-two-lines formula
        w0 = ray_origin - line_origin
        a = np.dot(ray_dir, ray_dir)     # always 1 for unit vector
        b = np.dot(ray_dir, line_dir)
        c = np.dot(line_dir, line_dir)   # always 1 for unit vector
        d = np.dot(ray_dir, w0)
        e = np.dot(line_dir, w0)

        denom = a * c - b * b
        if abs(denom) < 1e-10:
            return None  # Lines are parallel

        # t_pitch is the parameter along the pitch line
        t_pitch = (a * e - b * d) / denom

        # Get the 3D point on the pitch line
        point_on_pitch = line_origin + t_pitch * line_dir

        # Return the Z coordinate (depth from home plate)
        return float(point_on_pitch[2])

    def pixel_to_world_on_pitch_line(self, u: float, v: float
                                      ) -> Optional[Tuple[float, float, float]]:
        """
        Get the full 3D world position of a pixel point on the pitch line.

        Returns (x, y, z) in feet or None.
        """
        z = self.pixel_to_depth_on_pitch_line(u, v)
        if z is None:
            return None
        return self.pitch.position_at_z(z)

    def estimate_ball_depth_from_size(self, apparent_radius_px: float
                                       ) -> Optional[float]:
        """
        Estimate ball depth from its apparent size in pixels.

        Returns distance from camera in feet, or None.
        """
        if apparent_radius_px <= 0:
            return None
        apparent_diameter_px = apparent_radius_px * 2.0
        return self.camera.depth_from_apparent_size(
            apparent_diameter_px, BASEBALL_DIAMETER_INCHES
        )

    def compute_3d_speed(self, pos1_3d: Tuple[float, float, float],
                          pos2_3d: Tuple[float, float, float],
                          dt_seconds: float) -> Optional[float]:
        """
        Compute true 3D speed between two world positions.

        Returns speed in mph, or None if dt is zero.
        """
        if dt_seconds <= 0:
            return None

        dx = pos2_3d[0] - pos1_3d[0]
        dy = pos2_3d[1] - pos1_3d[1]
        dz = pos2_3d[2] - pos1_3d[2]
        dist_ft = np.sqrt(dx * dx + dy * dy + dz * dz)

        ft_per_sec = dist_ft / dt_seconds
        mph = ft_per_sec * SECONDS_PER_HOUR / FEET_PER_MILE
        return mph

    def compute_3d_velocity(self, pos1_3d: Tuple[float, float, float],
                             pos2_3d: Tuple[float, float, float],
                             dt_seconds: float
                             ) -> Optional[Tuple[float, float, float]]:
        """
        Compute 3D velocity vector between two world positions.

        Returns (vx, vy, vz) in ft/s, or None if dt is zero.
        """
        if dt_seconds <= 0:
            return None

        vx = (pos2_3d[0] - pos1_3d[0]) / dt_seconds
        vy = (pos2_3d[1] - pos1_3d[1]) / dt_seconds
        vz = (pos2_3d[2] - pos1_3d[2]) / dt_seconds

        return (vx, vy, vz)

    def pixel_speed_to_3d_speed_on_pitch_line(
            self, u1: float, v1: float, u2: float, v2: float,
            dt_seconds: float) -> Optional[float]:
        """
        Convert two pixel positions to true 3D speed assuming the ball
        is on the pitch line.

        This is the key function that solves the depth-dominant
        trajectory problem: even though the ball moves only 80px
        laterally in the image, we know it's traveling 60ft in depth.

        Returns speed in mph, or None.
        """
        pos1 = self.pixel_to_world_on_pitch_line(u1, v1)
        pos2 = self.pixel_to_world_on_pitch_line(u2, v2)

        if pos1 is None or pos2 is None:
            return None

        return self.compute_3d_speed(pos1, pos2, dt_seconds)

    def get_expected_pixel_velocity(self, z_ft: float, speed_mph: float
                                     ) -> Optional[Tuple[float, float]]:
        """
        Predict what pixel velocity a ball at depth Z moving at a given
        speed should produce. Useful for validating detections.

        Returns (du/dt, dv/dt) in pixels per second, or None.
        """
        if not self.is_calibrated or speed_mph <= 0:
            return None

        speed_ft_s = speed_mph * FEET_PER_MILE / SECONDS_PER_HOUR
        dt = 1.0 / 60.0  # simulate one frame at 60fps

        # Current position on pitch line
        pos1 = self.pitch.position_at_z(z_ft)
        # Position one time step later (ball moves toward plate = -Z)
        dz = self.pitch.direction[2] * speed_ft_s * dt
        pos2 = self.pitch.position_at_z(z_ft + dz)

        pix1 = self.camera.world_to_pixel(*pos1)
        pix2 = self.camera.world_to_pixel(*pos2)

        if pix1 is None or pix2 is None:
            return None

        du_per_sec = (pix2[0] - pix1[0]) / dt
        dv_per_sec = (pix2[1] - pix1[1]) / dt

        return (du_per_sec, dv_per_sec)

    def get_info_lines(self) -> List[str]:
        """Return geometry status lines for HUD display."""
        lines = []
        if self.is_calibrated:
            dist_tag = " +dist" if self.camera.has_distortion else ""
            lines.append(f"3D: focal={self.camera.focal_length_px:.0f}px{dist_tag}")
            lines.append(
                f"Cam: ({self.camera.camera_x_ft:.1f},"
                f"{self.camera.camera_y_ft:.1f},"
                f"{self.camera.camera_z_ft:.1f})ft"
            )
            if self.pitch.speed_ft_per_sec > 0:
                lines.append(
                    f"Pitch: {self.pitch.speed_ft_per_sec * SECONDS_PER_HOUR / FEET_PER_MILE:.0f}mph, "
                    f"{self.pitch.total_distance_ft:.0f}ft, "
                    f"{self.pitch.flight_time_sec:.2f}s"
                )
        else:
            lines.append("3D: NOT CALIBRATED")
        return lines

    def export_trajectory_3d(self, pixel_positions: List[Tuple[float, float]],
                              frame_indices: List[int], fps: float
                              ) -> List[dict]:
        """
        Convert a sequence of pixel positions to 3D trajectory data.

        Returns a list of dicts suitable for JSON export / 3D visualization:
        [
            {
                "frame": 97,
                "time_s": 1.618,
                "pixel": {"u": 988, "v": 450},
                "world_ft": {"x": 0.1, "y": 3.4, "z": 58.2},
                "depth_ft": 58.2,
                "speed_mph": 47.2,
                "velocity_ft_s": {"vx": 0.1, "vy": -1.2, "vz": -69.3}
            },
            ...
        ]
        """
        trajectory = []
        prev_pos_3d = None
        prev_time = None

        for i, (pos_px, frame_idx) in enumerate(zip(pixel_positions, frame_indices)):
            u, v = pos_px
            time_s = frame_idx / fps

            entry = {
                "frame": frame_idx,
                "time_s": round(time_s, 4),
                "pixel": {"u": round(u, 1), "v": round(v, 1)},
                "world_ft": None,
                "depth_ft": None,
                "speed_mph": None,
                "velocity_ft_s": None,
            }

            pos_3d = self.pixel_to_world_on_pitch_line(u, v)
            if pos_3d is not None:
                entry["world_ft"] = {
                    "x": round(pos_3d[0], 3),
                    "y": round(pos_3d[1], 3),
                    "z": round(pos_3d[2], 3),
                }
                entry["depth_ft"] = round(pos_3d[2], 2)

                if prev_pos_3d is not None and prev_time is not None:
                    dt = time_s - prev_time
                    if dt > 0:
                        speed = self.compute_3d_speed(prev_pos_3d, pos_3d, dt)
                        vel = self.compute_3d_velocity(prev_pos_3d, pos_3d, dt)
                        if speed is not None:
                            entry["speed_mph"] = round(speed, 1)
                        if vel is not None:
                            entry["velocity_ft_s"] = {
                                "vx": round(vel[0], 2),
                                "vy": round(vel[1], 2),
                                "vz": round(vel[2], 2),
                            }

                prev_pos_3d = pos_3d
                prev_time = time_s

            trajectory.append(entry)

        return trajectory
