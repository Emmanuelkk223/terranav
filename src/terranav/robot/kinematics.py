from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RobotGeometry:
    wheel_radius_m: float = 0.10
    track_width_m: float = 0.42

    body_length_m: float = 0.55
    body_width_m: float = 0.42

    max_wheel_speed_rad_s: float = 12.0

    def __post_init__(self):
        values = (
            self.wheel_radius_m,
            self.track_width_m,
            self.body_length_m,
            self.body_width_m,
            self.max_wheel_speed_rad_s,
        )

        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("All geometry parameters must be positive and finite.")


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


class DifferentialDrive:

    def __init__(self, geometry: RobotGeometry):
        self.geometry = geometry

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalise angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def wheel_to_body(
        self,
        left_speed: float,
        right_speed: float,
    ) -> tuple[float, float]:
        """
        Convert wheel angular velocities into
        linear and angular body velocities.
        """

        r = self.geometry.wheel_radius_m
        b = self.geometry.track_width_m

        if not all(math.isfinite(v) for v in (left_speed, right_speed)):
            raise ValueError("Wheel speeds must be finite.")

        linear_velocity = (
            r * (left_speed + right_speed) / 2.0
        )

        angular_velocity = (
            r * (right_speed - left_speed) / b
        )

        return linear_velocity, angular_velocity

    def body_to_wheels(
        self,
        linear_velocity: float,
        angular_velocity: float,
    ) -> tuple[float, float]:
        """
        Convert desired body velocities into wheel speeds.

        Wheel speed limits are enforced while preserving
        the requested turning curvature.
        """

        if not all(math.isfinite(v) for v in (
            linear_velocity, angular_velocity
        )):
            raise ValueError("Body velocities must be finite.")

        r = self.geometry.wheel_radius_m
        b = self.geometry.track_width_m

        left_speed = (
            linear_velocity - angular_velocity * b / 2
        ) / r

        right_speed = (
            linear_velocity + angular_velocity * b / 2
        ) / r

        max_speed = self.geometry.max_wheel_speed_rad_s

        peak_speed = max(abs(left_speed), abs(right_speed))

        if peak_speed > max_speed:
            scale = max_speed / peak_speed
            left_speed *= scale
            right_speed *= scale

        return left_speed, right_speed

    def integrate(
        self,
        pose: Pose2D,
        left_speed: float,
        right_speed: float,
        dt: float,
    ) -> Pose2D:
        """
        Integrate rover movement over one time step.

        Assumes constant wheel velocities during dt
        and initially ignores wheel slip.
        """

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Time step must be positive and finite.")

        v, omega = self.wheel_to_body(
            left_speed,
            right_speed,
        )

        theta = pose.yaw

        if abs(omega) < 1e-9:

            # Straight-line motion.
            new_x = pose.x + v * math.cos(theta) * dt
            new_y = pose.y + v * math.sin(theta) * dt

            new_yaw = theta

        else:

            # Exact circular-arc integration.
            new_yaw = theta + omega * dt

            new_x = pose.x + (
                v / omega
            ) * (
                math.sin(new_yaw) - math.sin(theta)
            )

            new_y = pose.y - (
                v / omega
            ) * (
                math.cos(new_yaw) - math.cos(theta)
            )

        return Pose2D(
            x=new_x,
            y=new_y,
            yaw=self.normalize_angle(new_yaw),
        )

    def step(
        self,
        pose: Pose2D,
        linear_velocity: float,
        angular_velocity: float,
        dt: float,
    ) -> Pose2D:
        """Convenience method for velocity-based control."""

        left, right = self.body_to_wheels(
            linear_velocity,
            angular_velocity,
        )

        return self.integrate(pose, left, right, dt)