import math

import pytest

from terranav.robot.kinematics import (
    RobotGeometry,
    Pose2D,
    DifferentialDrive,
)


@pytest.fixture
def robot():
    geometry = RobotGeometry()
    return DifferentialDrive(geometry)


def test_straight_line_motion(robot):

    pose = Pose2D(0.0, 0.0, 0.0)

    result = robot.integrate(
        pose,
        left_speed=5.0,
        right_speed=5.0,
        dt=1.0,
    )

    assert result.x == pytest.approx(0.5)
    assert result.y == pytest.approx(0.0)
    assert result.yaw == pytest.approx(0.0)


def test_pure_rotation(robot):

    pose = Pose2D(0.0, 0.0, 0.0)

    result = robot.integrate(
        pose,
        left_speed=-5.0,
        right_speed=5.0,
        dt=1.0,
    )

    expected_yaw = (
        0.1 * 10.0 / 0.42
    )

    assert result.x == pytest.approx(0.0)
    assert result.y == pytest.approx(0.0)
    assert result.yaw == pytest.approx(
        math.atan2(
            math.sin(expected_yaw),
            math.cos(expected_yaw),
        )
    )


def test_body_to_wheel_conversion(robot):

    left, right = robot.body_to_wheels(
        linear_velocity=0.5,
        angular_velocity=0.0,
    )

    assert left == pytest.approx(5.0)
    assert right == pytest.approx(5.0)


def test_wheel_speed_saturation(robot):

    left, right = robot.body_to_wheels(
        linear_velocity=10.0,
        angular_velocity=0.0,
    )

    assert max(abs(left), abs(right)) <= 12.0


def test_invalid_time_step(robot):

    pose = Pose2D(0.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        robot.integrate(
            pose,
            left_speed=5.0,
            right_speed=5.0,
            dt=0.0,
        )