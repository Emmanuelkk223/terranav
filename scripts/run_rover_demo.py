import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from terranav.world.generator import generate_demo_terrain

from terranav.robot.kinematics import (
    RobotGeometry,
    Pose2D,
    DifferentialDrive,
)


def robot_vertices(pose, length, width):
    """
    Calculate the robot's oriented polygon vertices
    in world coordinates.

    The front of the robot points along its local +X axis.
    """

    local = np.array([
        [length / 2, 0.0],
        [length / 2 - 0.15, width / 2],
        [-length / 2, width / 2],
        [-length / 2, -width / 2],
        [length / 2 - 0.15, -width / 2],
    ])

    c = np.cos(pose.yaw)
    s = np.sin(pose.yaw)

    rotation = np.array([
        [c, -s],
        [s,  c],
    ])

    world = local @ rotation.T

    world[:, 0] += pose.x
    world[:, 1] += pose.y

    return world


def command_at(t):
    """
    Open-loop demonstration commands.

    Returns:
        linear velocity (m/s),
        angular velocity (rad/s)
    """

    if t < 4.0:
        return 0.40, 0.0

    elif t < 7.0:
        return 0.25, 0.55

    elif t < 11.0:
        return 0.40, 0.0

    elif t < 14.0:
        return 0.25, -0.55

    else:
        return 0.40, 0.0


def simulate():

    dt = 0.05
    duration = 18.0

    geometry = RobotGeometry()
    rover = DifferentialDrive(geometry)

    pose = Pose2D(
        x=2.0,
        y=2.0,
        yaw=0.0,
    )

    poses = [pose]

    timestamps = np.arange(
        0.0,
        duration + dt,
        dt,
    )

    for t in timestamps[:-1]:

        linear_velocity, angular_velocity = command_at(t)

        pose = rover.step(
            pose,
            linear_velocity,
            angular_velocity,
            dt,
        )

        poses.append(pose)

    return timestamps, poses, geometry


def main():

    terrain = generate_demo_terrain(seed=42)

    timestamps, poses, geometry = simulate()

    x_history = [p.x for p in poses]
    y_history = [p.y for p in poses]

    fig, ax = plt.subplots(figsize=(10, 9))

    extent = (
        0,
        terrain.width_m,
        0,
        terrain.height_m,
    )

    ax.imshow(
        terrain.elevation_m,
        origin="lower",
        extent=extent,
        cmap="terrain",
    )

    ax.contour(
        terrain.x_centers,
        terrain.y_centers,
        terrain.elevation_m,
        levels=12,
        colors="white",
        alpha=0.4,
        linewidths=0.5,
    )

    obstacle_image = np.ma.masked_where(
        ~terrain.obstacles,
        terrain.obstacles.astype(float),
    )

    ax.imshow(
        obstacle_image,
        origin="lower",
        extent=extent,
        cmap="Reds",
        alpha=0.9,
        interpolation="nearest",
    )

    # Rover trajectory.
    trajectory, = ax.plot(
        [],
        [],
        color="cyan",
        linewidth=2,
        label="Rover trajectory",
    )

    # Robot body.
    initial_vertices = robot_vertices(
        poses[0],
        geometry.body_length_m,
        geometry.body_width_m,
    )

    robot_patch = Polygon(
        initial_vertices,
        closed=True,
        facecolor="deepskyblue",
        edgecolor="black",
        linewidth=2,
        zorder=5,
    )

    ax.add_patch(robot_patch)

    ax.scatter(
        [poses[0].x],
        [poses[0].y],
        color="lime",
        s=65,
        label="Starting position",
        zorder=6,
    )

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        color="white",
        fontsize=11,
        bbox=dict(facecolor="black", alpha=0.6),
    )

    ax.set_xlim(0, terrain.width_m)
    ax.set_ylim(0, terrain.height_m)

    ax.set_aspect("equal")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")

    ax.set_title(
        "TerraNav — Differential Drive Rover"
    )

    ax.legend(loc="lower right")

    def update(frame):

        pose = poses[frame]

        trajectory.set_data(
            x_history[:frame + 1],
            y_history[:frame + 1],
        )

        robot_patch.set_xy(
            robot_vertices(
                pose,
                geometry.body_length_m,
                geometry.body_width_m,
            )
        )

        time_text.set_text(
            f"Time: {timestamps[frame]:.2f} s\n"
            f"Position: ({pose.x:.2f}, {pose.y:.2f}) m\n"
            f"Heading: {np.degrees(pose.yaw):.1f}°"
        )

        return trajectory, robot_patch, time_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(poses),
        interval=50,
        blit=False,
        repeat=False,
    )

    # Retain reference to avoid garbage collection.
    _ = animation

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()