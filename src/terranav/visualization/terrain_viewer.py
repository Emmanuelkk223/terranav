
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from terranav.world.terrain import TerrainMap


MATERIAL_NAMES = [
    "Soil",
    "Grass",
    "Mud",
    "Gravel",
    "Rock",
]


MATERIAL_COLORS = [
    "#A67C52",
    "#72A66B",
    "#654321",
    "#B9B9B9",
    "#777777",
]


def _format_axis(ax, terrain):

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")

    ax.set_xlim(0, terrain.width_m)
    ax.set_ylim(0, terrain.height_m)

    ax.set_aspect("equal")


def show_terrain(terrain: TerrainMap):

    extent = (
        0,
        terrain.width_m,
        0,
        terrain.height_m,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        constrained_layout=True,
    )

    fig.suptitle(
        "TerraNav — Synthetic Terrain Environment",
        fontsize=16,
        fontweight="bold",
    )

    # --------------------------------------------------
    # 1. ELEVATION MAP
    # --------------------------------------------------

    ax = axes[0, 0]

    image = ax.imshow(
        terrain.elevation_m,
        origin="lower",
        extent=extent,
        cmap="terrain",
        interpolation="nearest",
    )

    ax.contour(
        terrain.x_centers,
        terrain.y_centers,
        terrain.elevation_m,
        levels=12,
        colors="white",
        linewidths=0.6,
        alpha=0.6,
    )

    ax.set_title("Elevation Map")

    fig.colorbar(
        image,
        ax=ax,
        label="Elevation (m)",
    )

    _format_axis(ax, terrain)

    # --------------------------------------------------
    # 2. FRICTION MAP
    # --------------------------------------------------

    ax = axes[0, 1]

    image = ax.imshow(
        terrain.friction,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_title("Friction Map")

    fig.colorbar(
        image,
        ax=ax,
        label="Friction coefficient",
    )

    _format_axis(ax, terrain)

    # --------------------------------------------------
    # 3. ROUGHNESS MAP
    # --------------------------------------------------

    ax = axes[0, 2]

    image = ax.imshow(
        terrain.roughness,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_title("Roughness Map")

    fig.colorbar(
        image,
        ax=ax,
        label="Normalised roughness",
    )

    _format_axis(ax, terrain)

    # --------------------------------------------------
    # 4. OBSTACLE MAP
    # --------------------------------------------------

    ax = axes[1, 0]

    image = ax.imshow(
        terrain.obstacles.astype(float),
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_title("Obstacle Map")

    fig.colorbar(
        image,
        ax=ax,
        label="Obstacle occupancy",
        ticks=[0, 1],
    )

    _format_axis(ax, terrain)

    # --------------------------------------------------
    # 5. MATERIAL MAP
    # --------------------------------------------------

    ax = axes[1, 1]

    material_cmap = ListedColormap(
        MATERIAL_COLORS
    )

    image = ax.imshow(
        terrain.material,
        origin="lower",
        extent=extent,
        cmap=material_cmap,
        vmin=-0.5,
        vmax=4.5,
        interpolation="nearest",
    )

    ax.set_title("Terrain Material Map")

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[0, 1, 2, 3, 4],
    )

    colorbar.ax.set_yticklabels(
        MATERIAL_NAMES
    )

    _format_axis(ax, terrain)

    # --------------------------------------------------
    # 6. TERRAIN STATISTICS
    # --------------------------------------------------

    ax = axes[1, 2]

    obstacle_percentage = (
        terrain.obstacles.mean() * 100
    )

    statistics = (
        f"GRID INFORMATION\n\n"
        f"Width: {terrain.width_m:.2f} m\n"
        f"Height: {terrain.height_m:.2f} m\n"
        f"Resolution: {terrain.resolution_m:.2f} m\n"
        f"Grid dimensions: {terrain.nx} x {terrain.ny}\n"
        f"Total cells: {terrain.nx * terrain.ny}\n\n"
        f"TERRAIN STATISTICS\n\n"
        f"Elevation min: {terrain.elevation_m.min():.3f} m\n"
        f"Elevation max: {terrain.elevation_m.max():.3f} m\n"
        f"Mean friction: {terrain.friction.mean():.3f}\n"
        f"Mean roughness: {terrain.roughness.mean():.3f}\n"
        f"Obstacle coverage: {obstacle_percentage:.2f}%"
    )

    ax.text(
        0.05,
        0.95,
        statistics,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        family="monospace",
    )

    ax.set_title("World Statistics")
    ax.axis("off")

    plt.show()