from terranav.world.generator import generate_demo_terrain

from terranav.visualization.terrain_viewer import show_terrain


def main():

    terrain = generate_demo_terrain(
        width_m=20.0,
        height_m=20.0,
        resolution_m=0.25,
        seed=42,
    )

    print("\nTerraNav Terrain Simulator")
    print("--------------------------")

    print(f"World: {terrain.width_m} x {terrain.height_m} metres")

    print(f"Grid: {terrain.nx} x {terrain.ny}")

    print(f"Resolution: {terrain.resolution_m} metres")

    print(
        f"Elevation range: "
        f"{terrain.elevation_m.min():.3f} to "
        f"{terrain.elevation_m.max():.3f} metres"
    )

    print(
        f"Mean friction: {terrain.friction.mean():.3f}"
    )

    print(
        f"Mean roughness: {terrain.roughness.mean():.3f}"
    )

    print(
        f"Obstacle coverage: "
        f"{terrain.obstacles.mean() * 100:.2f}%"
    )

    print("\nGenerating visualisation...")

    show_terrain(terrain)


if __name__ == "__main__":
    main()