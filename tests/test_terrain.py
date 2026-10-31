import numpy as np

from terranav.world.generator import generate_demo_terrain


def test_grid_dimensions():

    terrain = generate_demo_terrain()

    assert terrain.nx == 80
    assert terrain.ny == 80

    assert terrain.elevation_m.shape == (80, 80)


def test_friction_values_are_valid():

    terrain = generate_demo_terrain()

    assert np.all(terrain.friction >= 0)
    assert np.all(terrain.friction <= 1)


def test_roughness_values_are_valid():

    terrain = generate_demo_terrain()

    assert np.all(terrain.roughness >= 0)
    assert np.all(terrain.roughness <= 1)


def test_obstacles_exist():

    terrain = generate_demo_terrain()

    assert terrain.obstacles.any()


def test_terrain_is_reproducible():

    terrain_a = generate_demo_terrain(seed=42)
    terrain_b = generate_demo_terrain(seed=42)

    np.testing.assert_array_equal(
        terrain_a.elevation_m,
        terrain_b.elevation_m,
    )

    np.testing.assert_array_equal(
        terrain_a.friction,
        terrain_b.friction,
    )