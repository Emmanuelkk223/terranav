
import numpy as np

from terranav.world.terrain import TerrainMap


# Material identifiers
SOIL = 0
GRASS = 1
MUD = 2
GRAVEL = 3
ROCK = 4


def gaussian(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    amplitude: float,
) -> np.ndarray:
    """
    Generate a Gaussian-shaped elevation feature.

    Can represent hills or depressions depending on amplitude.
    """

    exponent = (
        ((x - cx) ** 2) / (2 * sx**2)
        +
        ((y - cy) ** 2) / (2 * sy**2)
    )

    return amplitude * np.exp(-exponent)


def ellipse_mask(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle: float = 0.0,
) -> np.ndarray:
    """
    Generate an elliptical region.

    The angle is measured in radians.
    """

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    dx = x - cx
    dy = y - cy

    rotated_x = cos_a * dx + sin_a * dy
    rotated_y = -sin_a * dx + cos_a * dy

    return (
        (rotated_x / rx) ** 2
        +
        (rotated_y / ry) ** 2
    ) <= 1.0


def generate_demo_terrain(
    width_m: float = 20.0,
    height_m: float = 20.0,
    resolution_m: float = 0.25,
    seed: int = 42,
) -> TerrainMap:
    """
    Generate a reproducible synthetic off-road environment.

    The generated environment includes:
        - Hills and depressions
        - Different terrain materials
        - Variable friction
        - Variable roughness
        - Physical obstacles
    """

    if (
        width_m <= 0
        or height_m <= 0
        or resolution_m <= 0
    ):
        raise ValueError("Terrain dimensions must be positive.")

    if not np.isclose(width_m / resolution_m,
                      round(width_m / resolution_m)):
        raise ValueError("Invalid width/resolution combination.")

    if not np.isclose(height_m / resolution_m,
                      round(height_m / resolution_m)):
        raise ValueError("Invalid height/resolution combination.")

    rng = np.random.default_rng(seed)

    nx = int(round(width_m / resolution_m))
    ny = int(round(height_m / resolution_m))

    x = (np.arange(nx) + 0.5) * resolution_m
    y = (np.arange(ny) + 0.5) * resolution_m

    # Creates two coordinate matrices of shape (ny, nx).
    X, Y = np.meshgrid(x, y)

    shape = (ny, nx)

    # --------------------------------------------------
    # 1. INITIALISE MATERIAL PROPERTIES
    # --------------------------------------------------

    material = np.full(shape, SOIL, dtype=np.uint8)

    friction = np.full(shape, 0.62, dtype=np.float32)

    roughness = np.full(shape, 0.12, dtype=np.float32)

    # --------------------------------------------------
    # 2. DEFINE TERRAIN REGIONS
    # --------------------------------------------------

    grass_region = ellipse_mask(
        X, Y,
        cx=4.0,
        cy=4.5,
        rx=3.0,
        ry=2.0,
        angle=0.3,
    )

    material[grass_region] = GRASS
    friction[grass_region] = 0.68
    roughness[grass_region] = 0.18

    mud_region = ellipse_mask(
        X, Y,
        cx=13.0,
        cy=13.5,
        rx=3.2,
        ry=2.2,
        angle=-0.4,
    )

    material[mud_region] = MUD
    friction[mud_region] = 0.24
    roughness[mud_region] = 0.48

    gravel_region = ellipse_mask(
        X, Y,
        cx=15.0,
        cy=5.0,
        rx=2.3,
        ry=1.7,
        angle=0.5,
    )

    material[gravel_region] = GRAVEL
    friction[gravel_region] = 0.48
    roughness[gravel_region] = 0.55

    rock_region = ellipse_mask(
        X, Y,
        cx=7.5,
        cy=12.0,
        rx=1.8,
        ry=1.2,
        angle=0.2,
    )

    material[rock_region] = ROCK
    friction[rock_region] = 0.42
    roughness[rock_region] = 0.78

    # --------------------------------------------------
    # 3. GENERATE ELEVATION
    # --------------------------------------------------

    # A gentle overall slope.
    elevation = (
        0.015 * X
        +
        0.010 * Y
    )

    # Main hill.
    elevation += gaussian(
        X, Y,
        cx=6.0,
        cy=14.0,
        sx=2.0,
        sy=2.5,
        amplitude=0.85,
    )

    # Elongated secondary hill.
    elevation += gaussian(
        X, Y,
        cx=15.0,
        cy=6.0,
        sx=1.8,
        sy=3.0,
        amplitude=0.45,
    )

    # A shallow depression.
    elevation += gaussian(
        X, Y,
        cx=11.0,
        cy=10.5,
        sx=2.5,
        sy=1.8,
        amplitude=-0.25,
    )

    # --------------------------------------------------
    # 4. INTRODUCE SURFACE IRREGULARITY
    # --------------------------------------------------

    phase = rng.uniform(0.0, 2.0 * np.pi)

    ripple = (
        0.003 + 0.025 * roughness
    ) * (
        np.sin(2.8 * X + phase)
        *
        np.cos(3.2 * Y - phase / 2)
    )

    random_noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=shape,
    )

    micro_noise = random_noise * (
        0.001 + 0.006 * roughness
    )

    elevation += ripple + micro_noise

    # --------------------------------------------------
    # 5. GENERATE PHYSICAL OBSTACLES
    # --------------------------------------------------

    obstacles = np.zeros(shape, dtype=bool)

    rock_obstacles = [
        (5.5, 5.7, 0.35),
        (8.1, 7.8, 0.50),
        (10.0, 16.5, 0.45),
        (17.0, 13.5, 0.60),
        (15.8, 9.4, 0.35),
    ]

    for cx, cy, radius in rock_obstacles:

        obstacle_mask = (
            (X - cx) ** 2
            +
            (Y - cy) ** 2
        ) <= radius**2

        obstacles |= obstacle_mask

    # Keep the synthetic parameter ranges bounded.
    friction = np.clip(friction, 0.15, 0.90)
    roughness = np.clip(roughness, 0.0, 1.0)

    # --------------------------------------------------
    # 6. RETURN TERRAIN MAP
    # --------------------------------------------------

    return TerrainMap(
        width_m=width_m,
        height_m=height_m,
        resolution_m=resolution_m,
        elevation_m=elevation.astype(np.float32),
        friction=friction,
        roughness=roughness,
        obstacles=obstacles,
        material=material,
    )
