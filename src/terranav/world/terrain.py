
from dataclasses import dataclass

import numpy as np


@dataclass
class TerrainMap:
    """
    Stores the ground-truth properties of a simulated terrain.

    Each terrain property is represented as a 2D NumPy array.

    Array indexing convention:
        terrain[y_index, x_index]
    """

    width_m: float
    height_m: float
    resolution_m: float

    elevation_m: np.ndarray
    friction: np.ndarray
    roughness: np.ndarray
    obstacles: np.ndarray
    material: np.ndarray

    def __post_init__(self):

        if self.resolution_m <= 0:
            raise ValueError("Resolution must be positive.")

        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("World dimensions must be positive.")

        nx_float = self.width_m / self.resolution_m
        ny_float = self.height_m / self.resolution_m

        if not np.isclose(nx_float, round(nx_float)):
            raise ValueError("Width must be divisible by resolution.")

        if not np.isclose(ny_float, round(ny_float)):
            raise ValueError("Height must be divisible by resolution.")

        expected_shape = (
            int(round(ny_float)),
            int(round(nx_float))
        )

        arrays = {
            "elevation_m": self.elevation_m,
            "friction": self.friction,
            "roughness": self.roughness,
            "obstacles": self.obstacles,
            "material": self.material,
        }

        for name, array in arrays.items():
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {array.shape}; "
                    f"expected {expected_shape}."
                )

        if not np.isfinite(self.elevation_m).all():
            raise ValueError("Elevation contains invalid values.")

        if np.any(self.friction < 0):
            raise ValueError("Friction cannot be negative.")

        if np.any(self.roughness < 0):
            raise ValueError("Roughness cannot be negative.")

    @property
    def nx(self) -> int:
        return self.elevation_m.shape[1]

    @property
    def ny(self) -> int:
        return self.elevation_m.shape[0]

    @property
    def x_centers(self) -> np.ndarray:
        return (
            np.arange(self.nx) + 0.5
        ) * self.resolution_m

    @property
    def y_centers(self) -> np.ndarray:
        return (
            np.arange(self.ny) + 0.5
        ) * self.resolution_m
