# src/tractography/seeds.py
from pathlib import Path
from typing import Optional
import numpy as np


def generate_seeds(
    mask: np.ndarray,
    fa_volume: np.ndarray,
    fa_thresh: float = 0.2,
    method: str = 'grid',
    density: int = 2,
    n_seeds_random: int = 10000,
    random_seed: int = 42,
) -> np.ndarray:
    """Genera coordenadas continuas 3D para iniciar la tractografía.

    Combina la máscara del cerebro y un umbral de Anisotropía Fraccional (FA).

    Parameters:
    mask : np.ndarray
        Máscara binaria del cerebro de forma (X, Y, Z).
    fa_volume : np.ndarray
        Mapa de Anisotropía Fraccional de forma (X, Y, Z).
    fa_thresh : float
        Umbral mínimo de FA para considerar materia blanca.
    method : str
        'grid' para sub-rejilla fija por vóxel o 'random' para muestreo
        estocástico.
    density : int
        Si el método es 'grid', define el número de semillas por eje dentro del
        vóxel (ej. 2 significa 2x2x2 = 8 semillas por vóxel).
    n_seeds_random : int
        Si el método es 'random', número total de semillas a generar.
    random_seed : int
        Semilla del generador aleatorio para reproducibilidad.

    Returns:
    np.ndarray
        Arreglo de coordenadas continuas de forma (N, 3) con tipo np.float32.
    """
    valid_voxels_mask = (mask > 0) & (fa_volume >= fa_thresh)
    x_indices, y_indices, z_indices = np.where(valid_voxels_mask)
    n_valid_voxels = len(x_indices)

    if n_valid_voxels == 0:
        raise ValueError(
            f'No se encontraron vóxeles que cumplan con FA >= {fa_thresh} dentro de la máscara.'
        )

    if method == 'random':
        rng = np.random.default_rng(random_seed)
        chosen_indices = rng.choice(
            n_valid_voxels, size=n_seeds_random, replace=True
        )

        x_base = x_indices[chosen_indices].astype(np.float32)
        y_base = y_indices[chosen_indices].astype(np.float32)
        z_base = z_indices[chosen_indices].astype(np.float32)

        x_seeds = x_base + rng.random(n_seeds_random, dtype=np.float32)
        y_seeds = y_base + rng.random(n_seeds_random, dtype=np.float32)
        z_seeds = z_base + rng.random(n_seeds_random, dtype=np.float32)

        return np.stack([x_seeds, y_seeds, z_seeds], axis=1)

    if method == 'grid':
        # Dividir el espacio [0, 1] del vóxel de manera perfectamente simétrica en float32
        offsets = np.linspace(
            1.0 / (2 * density),
            1.0 - 1.0 / (2 * density),
            density,
            dtype=np.float32,
        )

        gx, gy, gz = np.meshgrid(offsets, offsets, offsets, indexing='ij')
        grid_offsets = np.stack(
            [gx.ravel(), gy.ravel(), gz.ravel()], axis=1
        )  # (density^3, 3)

        d_cube = grid_offsets.shape[0]

        x_base = np.repeat(x_indices, d_cube).astype(np.float32)
        y_base = np.repeat(y_indices, d_cube).astype(np.float32)
        z_base = np.repeat(z_indices, d_cube).astype(np.float32)
        bases = np.stack([x_base, y_base, z_base], axis=1)

        all_offsets = np.tile(grid_offsets, (n_valid_voxels, 1))

        return bases + all_offsets

    raise ValueError(f"Método de siembra '{method}' no reconocido.")