# src/tractography/tracking.py
from pathlib import Path
from typing import List, Tuple
import nibabel as nib
import numpy as np
from tqdm import tqdm

from src.tractography.propagation import track_full_streamline


def track_multiple_seeds(
    seeds: np.ndarray,
    pdd_volume: np.ndarray,
    mask: np.ndarray,
    fa_volume: np.ndarray,
    step_size: float = 0.5,
    max_steps: int = 1000,
    angle_threshold: float = 45.0,
    fa_threshold: float = 0.15,
    min_length: float = 10.0,  # longitud mínima en mm (aprox. voxels)
    verbose: bool = True,
) -> List[np.ndarray]:
    """Realiza tractografía determinística para múltiples semillas.

    Returns:
    List[np.ndarray]
        Lista de streamlines (cada una es un array (N, 3))
    """
    streamlines = []

    iterator = tqdm(seeds, desc='Tracking streamlines') if verbose else seeds

    for seed in iterator:
        streamline = track_full_streamline(
            seed=seed,
            pdd_volume=pdd_volume,
            mask=mask,
            fa_volume=fa_volume,
            step_size=step_size,
            max_steps=max_steps,
            angle_threshold=angle_threshold,
            fa_threshold=fa_threshold,
        )

        # Filtrar streamlines muy cortas (ruido)
        length = np.sum(np.linalg.norm(np.diff(streamline, axis=0), axis=1))
        if length >= min_length:
            streamlines.append(streamline)

    print(
        f'\nCompletado. Streamlines válidas: {len(streamlines)} / {len(seeds)}'
    )
    return streamlines

def load_tractography_data(
    pdd_path: str, fa_path: str, mask_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga los volúmenes necesarios para tracking"""
    pdd_nii = nib.load(pdd_path)
    fa_nii = nib.load(fa_path)
    mask_nii = nib.load(mask_path)

    pdd = pdd_nii.get_fdata(dtype=np.float32)
    fa = fa_nii.get_fdata(dtype=np.float32)
    mask = mask_nii.get_fdata(dtype=np.float32)

    return pdd, fa, mask