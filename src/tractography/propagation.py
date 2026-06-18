#src/tractography/propagation.py
from typing import Optional, Tuple
import numpy as np
from src.tractography.interpolation import interpolate_pdd

def propagate_streamline(
    seed: np.ndarray,
    pdd_volume: np.ndarray,
    mask: np.ndarray,
    fa_volume: np.ndarray,
    step_size: float = 0.5,
    max_steps: int = 1000,
    angle_threshold: float = 45.0,
    fa_threshold: float = 0.15,
    direction: int = 1,
) -> np.ndarray:
    """Propaga una línea de corriente en una dirección usando integración de Euler.

    Corrige el signo antipodal del PDD con respecto al paso anterior y evalúa
    criterios de parada por límites anatómicos, curvatura y umbral de FA.

    Parameters:
        seed  (np.ndarray):      Coordenada inicial [x, y, z] de forma (3,).
        pdd_volume  (np.ndarray):Volumen PDD de forma (X, Y, Z, 3).
        mask  (np.ndarray):      Máscara binaria del cerebro de forma (X, Y, Z).
        fa_volume  (np.ndarray): Mapa de Anisotropía Fraccional de forma (X, Y, Z).
        step_size  (float):      Tamaño del paso en el espacio continuo de vóxeles.
        max_steps  (int):        Número máximo de pasos permitidos en esta dirección.
        angle_threshold  (float):Umbral máximo de cambio angular permitido en grados.
        fa_threshold  (float):   Umbral mínimo de FA para detener el seguimiento.
        direction  (int):        Sentido de propagación: 1 para forward, -1 para backward.

    Returns:
        np.ndarray
            Arreglo de forma (N_steps, 3) con los puntos continuos de la línea.
    """
    angle_thresh_rad = np.deg2rad(angle_threshold)

    streamline = [seed.copy().astype(np.float32)]
    current_pos = seed.copy().astype(np.float32)
    prev_dir = None

    for _ in range(max_steps):
        # Interpolar dirección en la posición continua actual
        current_dir = interpolate_pdd(current_pos, pdd_volume, mask)
        if current_dir is None:
            break

        # Verificar criterio de parada por FA (interpolación nearest neighbor)
        x, y, z = np.round(current_pos).astype(np.int32)
        if (
            x < 0
            or x >= fa_volume.shape[0]
            or y < 0
            or y >= fa_volume.shape[1]
            or z < 0
            or z >= fa_volume.shape[2]
        ):
            break

        if fa_volume[x, y, z] < fa_threshold:
            break

        # Mitigar el signo antipodal alineando current_dir con prev_dir
        if prev_dir is not None:
            dot_product = np.dot(current_dir, prev_dir)
            current_dir *= np.sign(dot_product) or 1.0

            # Verificar cambio de ángulo real tras la alineación de sentido
            cos_angle = np.clip(np.abs(dot_product), 0.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle > angle_thresh_rad:
                break

        # Avanzar posición mediante integración de Euler
        next_pos = current_pos + direction * step_size * current_dir

        streamline.append(next_pos.astype(np.float32))
        prev_dir = current_dir
        current_pos = next_pos

    return np.array(streamline, dtype=np.float32)


def track_full_streamline(
    seed: np.ndarray,
    pdd_volume: np.ndarray,
    mask: np.ndarray,
    fa_volume: np.ndarray,
    step_size: float = 0.5,
    max_steps: int = 1000,
    angle_threshold: float = 45.0,
    fa_threshold: float = 0.15,
) -> np.ndarray:
    """Genera una línea de corriente completa uniendo los segmentos forward y backward.

    Parameters:
    seed : np.ndarray
        Punto de inicio en el espacio tridimensional.

    Returns:
    np.ndarray
        Línea de corriente unificada contigua de forma (N_total, 3).
    """
    # Configuración compartida de parámetros de siembra
    params = {
        'seed': seed,
        'pdd_volume': pdd_volume,
        'mask': mask,
        'fa_volume': fa_volume,
        'step_size': step_size,
        'max_steps': max_steps,
        'angle_threshold': angle_threshold,
        'fa_threshold': fa_threshold,
    }

    forward = propagate_streamline(direction=1, **params)
    backward = propagate_streamline(direction=-1, **params)

    # Unir segmentos: invertir el bloque backward y excluir su última posición
    # (que es la semilla duplicada) antes de concatenar con forward
    if len(backward) > 1:
        full_streamline = np.vstack((backward[::-1][:-1], forward))
    else:
        full_streamline = forward

    return full_streamline