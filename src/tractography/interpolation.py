#src/tractography/interpolation.py
from typing import Optional
import numpy as np


def interpolate_pdd(
    pos: np.ndarray, pdd_volume: np.ndarray, mask: np.ndarray
) -> Optional[np.ndarray]:
    """Interpola de manera trilineal el vector de dirección principal (PDD)

    en una posición continua 3D dentro del espacio de vóxeles.

    Garantiza la alineación de los signos de los vectores vecinos para evitar
    la cancelación por inversión de polaridad (signo antipodal).

    Parameters:
        pos  (np.ndarray):         Coordenadas continuas de forma (3,) -> [x, y, z].
        pdd_volume  (np.ndarray):  Volumen 4D del PDD de forma (X, Y, Z, 3).
        mask  (np.ndarray):        Máscara binaria del cerebro de forma (X, Y, Z).

    Returns:
        Optional[np.ndarray]
            Vector unitario interpolado de forma (3,) con tipo np.float32,
            o None si la posición está fuera de los límites o de la máscara.
    """
    # Obtener las esquinas enteras (caja de 8 vóxeles vecinos)
    x0, y0, z0 = np.floor(pos).astype(np.int32)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    # Validar límites del volumen para evitar IndexError
    if (
        x0 < 0
        or x1 >= pdd_volume.shape[0]
        or y0 < 0
        or y1 >= pdd_volume.shape[1]
        or z0 < 0
        or z1 >= pdd_volume.shape[2]
    ):
        return None

    # Verificar si todo el cubo indexado pertenece a la máscara del cerebro
    # Extrae el bloque de 2x2x2 y valida que todos los vóxeles sean mayores a 0
    brain_neighborhood = mask[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1]
    if not np.all(brain_neighborhood > 0):
        return None

    # Calcular los pesos de interpolación (distancias fraccionarias)
    fx = pos[0] - x0
    fy = pos[1] - y0
    fz = pos[2] - z0

    # Extraer los 8 vectores vecinos
    v000 = pdd_volume[x0, y0, z0]
    v100 = pdd_volume[x1, y0, z0]
    v010 = pdd_volume[x0, y1, z0]
    v110 = pdd_volume[x1, y1, z0]
    v001 = pdd_volume[x0, y0, z1]
    v101 = pdd_volume[x1, y0, z1]
    v011 = pdd_volume[x0, y1, z1]
    v111 = pdd_volume[x1, y1, z1]

    # Alineación de signos (Mitigación del problema del signo antipodal)
    # Tomamos v000 como referencia. Si el producto punto es negativo,
    # multiplicamos por -1 para alinear el sentido antes de interpolar.
    v100 *= np.sign(np.dot(v100, v000)) or 1.0
    v010 *= np.sign(np.dot(v010, v000)) or 1.0
    v110 *= np.sign(np.dot(v110, v000)) or 1.0
    v001 *= np.sign(np.dot(v001, v000)) or 1.0
    v101 *= np.sign(np.dot(v101, v000)) or 1.0
    v011 *= np.sign(np.dot(v011, v000)) or 1.0
    v111 *= np.sign(np.dot(v111, v000)) or 1.0

    # Combinación lineal trilineal estándar
    # Plano inferior (z0)
    c00 = v000 * (1.0 - fx) + v100 * fx
    c10 = v010 * (1.0 - fx) + v110 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy

    # Plano superior (z1)
    c01 = v001 * (1.0 - fx) + v101 * fx
    c11 = v011 * (1.0 - fx) + v111 * fx
    c1 = c01 * (1.0 - fy) + c11 * fy

    # Vector final interpolado
    v_interp = c0 * (1.0 - fz) + c1 * fz

    # Re-normalizar para asegurar que siga siendo un vector unitario d_t
    norm = np.linalg.norm(v_interp)
    if norm < 1e-6:
        return None  # Región con vectores nulos o cancelación por ruido extremo

    return (v_interp / norm).astype(np.float32)
