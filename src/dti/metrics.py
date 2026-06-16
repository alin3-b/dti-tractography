# src/dti/metrics.py
import numpy as np

def calculate_dti_metrics(
    dt_volume: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula mapas de métricas escalares (FA, MD) y vectoriales (PDD)
    a partir del tensor de difusión de forma optimizada y vectorizada.

    Parameters:
        dt_volume (np.ndarray): Tensor de difusión de forma (X, Y, Z, 6).
                                Orden de componentes: [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz].
        mask (np.ndarray): Máscara binaria del cerebro de forma (X, Y, Z).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - FA: Anisotropía Fraccional de forma (X, Y, Z).
            - MD: Difusividad Media de forma (X, Y, Z).
            - PDD: Dirección Principal de Difusión de forma (X, Y, Z, 3).
    """
    shape_3d = dt_volume.shape[:3]

    # Inicializar los arreglos de salida
    fa = np.zeros(shape_3d, dtype=np.float32)
    md = np.zeros(shape_3d, dtype=np.float32)
    pdd = np.zeros((*shape_3d, 3), dtype=np.float32)

    # Identificar los índices donde la máscara está activa
    idx_mask = mask > 0
    if not np.any(idx_mask):
        return fa, md, pdd

    # Extraer solo los tensores que están dentro de la máscara
    # d_voxels tendrá forma (N_voxels, 6)
    d_voxels = dt_volume[idx_mask]
    n_voxels = d_voxels.shape[0]

    # Reconstruir las matrices simétricas 3x3 de forma vectorizada
    # d_matrices tendrá forma (N_voxels, 3, 3)
    d_matrices = np.zeros((n_voxels, 3, 3), dtype=np.float64)
    d_matrices[:, 0, 0] = d_voxels[:, 0]  # Dxx
    d_matrices[:, 1, 1] = d_voxels[:, 1]  # Dyy
    d_matrices[:, 2, 2] = d_voxels[:, 2]  # Dzz

    d_matrices[:, 0, 1] = d_matrices[:, 1, 0] = d_voxels[:, 3]  # Dxy
    d_matrices[:, 0, 2] = d_matrices[:, 2, 0] = d_voxels[:, 4]  # Dxz
    d_matrices[:, 1, 2] = d_matrices[:, 2, 1] = d_voxels[:, 5]  # Dyz

    # Calcular autovalores y autovectores en bloque
    # eigvals: (N_voxels, 3), eigvecs: (N_voxels, 3, 3)
    eigvals, eigvecs = np.linalg.eigh(d_matrices)

    # Proteger contra ruido (valores negativos)
    eigvals = np.maximum(eigvals, 0.0)

    # np.linalg.eigh entrega orden ascendente.
    # l1 (principal) es el último índice (2), l3 el primero (0)
    l3 = eigvals[:, 0]
    l2 = eigvals[:, 1]
    l1 = eigvals[:, 2]

    # Calcular Difusividad Media (MD)
    trace = l1 + l2 + l3
    md_voxels = trace / 3.0
    md[idx_mask] = md_voxels

    # Calcular Anisotropía Fraccional (FA)
    # Evitar divisiones por cero usando un umbral de tolerancia
    denominator = np.sqrt(l1**2 + l2**2 + l3**2)
    valid_den = denominator > 1e-10

    num = np.sqrt(
        (l1 - md_voxels) ** 2 + (l2 - md_voxels) ** 2 + (l3 - md_voxels) ** 2
    )

    fa_voxels = np.zeros(n_voxels, dtype=np.float32)
    fa_voxels[valid_den] = np.sqrt(1.5) * (num[valid_den] / denominator[valid_den])
    fa[idx_mask] = fa_voxels

    # Extraer la Dirección Principal de Difusión (PDD)
    # Corresponde al autovector asociado a l1 (última columna de cada matriz)
    v1 = eigvecs[:, :, 2]

    # Normalizar los vectores
    norm = np.linalg.norm(v1, axis=1, keepdims=True)
    valid_norm = norm[:, 0] > 1e-10

    pdd_voxels = np.zeros((n_voxels, 3), dtype=np.float32)
    pdd_voxels[valid_norm] = v1[valid_norm] / norm[valid_norm]
    pdd[idx_mask] = pdd_voxels

    return fa, md, pdd