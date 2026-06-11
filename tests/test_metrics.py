import pytest
import numpy as np
from src.dti.metrics import calculate_dti_metrics

def test_calculate_dti_metrics_isotropy():
    """
    Prueba un comportamiento puramente isotrópico (esférico).
    Si los tres eigenvalores son idénticos, la FA debe ser 0.0
    y la MD debe ser igual al valor común de los eigenvalores.
    """
    # Creamos un volumen de 1x1x1 vóxeles
    # Tensor isotrópico donde Dxx = Dyy = Dzz = 2.0e-3 y los cruzados son 0
    # Orden: [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    DT = np.array([[[[2.0e-3, 2.0e-3, 2.0e-3, 0.0, 0.0, 0.0]]]], dtype=np.float32)
    mask = np.array([[[1]]], dtype=np.int32)
    
    FA, MD = calculate_dti_metrics(DT, mask)
    
    # Assertions
    assert np.isclose(FA[0, 0, 0], 0.0, atol=1e-6), f"Se esperaba FA=0 para un tensor isotrópico, se obtuvo {FA[0,0,0]}"
    assert np.isclose(MD[0, 0, 0], 2.0e-3, atol=1e-6), f"Se esperaba MD=2.0e-3, se obtuvo {MD[0,0,0]}"

def test_calculate_dti_metrics_extreme_anisotropy():
    """
    Prueba un comportamiento de anisotropía lineal extrema (forma de aguja).
    Si el tensor difunde solo en un eje (l1 = 3.0e-3, l2=0, l3=0),
    la FA analítica exacta debe ser 1.0 y la MD debe ser l1 / 3.
    """
    # Configuramos un tensor puramente anisotrópico en el eje X: Dxx = 3.0e-3, los demás 0
    DT = np.array([[[[3.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32)
    mask = np.array([[[1]]], dtype=np.int32)
    
    FA, MD = calculate_dti_metrics(DT, mask)
    
    # Assertions
    assert np.isclose(FA[0, 0, 0], 1.0, atol=1e-6), f"Se esperaba FA=1 para anisotropía extrema, se obtuvo {FA[0,0,0]}"
    assert np.isclose(MD[0, 0, 0], 1.0e-3, atol=1e-6), f"Se esperaba MD=1.0e-3, se obtuvo {MD[0,0,0]}"

def test_calculate_dti_metrics_mask_and_dimensions():
    """Valida que se respeten las dimensiones físicas y las zonas excluidas por la máscara."""
    nx, ny, nz = 2, 2, 2
    # Llenamos todo el volumen con tensores isotrópicos válidos
    DT = np.zeros((nx, ny, nz, 6), dtype=np.float32)
    DT[:, :, :, 0:3] = 1.5e-3  # Dxx, Dyy, Dzz activos en todo el bloque
    
    # Activamos la máscara únicamente en el primer vóxel [0,0,0]
    mask = np.zeros((nx, ny, nz), dtype=np.int32)
    mask[0, 0, 0] = 1
    
    FA, MD = calculate_dti_metrics(DT, mask)
    
    # Comprobar dimensiones de los mapas de salida
    assert FA.shape == (nx, ny, nz)
    assert MD.shape == (nx, ny, nz)
    
    # El vóxel activo bajo la máscara debe estar calculado (Isotrópico -> FA=0, MD=1.5e-3)
    assert np.isclose(MD[0, 0, 0], 1.5e-3)
    
    # El resto de los vóxeles que están fuera de la máscara deben quedarse estrictamente en 0.0
    assert FA[1, 1, 1] == 0.0
    assert MD[1, 1, 1] == 0.0