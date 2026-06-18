import pytest
import numpy as np
from src.dti.least_squares import calculate_pseudoinverse, fit_ols

def test_calculate_pseudoinverse_dimensions():
    """Valida que las dimensiones de la pseudoinversa sean las inversas exactas de A (6 x N)."""
    N = 64
    fake_A = np.random.rand(N, 6)
    
    A_pinv = calculate_pseudoinverse(fake_A)
    
    assert A_pinv.shape == (6, N), f"Se esperaba una dimensión (6, {N}), pero se obtuvo {A_pinv.shape}"

def test_fit_ols_exact_recovery():
    """
    Prueba analítica de Mínimos Cuadrados (OLS).
    Construye un sistema ideal B = A * d_true y verifica que fit_ols recupere
    exactamente el vector d_true original sin desviaciones.
    """
    # Definimos una matriz A artificial de rango completo (N=8, más de los 6 necesarios)
    A = np.array([
        [1.0, 0.2, 0.1, 0.5, 0.0, 0.1],
        [0.1, 1.0, 0.3, 0.0, 0.4, 0.2],
        [0.0, 0.1, 1.0, 0.2, 0.1, 0.5],
        [0.5, 0.0, 0.2, 1.0, 0.3, 0.0],
        [0.2, 0.4, 0.0, 0.1, 1.0, 0.3],
        [0.1, 0.0, 0.5, 0.3, 0.2, 1.0],
        [0.6, 0.1, 0.2, 0.1, 0.2, 0.1],
        [0.1, 0.5, 0.1, 0.6, 0.1, 0.2]
    ])
    
    # Definimos un vector de coeficientes del tensor real (d_true)
    # Siguiendo el orden teórico: [D11, D12, D13, D22, D23, D33]
    d_true = np.array([1.5e-3, 0.2e-3, 0.1e-3, 1.2e-3, 0.0, 0.9e-3])
    
    # Generamos el vector B teórico limpio (B = A * d_true)
    B = np.dot(A, d_true)
    
    # Calculamos pseudoinversa y ajustamos por OLS usando tus funciones
    A_pinv = calculate_pseudoinverse(A)
    d_estimated = fit_ols(A_pinv, B)
    
    # El vector estimado debe ser idéntico al real en precisión de punto flotante
    assert d_estimated.shape == (6,), "El vector de parámetros estimado debe contener exactamente 6 elementos."
    assert np.allclose(d_estimated, d_true, atol=1e-12), (
        f"El estimador OLS no recuperó el tensor exacto.\n"
        f"Esperado: {d_true}\nObtenido: {d_estimated}"
    )