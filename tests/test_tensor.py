import pytest
import numpy as np
from src.dti.tensor import calculate_B_vector

def test_calculate_B_vector_analytical():
    """Valida el cálculo correcto del vector B bajo condiciones ideales."""
    S0 = 100.0
    # Simulamos 3 señales donde sabemos el resultado exacto analíticamente
    # Si = S0 * exp(-bi * B_esperado)
    # Si bi = 3000 y B_esperado = 0.001 -> Si = 100 * exp(-3) = 100 * 0.049787...
    bi = np.array([3000.0, 3000.0, 3000.0])
    Si = np.array([
        S0 * np.exp(-3000.0 * 0.001),  # B debería ser 0.001
        S0 * np.exp(-3000.0 * 0.002),  # B debería ser 0.002
        S0 * np.exp(-3000.0 * 0.000)   # B debería ser 0.0
    ])
    
    B = calculate_B_vector(S0, Si, bi)
    
    expected_B = np.array([0.001, 0.002, 0.0])
    
    assert B.shape == Si.shape, "El vector B debe mantener la misma dimensión que Si."
    assert np.allclose(B, expected_B, atol=1e-7), f"El cálculo matemático falló. Obtenido: {B}"

def test_calculate_B_vector_protection_Si_negative_or_zero():
    """Valida que señales Si <= 0 causadas por ruido se manejen de forma segura sin colapsar."""
    S0 = 100.0
    bi = np.array([3000.0, 3000.0, 3000.0])
    # Forzamos un valor normal, uno cero y uno negativo (ruido extremo)
    Si = np.array([S0 * np.exp(-3), 0.0, -10.5])
    
    # Si no tuviera protección, np.log() lanzaría un RuntimeWarning o error
    B = calculate_B_vector(S0, Si, bi)
    
    # El primer elemento debe ser correcto, los inválidos deben quedarse en 0.0 de forma segura
    assert np.isclose(B[0], 0.001, atol=1e-7)
    assert B[1] == 0.0, f"Se esperaba un fallback seguro a 0.0 para Si=0, pero se obtuvo {B[1]}"
    assert B[2] == 0.0, f"Se esperaba un fallback seguro a 0.0 para Si<0, pero se obtuvo {B[2]}"

def test_calculate_B_vector_protection_S0_zero():
    """Valida que si S0 es 0 (vóxel de fondo fuera del cerebro), devuelva un vector de ceros seguro."""
    S0 = 0.0
    bi = np.array([3000.0, 3000.0])
    Si = np.array([50.0, 40.0])
    
    B = calculate_B_vector(S0, Si, bi)
    
    assert np.all(B == 0.0), "Si S0 es cero, todo el vector B resultante debe ser un arreglo seguro de ceros."