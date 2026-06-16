import pytest
import numpy as np
from src.dti.design_matrix import build_design_matrix

def test_build_design_matrix_shape():
    """Valida que la matriz devuelta tenga las dimensiones correctas (N x 6)."""
    # Creamos un set aleatorio de 15 direcciones de gradiente
    fake_bvecs = np.random.rand(15, 3)
    
    A = build_design_matrix(fake_bvecs)
    
    assert A.shape == (15, 6), f"Se esperaba una forma (15, 6), pero se obtuvo {A.shape}"

def test_build_design_matrix_analytical_values():
    """
    Prueba analíticamente la construcción de la matriz A usando direcciones de
    gradiente base conocidas (Ejes principales y una diagonal a 45 grados).
    """
    # Diseñamos un conjunto de 4 gradientes bien conocidos:
    # 1. Dirección pura en X
    # 2. Dirección pura en Y
    # 3. Dirección pura en Z
    # 4. Dirección diagonal plana en el plano XY (1/sqrt(2), 1/sqrt(2), 0)
    bvecs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0]
    ])
    
    A = build_design_matrix(bvecs)
    
    # --- Fila 0: [1, 0, 0] -> gx^2=1, los demás componentes cruzados son 0
    # Esperado: [1, 0, 0, 0, 0, 0]
    expected_row_0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(A[0], expected_row_0), f"Error en dirección X pura. Obtenido: {A[0]}"
    
    # --- Fila 1: [0, 1, 0] -> gy^2=1, los demás componentes son 0
    # Esperado: [0, 0, 0, 1, 0, 0]
    expected_row_1 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(A[1], expected_row_1), f"Error en dirección Y pura. Obtenido: {A[1]}"
    
    # --- Fila 2: [0, 0, 1] -> gz^2=1, los demás componentes son 0
    # Esperado: [0, 0, 0, 0, 0, 1]
    expected_row_2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(A[2], expected_row_2), f"Error en dirección Z pura. Obtenido: {A[2]}"
    
    # --- Fila 3: [1/sqrt(2), 1/sqrt(2), 0] 
    # gx^2 = 0.5
    # 2*gx*gy = 2 * (1/sqrt(2)) * (1/sqrt(2)) = 2 * 0.5 = 1.0
    # gy^2 = 0.5
    # Componentes con Z son 0
    # Esperado: [0.5, 1.0, 0.0, 0.5, 0.0, 0.0]
    expected_row_3 = np.array([0.5, 1.0, 0.0, 0.5, 0.0, 0.0])
    assert np.allclose(A[3], expected_row_3), f"Error en dirección diagonal XY. Obtenido: {A[3]}"

def test_build_design_matrix_zeros():
    """Valida que un vector de ceros genere una fila de ceros sin colapsar."""
    bvecs = np.array([[0.0, 0.0, 0.0]])
    A = build_design_matrix(bvecs)
    
    assert np.all(A == 0.0), "Un gradiente nulo debería producir coeficientes nulos en la matriz de diseño."