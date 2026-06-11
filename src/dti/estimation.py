# src/dti/estimation.py
import numpy as np
from .design_matrix import build_design_matrix
from .least_squares import calculate_pseudoinverse, fit_ols
from .tensor import calculate_B_vector


def fit_tensor_model(dwi_data, mask, bvals, bvecs, s0_index=0):
    """
    Ajusta el modelo del Tensor de Difusión (DTI) vóxel por vóxel.
    
    Parameters:
        dwi_data (numpy.ndarray): Datos DWI de forma (X, Y, Z, N).
        mask (numpy.ndarray): Máscara binaria de forma (X, Y, Z).
        bvals (numpy.ndarray): Vector de b-values.
        bvecs (numpy.ndarray): Matriz de vectores de gradiente (N, 3).
        s0_index (int): Índice del volumen b=0 (S0).

    Returns:
        numpy.ndarray: Tensor de difusión de forma (X, Y, Z, 6) en orden:
                       [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    """
    nx, ny, nz, n_volumes = dwi_data.shape
    
    # Índices de las direcciones de difusión (excluyendo S0)
    indices_Si = [i for i in range(n_volumes) if i != s0_index]
    
    # Filtrar b-values y b-vectors
    bi = bvals[indices_Si]
    g_i = bvecs[indices_Si]
    
    # Precomputar matriz de diseño y su pseudoinversa (una sola vez)
    A = build_design_matrix(g_i)
    A_pinv = calculate_pseudoinverse(A)
    
    # Inicializar volumen de tensores (float32 para optimizar memoria)
    DT = np.zeros((nx, ny, nz, 6), dtype=np.float32)
    
    print("Iniciando estimación del tensor de difusión...")
    
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if mask[x, y, z] <= 0:
                    continue
                
                S0 = dwi_data[x, y, z, s0_index]
                Si = dwi_data[x, y, z, indices_Si]
                
                # Calcular vector B (con protección contra ruido)
                B_vector = calculate_B_vector(S0, Si, bi)
                
                # Estimación por mínimos cuadrados
                d_hat = fit_ols(A_pinv, B_vector)
                
                # Mapeo correcto al orden estándar [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
                # d_hat viene como: [d1=Dxx, d2=Dxy, d3=Dxz, d4=Dyy, d5=Dyz, d6=Dzz]
                DT[x, y, z, 0] = d_hat[0]   # Dxx
                DT[x, y, z, 1] = d_hat[3]   # Dyy
                DT[x, y, z, 2] = d_hat[5]   # Dzz
                DT[x, y, z, 3] = d_hat[1]   # Dxy
                DT[x, y, z, 4] = d_hat[2]   # Dxz
                DT[x, y, z, 5] = d_hat[4]   # Dyz
    
    print("Estimación del tensor finalizada exitosamente.")
    return DT