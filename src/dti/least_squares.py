# src/dti/least_squares.py
import numpy as np

def calculate_pseudoinverse(A):
    """
    Calcula la pseudoinversa de Moore-Penrose para la matriz de diseño.
    Matemáticamente corresponde a: (A^T A)^-1 A^T
    
    Parameters:
        A (numpy.ndarray): Matriz de diseño de forma (N, 6).
        
    Returns:
        numpy.ndarray: Matriz pseudoinversa de forma (6, N).
    """
    return np.linalg.pinv(A)

def fit_ols(A_pinv, B):
    """
    Estima el vector de parámetros d^ resolviendo el sistema lineal
    por Mínimos Cuadrados Ordinarios (Ordinary Least Squares).
    
    Ecuación: d^ = (A^T A)^-1 A^T B = A_pinv * B
    
    Parameters:
        A_pinv (numpy.ndarray): Pseudoinversa de la matriz A (6, N).
        B (numpy.ndarray): Vector B de observaciones para un vóxel (N,).
        
    Returns:
        numpy.ndarray: Vector d^ con los 6 elementos del tensor de difusión.
    """
    # El producto punto entre la pseudoinversa y el vector B nos da los coeficientes
    d_hat = np.dot(A_pinv, B)
    
    return d_hat