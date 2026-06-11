# src/dti/design_matrix.py
import numpy as np

def build_design_matrix(bvecs):
    """
    Construye la matriz de diseño A (N x 6) a partir de los vectores de gradiente.
    
    Basado en el modelo donde:
    a_1 = gx^2, a_2 = 2*gx*gy, a_3 = 2*gx*gz, 
    a_4 = gy^2, a_5 = 2*gy*gz, a_6 = gz^2
    
    Parameters:
        bvecs (numpy.ndarray): Arreglo de vectores de gradiente de forma (N, 3).
                               Debe excluir la dirección correspondiente a S0 (b=0).
                               
    Returns:
        numpy.ndarray: Matriz de diseño A de forma (N, 6).
    """
    # Extraer las componentes espaciales de los gradientes
    gx = bvecs[:, 0]
    gy = bvecs[:, 1]
    gz = bvecs[:, 2]
    
    # Inicializar la matriz A
    N = bvecs.shape[0]
    A = np.zeros((N, 6))
    
    # Asignar las columnas según la derivación teórica
    A[:, 0] = gx**2
    A[:, 1] = 2 * gx * gy
    A[:, 2] = 2 * gx * gz
    A[:, 3] = gy**2
    A[:, 4] = 2 * gy * gz
    A[:, 5] = gz**2
    
    return A