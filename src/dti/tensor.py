# src/dti/tensor.py
import numpy as np

def calculate_B_vector(S0, Si, bi):
    """
    Calcula el vector B para un vóxel dado resolviendo:
    B_i = -ln(S_i / S_0) / b_i
    
    Incluye protecciones contra ruido en la imagen (S0 nulo o Si <= 0)
    para evitar errores de dominio matemático en el logaritmo.
    
    Parameters:
        S0 (float): Señal del vóxel sin ponderación de difusión (b=0).
        Si (numpy.ndarray): Arreglo con las señales ponderadas (N,).
        bi (numpy.ndarray): Arreglo con los b-values correspondientes (N,).
        
    Returns:
        numpy.ndarray: Vector B de forma (N,).
    """
    # Inicializar el vector B con ceros
    B = np.zeros_like(Si, dtype=float)
    
    # Solo procedemos si tenemos una señal base válida
    if S0 > 0:
        # Encontrar los índices donde la señal Si es estrictamente positiva
        # (El logaritmo natural de 0 o números negativos está indefinido)
        valid_mask = Si > 0
        
        # Calcular el vector B solo en los puntos válidos
        # Se asume que bi tampoco es 0 porque ya filtramos la imagen S0
        B[valid_mask] = -np.log(Si[valid_mask] / S0) / bi[valid_mask]
        
    return B