# src/dti/metrics.py
import numpy as np

def calculate_dti_metrics(DT, mask):
    """
    Calcula mapas de métricas escalares (FA y MD) a partir del tensor de difusión.
    
    Parameters:
        DT (numpy.ndarray): Tensor de difusión de forma (X, Y, Z, 6).
                            Orden: [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz].
        mask (numpy.ndarray): Máscara binaria del cerebro de forma (X, Y, Z).
        
    Returns:
        tuple: (FA, MD) Mapas de Anisotropía Fraccional y Difusividad Media.
    """
    nx, ny, nz, _ = DT.shape
    
    # Inicializar mapas de resultados
    FA = np.zeros((nx, ny, nz), dtype=np.float32)
    MD = np.zeros((nx, ny, nz), dtype=np.float32)
    
    print("Calculando métricas del tensor (FA y MD)...")
    
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if mask[x, y, z] > 0:
                    
                    # Reconstruir la matriz simétrica 3x3 del tensor
                    # Recordando el orden: [Dxx(0), Dyy(1), Dzz(2), Dxy(3), Dxz(4), Dyz(5)]
                    D_matrix = np.array([
                        [DT[x, y, z, 0], DT[x, y, z, 3], DT[x, y, z, 4]],
                        [DT[x, y, z, 3], DT[x, y, z, 1], DT[x, y, z, 5]],
                        [DT[x, y, z, 4], DT[x, y, z, 5], DT[x, y, z, 2]]
                    ])
                    
                    # Calcular valores propios (eigenvalores)
                    # np.linalg.eigvalsh es más rápido y estable para matrices simétricas
                    evals = np.linalg.eigvalsh(D_matrix)
                    
                    # Proteger contra eigenvalores negativos provocados por el ruido
                    evals = np.maximum(evals, 0)
                    
                    l1, l2, l3 = evals[0], evals[1], evals[2]
                    trace = l1 + l2 + l3
                    
                    # Calcular MD (Difusividad Media)
                    if trace > 0:
                        md = trace / 3.0
                        MD[x, y, z] = md
                        
                        # Calcular FA (Anisotropía Fraccional)
                        num = np.sqrt((l1 - md)**2 + (l2 - md)**2 + (l3 - md)**2)
                        den = np.sqrt(l1**2 + l2**2 + l3**2)
                        
                        if den > 0:
                            FA[x, y, z] = np.sqrt(1.5) * (num / den)
                            
    print("Cálculo de métricas finalizado.")
    return FA, MD