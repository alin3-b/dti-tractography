import pytest
import numpy as np
from src.dti.estimation import fit_tensor_model

def test_fit_tensor_model_integration():
    """
    Test de integración para fit_tensor_model.
    Verifica que el ciclo orquestador procese los voxeles bajo la máscara,
    ignore el fondo y ordene correctamente los canales del tensor resultante.
    """
    # Definir dimensiones pequeñas para el test (2x2x2 voxeles)
    # Usaremos 7 volúmenes en total (1 volumen b0 y 6 direcciones de gradiente)
    nx, ny, nz = 2, 2, 2
    n_volumes = 7
    
    # 2. Configurar b-values y b-vectors sintéticos
    bvals = np.array([0.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0])
    bvecs = np.array([
        [0.0, 0.0, 0.0],  # Corresponde a s0_index = 0
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0],
        [1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2)],
        [0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
    ])
    
    # Diseñar un tensor teórico conocido que queremos recuperar
    # Orden deseado final: [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    # d_hat intermedio de mínimos cuadrados: [Dxx, 2Dxy, 2Dxz, Dyy, 2Dyz, Dzz]
    Dxx, Dyy, Dzz = 1.5e-3, 1.0e-3, 0.8e-3
    Dxy, Dxz, Dyz = 0.2e-3, 0.1e-3, 0.3e-3
    expected_tensor = np.array([Dxx, Dyy, Dzz, Dxy, Dxz, Dyz], dtype=np.float32)
    
    # Reconstruimos la señal exacta analíticamente usando el modelo físico para las 6 direcciones activas:
    # Si = S0 * exp(-bi * (gx^2*Dxx + 2gxgy*Dxy + 2gxgz*Dxz + gy^2*Dyy + 2gygz*Dyz + gz^2*Dzz))
    S0_val = 100.0
    signals = []
    for g in bvecs[1:]:
        exponent = (g[0]**2 * Dxx + 
                    2 * g[0] * g[1] * Dxy + 
                    2 * g[0] * g[2] * Dxz + 
                    g[1]**2 * Dyy + 
                    2 * g[1] * g[2] * Dyz + 
                    g[2]**2 * Dzz)
        signals.append(S0_val * np.exp(-3000.0 * exponent))
        
    # Construir el vector completo de señales para un vóxel activo (b0 + 6 direcciones)
    voxel_signals = np.array([S0_val] + signals)
    
    # Poblar la matriz de datos DWI 4D
    dwi_data = np.zeros((nx, ny, nz, n_volumes))
    # Ponemos la señal sintética únicamente en el vóxel [1, 1, 1]
    dwi_data[1, 1, 1, :] = voxel_signals
    
    # Configurar la máscara: Solo activamos el vóxel [1, 1, 1]. El resto [0,0,0] será fondo.
    mask = np.zeros((nx, ny, nz), dtype=np.int32)
    mask[1, 1, 1] = 1
    
    # Ejecutar la estimación completa
    DT = fit_tensor_model(dwi_data, mask, bvals, bvecs, s0_index=0)
    
 
    # VERIFICACIONES (ASSERTS)
    # A. Comprobar dimensiones del output
    assert DT.shape == (nx, ny, nz, 6), f"Forma inesperada del tensor: {DT.shape}"
    assert DT.dtype == np.float32, "El tipo de dato del tensor debe ser obligatoriamente float32."
    
    # B. Comprobar que las regiones desmarcadas se mantuvieron vacías (ceros)
    assert np.all(DT[0, 0, 0, :] == 0.0), "Los vóxeles fuera de la máscara deben permanecer en cero."
    
    # C. Comprobar la precisión del mapeo de canales del tensor recuperado
    assert np.allclose(DT[1, 1, 1, :], expected_tensor, atol=1e-6), (
        f"El tensor recuperado en el vóxel activo falló en el mapeo u ordenación.\n"
        f"Esperado: {expected_tensor}\nObtenido: {DT[1, 1, 1, :]}"
    )