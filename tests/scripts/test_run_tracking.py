# tests/scripts/test_run_tracking.py
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from scripts.run_tracking import main


@patch('scripts.run_tracking.Path.exists')
@patch('scripts.run_tracking.load_nifti')
@patch('scripts.run_tracking.load_seeds')
@patch('scripts.run_tracking.track_multiple_seeds')
@patch('scripts.run_tracking.save_tractogram')
def test_run_tracking_pipeline_flow(
    mock_save_tractogram,
    mock_track_seeds,
    mock_load_seeds,
    mock_load_nifti,
    mock_exists,
):
    """Prueba de flujo completo para el script ejecutable de tractografía.

    Asegura que el pipeline valide las dependencias en disco, cargue de forma
    optimizada los tensores y mapas microestructurales, pase los parámetros del
    algoritmo mediante desempaquetado y exporte el archivo final .trk.
    """
    # 1. Forzar a que la validación defensiva detecte todos los archivos obligatorios
    mock_exists.return_value = True

    # 2. Configurar arreglos simulados en memoria para los retornos de I/O
    fake_pdd = np.zeros((4, 4, 4, 3), dtype=np.float32)
    fake_fa = np.ones((4, 4, 4), dtype=np.float32) * 0.4
    fake_mask = np.ones((4, 4, 4), dtype=bool)
    fake_seeds = np.array([[1.5, 1.5, 1.5]], dtype=np.float32)

    # Configurar cargas secuenciales simuladas
    mock_load_nifti.side_effect = [
        (fake_pdd, None, None),
        (fake_fa, None, None),
        (fake_mask, None, None),
    ]
    mock_load_seeds.return_value = fake_seeds

    # 3. Configurar el retorno del motor de propagación del tracking
    fake_streamlines = [np.array([[1.5, 1.5, 1.5]], dtype=np.float32)]
    mock_track_seeds.return_value = fake_streamlines

    # 4. Ejecutar la función principal del script ejecutable
    main()

    # VERIFICACIONES DE INTEGRACIÓN (ASSERTS)

    # A. Corroborar que se cargaron exactamente los 3 volúmenes NIfTI y las semillas mapeadas
    assert mock_load_nifti.call_count == 3
    mock_load_seeds.assert_called_once_with(
        str(Path('data/processed/seeds/seeds.npy'))
    )

    # B. Verificar que el algoritmo orquestador se invocó con la configuración técnica configurada
    mock_track_seeds.assert_called_once_with(
        seeds=fake_seeds,
        pdd_volume=fake_pdd,
        mask=fake_mask,
        fa_volume=fake_fa,
        step_size=0.5,
        max_steps=1200,
        angle_threshold=45.0,
        fa_threshold=0.15,
        min_length=15.0,
        verbose=True,
    )

    # C. Verificar el anclaje geométrico y el almacenamiento del tractograma en disco
    expected_ref_path = str(Path('data/raw/ISMRM_2023_b3000_mask.nii'))
    expected_out_path = str(Path('data/processed/tracts/tractogram.trk'))

    mock_save_tractogram.assert_called_once_with(
        streamlines=fake_streamlines,
        reference_path=expected_ref_path,
        output_path=expected_out_path,
    )


@patch('scripts.run_tracking.Path.exists')
@patch('scripts.run_tracking.track_multiple_seeds')
def test_run_tracking_early_return_if_dependencies_missing(
    mock_track_seeds, mock_exists
):
    """Valida el comportamiento defensivo del script abortando de forma segura

    si el pipeline de ajuste previo o de siembra no se completó con éxito.
    """
    # Simular que los archivos de métricas previas o semillas no existen en el disco
    mock_exists.return_value = False

    # Ejecutar main
    main()

    # Garantizar que el script se interrumpió tempranamente sin procesar datos
    mock_track_seeds.assert_not_called()