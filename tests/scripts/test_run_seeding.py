# tests/scripts/test_run_seeding.py
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from scripts.run_seeding import main

@patch('scripts.run_seeding.Path.exists')
@patch('nibabel.load')
@patch('scripts.run_seeding.generate_seeds')
@patch('scripts.run_seeding.save_seeds')
def test_run_seeding_pipeline_flow(
    mock_save_seeds, mock_generate_seeds, mock_nib_load, mock_exists
):
    """Prueba de flujo completo para el script ejecutable de siembra.

    Asegura que el script cargue los archivos correctos de imágenes médicas,
    desempaquete los hiperparámetros de configuración del diccionario y guarde
    la matriz resultante en la ruta procesada estipulada.
    """
    # Simular que ambos archivos (FA y Máscara) existen en el sistema de archivos
    mock_exists.return_value = True

    # Configurar mocks para los objetos NiftiImage retornados por nibabel
    mock_fa_img = MagicMock()
    mock_mask_img = MagicMock()

    # Configurar los retornos de get_fdata() simulando arreglos de NumPy
    fake_fa_data = np.zeros((4, 4, 4), dtype=np.float32)
    fake_mask_data = np.ones((4, 4, 4), dtype=np.float32)

    mock_fa_img.get_fdata.return_value = fake_fa_data
    mock_mask_img.get_fdata.return_value = fake_mask_data

    # Configurar nib.load para que devuelva los mocks correspondientes en orden de llamada
    mock_nib_load.side_effect = [mock_fa_img, mock_mask_img]

    # Configurar retorno ficticio para el generador de semillas
    mock_generate_seeds.return_value = np.array(
        [[1.25, 1.25, 1.25]], dtype=np.float32
    )

    # Ejecutar la función principal del script
    main()

    # VERIFICACIONES DE INTEGRACIÓN (ASSERTS)

    # Comprobar que nib.load cargó los archivos NIfTI con las rutas correctas
    expected_fa_path = str(
        Path('data/processed/metrics/ISMRM_2023_b3000_FA.nii.gz')
    )
    expected_mask_path = str(Path('data/raw/ISMRM_2023_b3000_mask.nii'))

    mock_nib_load.assert_any_call(expected_fa_path)
    mock_nib_load.assert_any_call(expected_mask_path)

    # Verificar que se extrajeron los datos optimizando el tipo flotante de memoria
    mock_fa_img.get_fdata.assert_called_once_with(dtype=np.float32)
    mock_mask_img.get_fdata.assert_called_once_with(dtype=np.float32)

    # Verificar el desempaquetado exacto de los parámetros de `config` hacia `generate_seeds`
    mock_generate_seeds.assert_called_once_with(
        mask=fake_mask_data,
        fa_volume=fake_fa_data,
        fa_thresh=0.20,
        method='grid',
        density=2,
        n_seeds_random=8000,
        random_seed=42,
    )

    # Verificar que las semillas se guardaron en formato binario en la ruta adecuada
    expected_output_path = str(Path('data/processed/seeds/seeds.npy'))
    mock_save_seeds.assert_called_once_with(
        mock_generate_seeds.return_value, expected_output_path
    )


@patch('scripts.run_seeding.Path.exists')
@patch('scripts.run_seeding.generate_seeds')
def test_run_seeding_early_return_if_files_missing(
    mock_generate_seeds, mock_exists
):
    """Valida que el script termine de manera temprana y segura si los mapas

    previos de DTI no se encuentran en el disco.
    """
    # Forzar a que falle la existencia de archivos
    mock_exists.return_value = False

    # Ejecutar main
    main()

    # Verificar que el pipeline abortó y jamás intentó calcular las semillas
    mock_generate_seeds.assert_not_called()