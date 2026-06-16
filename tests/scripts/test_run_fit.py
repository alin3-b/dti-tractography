# tests/scripts/test_run_fit.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Importamos la función main del script ejecutable
from scripts.run_fit import main


@patch('scripts.run_fit.load_dwi_data')
@patch('scripts.run_fit.fit_tensor_model')
@patch('scripts.run_fit.calculate_dti_metrics')
@patch('scripts.run_fit.save_nifti')
def test_main_pipeline_execution_flow(
    mock_save_nifti, mock_calculate_metrics, mock_fit_tensor, mock_load_data
):
    """Test de flujo para el script principal.

    Asegura que el pipeline cargue, estime, calcule métricas y guarde los
    4 archivos resultantes (DT, FA, MD, PDD) con las rutas y carpetas
    especificas configuradas en el script ejecutable.
    """
    # Configurar el retorno simulado del cargador de datos
    fake_affine = np.eye(4)
    fake_header = 'FakeNiftiHeader'
    mock_load_data.return_value = {
        'data': np.zeros((2, 2, 2, 7)),
        'mask': np.ones((2, 2, 2)),
        'bvals': np.array([0, 3000]),
        'bvecs': np.array([[0, 0, 0], [1, 0, 0]]),
        'affine': fake_affine,
        'header': fake_header,
    }

    # Configurar retornos ficticios para el estimador y las métricas
    mock_fit_tensor.return_value = 'Mocked_DT_Volume'
    mock_calculate_metrics.return_value = (
        'Mocked_FA_Map',
        'Mocked_MD_Map',
        'Mocked_PDD_Map',
    )

    # Ejecutar el script principal
    main()

    # VERIFICACIONES DE INTEGRACIÓN (ASSERTS)

    # Verificar que los datos se buscaron en las rutas correctas de la ISMRM
    mock_load_data.assert_called_once_with(
        dwi_path='data/raw/ISMRM_2023_b3000.nii.gz',
        bval_path='data/raw/ISMRM_2023_b3000.bval',
        bvec_path='data/raw/ISMRM_2023_b3000.bvec',
        mask_path='data/raw/ISMRM_2023_b3000_mask.nii',
    )

    # Verificar que el estimador se ejecutó con los datos del dataset cargado
    mock_fit_tensor.assert_called_once_with(
        dwi_data=mock_load_data.return_value['data'],
        mask=mock_load_data.return_value['mask'],
        bvals=mock_load_data.return_value['bvals'],
        bvecs=mock_load_data.return_value['bvecs'],
        s0_index=0,
    )

    # Verificar que las métricas se calcularon usando el Tensor estimado y la máscara
    mock_calculate_metrics.assert_called_once_with(
        'Mocked_DT_Volume', mock_load_data.return_value['mask']
    )

    # Verificar que save_nifti se llamó exactamente 4 veces (para DT, FA, MD y PDD)
    assert mock_save_nifti.call_count == 4

    # Inspeccionar las llamadas individuales de guardado para corroborar rutas y metadatos
    calls = mock_save_nifti.call_args_list

    # Primera llamada: Debe guardar el Tensor Completo (DT) en la carpeta /tensors/
    assert calls[0][1]['data'] == 'Mocked_DT_Volume'
    assert (
        calls[0][1]['output_path']
        == 'data/processed/tensors/ISMRM_2023_b3000_DT.nii.gz'
    )
    assert np.array_equal(calls[0][1]['affine'], fake_affine)
    assert calls[0][1]['header'] == fake_header

    # Segunda llamada: Debe guardar la Anisotropía Fraccional (FA) en la carpeta /metrics/
    assert calls[1][1]['data'] == 'Mocked_FA_Map'
    assert (
        calls[1][1]['output_path']
        == 'data/processed/metrics/ISMRM_2023_b3000_FA.nii.gz'
    )
    assert np.array_equal(calls[1][1]['affine'], fake_affine)
    assert calls[1][1]['header'] == fake_header

    # Tercera llamada: Debe guardar la Difusividad Media (MD) en la carpeta /metrics/
    assert calls[2][1]['data'] == 'Mocked_MD_Map'
    assert (
        calls[2][1]['output_path']
        == 'data/processed/metrics/ISMRM_2023_b3000_MD.nii.gz'
    )
    assert np.array_equal(calls[2][1]['affine'], fake_affine)
    assert calls[2][1]['header'] == fake_header

    # Cuarta llamada: Debe guardar el mapa de dirección (PDD) en la carpeta /metrics/
    assert calls[3][1]['data'] == 'Mocked_PDD_Map'
    assert (
        calls[3][1]['output_path']
        == 'data/processed/metrics/ISMRM_2023_b3000_PDD.nii.gz'
    )
    assert np.array_equal(calls[3][1]['affine'], fake_affine)
    assert calls[3][1]['header'] == fake_header