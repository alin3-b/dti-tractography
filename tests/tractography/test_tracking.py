# tests/tractography/test_tracking.py
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from src.io.save_data import save_tractogram
from src.tractography.tracking import (
    load_tractography_data,
    track_multiple_seeds,
)


@pytest.fixture
def fake_tracking_inputs():
    """Fixture que provee datos y volúmenes mínimos para la orquestación del tracking."""
    seeds = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    shape = (4, 4, 4)
    mask = np.ones(shape, dtype=np.int32)
    fa_volume = np.ones(shape, dtype=np.float32) * 0.4
    pdd_volume = np.zeros((*shape, 3), dtype=np.float32)
    pdd_volume[:, :, :] = [1.0, 0.0, 0.0]

    return seeds, pdd_volume, mask, fa_volume


def test_track_multiple_seeds_filters_by_length(fake_tracking_inputs):
    """Valida el cálculo integrado de la longitud euclidiana y el filtrado de ruido."""
    seeds, pdd_volume, mask, fa_volume = fake_tracking_inputs

    with patch(
        'src.tractography.tracking.track_full_streamline'
    ) as mock_track_streamline:
        # Streamline 1: Suficientemente larga (recorre un segmento largo en el eje X)
        # Diferencias consecutivas: [1.0, 0.0, 0.0] -> norma = 1.0. Suma total = 2.0 mm
        long_streamline = np.array(
            [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
            dtype=np.float32,
        )

        # Streamline 2: Demasiado corta (fragmento espurio)
        # Diferencia: [0.1, 0.0, 0.0] -> norma = 0.1 mm
        short_streamline = np.array(
            [[2.0, 2.0, 2.0], [2.1, 2.0, 2.0]], dtype=np.float32
        )

        # Configurar el mock para retornar la larga y luego la corta
        mock_track_streamline.side_effect = [long_streamline, short_streamline]

        streamlines = track_multiple_seeds(
            seeds=seeds,
            pdd_volume=pdd_volume,
            mask=mask,
            fa_volume=fa_volume,
            min_length=1.0,  # Exige mínimo 1.0 mm de longitud integrada
            verbose=False,
        )

        # Debe conservar únicamente la línea larga (1 de las 2 procesadas)
        assert len(streamlines) == 1
        assert np.array_equal(streamlines[0], long_streamline)


@patch('nibabel.load')
@patch('dipy.io.streamline.save_tractogram')
@patch('dipy.tracking.streamline.Streamlines')
def test_save_tractogram_integration(
    mock_dipy_streamlines, mock_dipy_save, mock_nib_load, tmp_path
):
    """Valida la interoperabilidad con Dipy y la inyección correcta de metadatos afines."""
    fake_streamlines = [np.array([[1.0, 1.0, 1.0]], dtype=np.float32)]
    output_file = tmp_path / 'processed' / 'output_tracts.trk'

    # Mockear la imagen NIfTI de referencia para extraer el header y el affine
    mock_nii = MagicMock()
    fake_affine = np.eye(4)
    mock_nii.affine = fake_affine
    mock_nii.header = 'FakeHeader'
    mock_nib_load.return_value = mock_nii

    # Instanciar el flujo de guardado
    save_tractogram(
        streamlines=fake_streamlines,
        reference_path='dummy_ref.nii',
        output_path=str(output_file),
    )

    # Verificar que nibabel leyó la imagen anatómica de referencia estructural
    mock_nib_load.assert_called_once_with('dummy_ref.nii')

    # Verificar que el objeto contenedor de Dipy fue instanciado correctamente
    mock_dipy_streamlines.assert_called_once_with(fake_streamlines)

    # Verificar que la función de persistencia de dipy recibió los metadatos necesarios
    mock_dipy_save.assert_called_once_with(
        tractogram=mock_dipy_streamlines.return_value,
        filename=str(output_file),
        reference=mock_nii,
    )


@patch('nibabel.load')
def test_load_tractography_data_ram_optimization(mock_nib_load):
    """Asegura la carga segura de datos forzando el tipo flotante de 32 bits en RAM."""
    mock_img = MagicMock()
    mock_nib_load.return_value = mock_img

    # Ejecutar la función de carga
    _ = load_tractography_data(
        pdd_path='pdd.nii', fa_path='fa.nii', mask_path='mask.nii'
    )

    # Validar que se llamó tres veces (una para cada volumen espacial del DTI)
    assert mock_nib_load.call_count == 3

    # Validar que get_fdata forzó la optimización de precisión simple (float32)
    mock_img.get_fdata.assert_called_with(dtype=np.float32)