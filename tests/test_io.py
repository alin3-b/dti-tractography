import os
import pytest
import numpy as np
import nibabel as nib
from unittest.mock import MagicMock, patch

# Importamos tus funciones modulares
from src.io.load_data import load_dwi_data
from src.io.save_data import save_nifti

# FIXTURES: Datos sintéticos que usaremos repetidamente en los tests
@pytest.fixture
def mock_dwi_setup():
    """Genera datos sintéticos pequeños simulando un volumen DWI."""
    shape = (10, 10, 10, 65)  # 65 volúmenes (p.ej., 1 b0 + 64 gradientes)
    data = np.random.rand(*shape)
    affine = np.eye(4)
    header = nib.Nifti1Header()
    
    # Creamos un objeto Mock que simule el comportamiento de nibabel.load()
    mock_img = MagicMock()
    mock_img.get_fdata.return_value = data
    mock_img.affine = affine
    mock_img.header = header
    return mock_img, data, affine, header



# TESTS PARA: src/io/load_data.py

def test_load_dwi_data_file_not_found():
    """Prueba que se lance FileNotFoundError si algún archivo obligatorio no existe."""
    with pytest.raises(FileNotFoundError):
        load_dwi_data("inexistente.nii.gz", "bval_fake", "bvec_fake")


@patch('src.io.load_data.os.path.exists')
@patch('src.io.load_data.nib.load')
@patch('src.io.load_data.read_bvals_bvecs')
@patch('src.io.load_data.gradient_table')
def test_load_dwi_data_success_without_mask(mock_gtab, mock_read_bvals, mock_nib_load, mock_exists, mock_dwi_setup):
    """Prueba la carga exitosa de DWI sin máscara (debe generar una máscara de unos)."""
    # Configurar los mocks
    mock_img, dwi_data, affine, header = mock_dwi_setup
    mock_exists.return_value = True  # Simula que todos los archivos existen
    mock_nib_load.return_value = mock_img
    
    fake_bvals = np.array([0, 3000, 3000])
    fake_bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mock_read_bvals.return_value = (fake_bvals, fake_bvecs)
    mock_gtab.return_value = "MockedGradientTable"
    
    # Ejecutar la función (pasando mask_path=None explícitamente)
    result = load_dwi_data("dwi.nii.gz", "dwi.bval", "dwi.bvec", mask_path=None)
    
    # Verificaciones (Asserts)
    assert result["data"].shape == (10, 10, 10, 65)
    assert result["n_volumes"] == 65
    assert np.array_equal(result["affine"], affine)
    assert result["gtab"] == "MockedGradientTable"
    # Al no haber máscara, debe ser una matriz de True con la misma forma espacial (X, Y, Z)
    assert result["mask"].shape == (10, 10, 10)
    assert np.all(result["mask"] == True)


@patch('src.io.load_data.os.path.exists')
@patch('src.io.load_data.nib.load')
@patch('src.io.load_data.read_bvals_bvecs')
@patch('src.io.load_data.gradient_table')
def test_load_dwi_data_success_with_mask(mock_gtab, mock_read_bvals, mock_nib_load, mock_exists, mock_dwi_setup):
    """Prueba la carga exitosa incluyendo una máscara real binaria."""
    mock_img, dwi_data, affine, header = mock_dwi_setup
    mock_exists.return_value = True
    
    # Crear un mock secundario para la máscara
    mock_mask_img = MagicMock()
    mask_array = np.zeros((10, 10, 10))
    mask_array[3:7, 3:7, 3:7] = 1  # Un cubo central de unos
    mock_mask_img.get_fdata.return_value = mask_array
    
    # nib.load será llamado dos veces: primero para DWI, luego para la máscara
    mock_nib_load.side_effect = [mock_img, mock_mask_img]
    
    mock_read_bvals.return_value = (np.array([0, 3000]), np.array([[0,0,0], [1,0,0]]))
    mock_gtab.return_value = "MockedGradientTable"
    
    result = load_dwi_data("dwi.nii.gz", "dwi.bval", "dwi.bvec", mask_path="mask.nii")
    
    assert result["mask"].shape == (10, 10, 10)
    assert result["mask"][5, 5, 5] == 1  # Centro activo
    assert result["mask"][0, 0, 0] == 0  # Fondo inactivo



# TESTS PARA: src/io/save_data.py

@patch('src.io.save_data.os.path.exists')
@patch('src.io.save_data.os.makedirs')
@patch('src.io.save_data.nib.Nifti1Image')
@patch('src.io.save_data.nib.save')
def test_save_nifti_creates_directory_and_saves(mock_save, mock_nifti_image, mock_makedirs, mock_exists):
    """Prueba que el guardado cree las carpetas destino si no existen y salve el NIfTI."""
    # Configuración de mocks
    mock_exists.return_value = False  # Simula que la carpeta destino NO existe
    fake_dt_data = np.zeros((10, 10, 10, 6))
    fake_affine = np.eye(4)
    fake_header = nib.Nifti1Header()
    
    mock_img_instance = MagicMock()
    mock_nifti_image.return_value = mock_img_instance
    
    # Ejecutar guardado
    output_path = "data/processed/test_output_DT.nii.gz"
    save_nifti(fake_dt_data, fake_affine, fake_header, output_path)
    
    # Verificar que detectó la falta de directorio y lo creó
    mock_makedirs.assert_called_once_with("data/processed")
    
    # Verificar que se construyó el objeto Nifti con los parámetros correctos
    mock_nifti_image.assert_called_once_with(fake_dt_data, fake_affine, header=fake_header)
    
    # Verificar que finalmente se llamó al escritor de nibabel
    mock_save.assert_called_once_with(mock_img_instance, output_path)