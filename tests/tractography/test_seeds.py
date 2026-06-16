# tests/tractography/test_seeds.py
import os
from pathlib import Path
import numpy as np
import pytest
from src.io.save_data import save_seeds
from src.tractography.seeds import generate_seeds


@pytest.fixture
def fake_dti_data():
    """Fixture que provee una máscara y un volumen de FA básicos para pruebas."""
    shape = (4, 4, 4)
    mask = np.zeros(shape, dtype=np.int32)
    fa_volume = np.zeros(shape, dtype=np.float32)

    # Activar un par de vóxeles estratégicos
    mask[1, 1, 1] = 1
    fa_volume[1, 1, 1] = 0.4  # Por encima del umbral por defecto (0.2)

    mask[2, 2, 2] = 1
    fa_volume[2, 2, 2] = 0.1  # Por debajo del umbral por defecto

    return mask, fa_volume


def test_generate_seeds_grid_method(fake_dti_data):
    """Valida la generación de semillas mediante el método de rejilla fija (grid)."""
    mask, fa_volume = fake_dti_data
    density = 2  # 2^3 = 8 semillas por vóxel válido

    seeds = generate_seeds(
        mask=mask,
        fa_volume=fa_volume,
        fa_thresh=0.2,
        method='grid',
        density=density,
    )

    # Solo el vóxel [1, 1, 1] es válido -> 1 * 8 = 8 semillas en total
    assert seeds.shape == (8, 3)
    assert seeds.dtype == np.float32

    # Verificar que todas las semillas caigan dentro de los límites del vóxel [1, 1, 1]
    # Espacio continuo: [1.0, 2.0)
    assert np.all(seeds >= 1.0)
    assert np.all(seeds < 2.0)

    # Verificar la simetría matemática exacta de los offsets del grid ([1.25, 1.75])
    unique_coordinates = np.unique(seeds)
    expected_offsets = np.array([1.25, 1.75], dtype=np.float32)
    assert np.allclose(unique_coordinates, expected_offsets)


def test_generate_seeds_random_method(fake_dti_data):
    """Valida la generación de semillas estocásticas mediante el método random y su reproducibilidad."""
    mask, fa_volume = fake_dti_data
    n_seeds = 15

    seeds_1 = generate_seeds(
        mask=mask,
        fa_volume=fa_volume,
        fa_thresh=0.2,
        method='random',
        n_seeds_random=n_seeds,
        random_seed=42,
    )

    seeds_2 = generate_seeds(
        mask=mask,
        fa_volume=fa_volume,
        fa_thresh=0.2,
        method='random',
        n_seeds_random=n_seeds,
        random_seed=42,
    )

    # Validar dimensiones
    assert seeds_1.shape == (n_seeds, 3)

    # Validar que caigan en el único vóxel con FA suficiente [1, 1, 1]
    assert np.all(seeds_1 >= 1.0)
    assert np.all(seeds_1 < 2.0)

    # Validar reproducibilidad por la semilla pseudoaleatoria
    assert np.array_equal(seeds_1, seeds_2)


def test_generate_seeds_raises_value_error_if_no_voxels_found(fake_dti_data):
    """Asegura que se lance un ValueError si ningún vóxel cumple con el umbral microestructural."""
    mask, fa_volume = fake_dti_data

    # Forzar un umbral de FA inalcanzable
    with pytest.raises(ValueError, match='No se encontraron vóxeles'):
        generate_seeds(mask=mask, fa_volume=fa_volume, fa_thresh=0.9)


def test_generate_seeds_raises_value_error_for_unknown_method(fake_dti_data):
    """Asegura que se lance un ValueError si el método de siembra no existe."""
    mask, fa_volume = fake_dti_data

    with pytest.raises(ValueError, match="Método de siembra 'invalid' no reconocido"):
        generate_seeds(mask=mask, fa_volume=fa_volume, method='invalid')


def test_save_seeds(tmp_path):
    """Prueba que las coordenadas calculadas se almacenen correctamente en disco en formato .npy."""
    fake_seeds = np.array([[1.25, 1.25, 1.25], [1.75, 1.75, 1.75]], dtype=np.float32)
    output_file = tmp_path / 'subfolder' / 'test_seeds.npy'

    # Guardar usando la función bajo prueba importada de src.io.save_data
    save_seeds(seeds=fake_seeds, output_path=str(output_file))

    # Verificaciones en el sistema de archivos
    assert output_file.exists()

    # Cargar y corroborar integridad de la información
    loaded_seeds = np.load(output_file)
    assert np.array_equal(loaded_seeds, fake_seeds)
    assert loaded_seeds.dtype == np.float32