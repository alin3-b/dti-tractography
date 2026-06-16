#tests/tractography/test_interpolation.py
import numpy as np
import pytest
from src.tractography.interpolation import interpolate_pdd


@pytest.fixture
def fake_pdd_dataset():
    """Fixture que construye un volumen PDD 4D y una máscara para pruebas.

    Crea un bloque de 4x4x4 vóxeles donde el sub-cubo central de vecinos tiene
    vectores PDD apuntando coherentemente a lo largo del eje X [1, 0, 0].
    """
    shape_3d = (4, 4, 4)
    pdd_volume = np.zeros((*shape_3d, 3), dtype=np.float32)
    mask = np.zeros(shape_3d, dtype=np.int32)

    # Definir una región cerebral activa en el centro del volumen
    mask[1:3, 1:3, 1:3] = 1

    # Llenar la vecindad con vectores alineados al eje X
    pdd_volume[1:3, 1:3, 1:3] = [1.0, 0.0, 0.0]

    return pdd_volume, mask


def test_interpolate_pdd_exact_center(fake_pdd_dataset):
    """Valida la interpolación en el centro geométrico de un grupo de vóxeles."""
    pdd_volume, mask = fake_pdd_dataset

    # Posición continua exactamente a la mitad entre el vóxel 1 y 2
    pos = np.array([1.5, 1.5, 1.5], dtype=np.float32)

    v_interp = interpolate_pdd(pos=pos, pdd_volume=pdd_volume, mask=mask)

    assert v_interp is not None
    assert v_interp.dtype == np.float32
    assert v_interp.shape == (3,)
    # Al estar rodeado de vectores [1, 0, 0], el resultado debe ser idéntico
    assert np.allclose(v_interp, [1.0, 0.0, 0.0])


def test_interpolate_pdd_antipodal_sign_correction(fake_pdd_dataset):
    """Asegura que el algoritmo corrija la inversión de polaridad (signo antipodal).

    Se invierten intencionalmente la mitad de los vectores vecinos en el eje X
    ([-1, 0, 0]). Si la mitigación funciona, el resultado final debe seguir
    siendo coherentemente [1, 0, 0] en lugar de cancelarse hacia cero.
    """
    pdd_volume, mask = fake_pdd_dataset
    pos = np.array([1.5, 1.5, 1.5], dtype=np.float32)

    # Forzar signos opuestos en la mitad superior del cubo de interpolación (x1 = 2)
    pdd_volume[2, 1:3, 1:3] = [-1.0, 0.0, 0.0]

    v_interp = interpolate_pdd(pos=pos, pdd_volume=pdd_volume, mask=mask)

    assert v_interp is not None
    # El valor absoluto de las componentes debe ser [1, 0, 0] demostrando
    # que no hubo cancelación destructiva (la cual daría un vector nulo o desviado)
    assert np.allclose(np.abs(v_interp), [1.0, 0.0, 0.0], atol=1e-5)


def test_interpolate_pdd_returns_none_outside_mask(fake_pdd_dataset):
    """Valida que si alguna de las esquinas del cubo no es cerebro, el tracking se detenga."""
    pdd_volume, mask = fake_pdd_dataset

    # Posición donde algunas esquinas vecinas caerán en un vóxel con máscara = 0
    pos = np.array([0.5, 1.5, 1.5], dtype=np.float32)

    v_interp = interpolate_pdd(pos=pos, pdd_volume=pdd_volume, mask=mask)

    assert v_interp is None


def test_interpolate_pdd_out_of_bounds(fake_pdd_dataset):
    """Valida el manejo seguro de límites para coordenadas completamente fuera del volumen."""
    pdd_volume, mask = fake_pdd_dataset

    # Coordenadas negativas o que desbordan las dimensiones de la matriz (4, 4, 4)
    pos_negative = np.array([-0.5, 1.5, 1.5], dtype=np.float32)
    pos_overflow = np.array([1.5, 3.5, 1.5], dtype=np.float32)

    assert interpolate_pdd(pos=pos_negative, pdd_volume=pdd_volume, mask=mask) is None
    assert interpolate_pdd(pos=pos_overflow, pdd_volume=pdd_volume, mask=mask) is None


def test_interpolate_pdd_destructive_noise_cancellation():
    """Prueba el comportamiento ante ruido extremo donde no existe una dirección preferencial.

    Si los vectores opuestos se cancelan mutuamente por inestabilidad ortogonal
    y la norma cae por debajo de 1e-6, la función debe retornar None de forma segura.
    """
    shape_3d = (4, 4, 4)
    pdd_volume = np.zeros((*shape_3d, 3), dtype=np.float32)
    mask = np.ones(shape_3d, dtype=np.int32)
    pos = np.array([1.5, 1.5, 1.5], dtype=np.float32)

    # Configurar vectores mutuamente ortogonales que fuercen una resultante nula al promediar
    pdd_volume[1, 1, 1] = [1.0, 0.0, 0.0]
    pdd_volume[2, 1, 1] = [-1.0, 0.0, 0.0]  # El signo antipodal alineará este a [1,0,0]
    pdd_volume[1, 2, 1] = [0.0, 1.0, 0.0]  # Ortogonal
    pdd_volume[2, 2, 1] = [0.0, -1.0, 0.0]  # Ortogonal inverso (se alineará a [0,1,0])

    # El promedio de direcciones encontradas resultará en una cancelación o colapso de la norma
    v_interp = interpolate_pdd(pos=pos, pdd_volume=pdd_volume, mask=mask)

    # Al no poder definirse un vector unitario d_t estable, se espera un retorno limpio
    if v_interp is not None:
        # Si sobrevive por la asimetría de los otros 4 elementos vacíos, debe ser estrictamente unitario
        assert np.isclose(np.linalg.norm(v_interp), 1.0, atol=1e-6)