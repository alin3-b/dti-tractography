#tests/tractography/test_propagation.py
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from src.tractography.propagation import propagate_streamline, track_full_streamline


@pytest.fixture
def fake_volumes():
    """Fixture que genera volúmenes ficticios mínimos para simular el cerebro."""
    shape = (5, 5, 5)
    mask = np.ones(shape, dtype=np.int32)
    fa_volume = np.ones(shape, dtype=np.float32) * 0.4  # Todo materia blanca alta
    pdd_volume = np.zeros((*shape, 3), dtype=np.float32)
    pdd_volume[:, :, :] = [1.0, 0.0, 0.0]  # Tracto uniforme hacia el eje X

    return pdd_volume, mask, fa_volume


def test_propagate_streamline_straight_line(fake_volumes):
    """Prueba una propagación ideal en línea recta hacia adelante (forward)."""
    pdd_volume, mask, fa_volume = fake_volumes
    seed = np.array([1.0, 2.0, 2.0], dtype=np.float32)

    with patch(
        'src.tractography.propagation.interpolate_pdd'
    ) as mock_interpolate:
        # Forzar a que la interpolación siempre retorne el vector unitario del eje X
        mock_interpolate.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        streamline = propagate_streamline(
            seed=seed,
            pdd_volume=pdd_volume,
            mask=mask,
            fa_volume=fa_volume,
            step_size=0.5,
            max_steps=2,
            angle_threshold=45.0,
            fa_threshold=0.15,
            direction=1,
        )

        # Semilla + 2 pasos = 3 puntos totales
        assert streamline.shape == (3, 3)
        assert streamline.dtype == np.float32

        # Verificar integración de Euler exacta: x_(t+1) = x_t + alpha * d_t
        assert np.allclose(streamline[0], [1.0, 2.0, 2.0])  # Semilla
        assert np.allclose(streamline[1], [1.5, 2.0, 2.0])  # Paso 1
        assert np.allclose(streamline[2], [2.0, 2.0, 2.0])  # Paso 2


def test_propagate_streamline_stops_at_low_fa(fake_volumes):
    """Valida que el seguimiento se detega inmediatamente si decae el umbral de FA."""
    pdd_volume, mask, fa_volume = fake_volumes
    seed = np.array([1.0, 2.0, 2.0], dtype=np.float32)

    # Forzar que el vóxel vecino en [2, 2, 2] tenga FA baja (materia gris/LCR)
    fa_volume[2, 2, 2] = 0.05

    with patch(
        'src.tractography.propagation.interpolate_pdd'
    ) as mock_interpolate:
        mock_interpolate.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        streamline = propagate_streamline(
            seed=seed,
            pdd_volume=pdd_volume,
            mask=mask,
            fa_volume=fa_volume,
            step_size=1.0,  # Avanza directo al vóxel [2, 2, 2]
            max_steps=5,
            fa_threshold=0.15,
        )

        # Debe contener la semilla y el primer paso que cayó en zona prohibida, deteniendo el bucle
        assert streamline.shape == (2, 3)


def test_propagate_streamline_antipodal_and_curvature_handling(fake_volumes):
    """Prueba la mitigación del signo antipodal y el quiebre por curvatura excesiva.

    Paso 1: Recibe [1, 0, 0] en el inicio (arranca en +X).
    Paso 2: Recibe [-1, 0, 0], lo invierte internamente a [1, 0, 0] al detectar el signo
            antipodal con respecto al paso previo (pasa la restricción angular).
    Paso 3: Recibe [0, 1, 0] (giro ortogonal brusco de 90°), viola los 45° del umbral y frena.
    """
    pdd_volume, mask, fa_volume = fake_volumes
    seed = np.array([1.0, 2.0, 2.0], dtype=np.float32)

    with patch(
        'src.tractography.propagation.interpolate_pdd'
    ) as mock_interpolate:
        # Secuencia controlada de vectores interpolados devueltos paso a paso
        mock_interpolate.side_effect = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),   # Paso 1: Dirección inicial
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # Paso 2: Antipodal (se corregirá a +X)
            np.array([0.0, 1.0, 0.0], dtype=np.float32),   # Paso 3: Giro de 90° (causará el quiebre)
        ]

        streamline = propagate_streamline(
            seed=seed,
            pdd_volume=pdd_volume,
            mask=mask,
            fa_volume=fa_volume,
            step_size=0.5,
            max_steps=5,
            angle_threshold=45.0,
        )

        # El resultado esperado son 3 puntos: Semilla, Paso 1 y Paso 2 (el paso 3 aborta antes de agregarse)
        assert streamline.shape == (3, 3)
        assert np.allclose(streamline[0], [1.0, 2.0, 2.0])  # Semilla
        assert np.allclose(streamline[1], [1.5, 2.0, 2.0])  # Paso 1
        assert np.allclose(streamline[2], [2.0, 2.0, 2.0])  # Paso 2 (avanzó en +X por la corrección)


def test_track_full_streamline_unification(fake_volumes):
    """Valida la correcta unión bidireccional (forward + backward).

    Asegura la inversión de los puntos del backward y la eliminación de la
    semilla duplicada en la costura central.
    """
    pdd_volume, mask, fa_volume = fake_volumes
    seed = np.array([2.0, 2.0, 2.0], dtype=np.float32)

    with patch(
        'src.tractography.propagation.propagate_streamline'
    ) as mock_propagate:
        # Simular lo que retornarían las llamadas internas independientes
        # Forward avanza en +X: [[Semilla], [2.5, 2, 2], [3.0, 2, 2]]
        fake_forward = np.array(
            [[2.0, 2.0, 2.0], [2.5, 2.0, 2.0], [3.0, 2.0, 2.0]], dtype=np.float32
        )
        # Backward avanza en -X: [[Semilla], [1.5, 2, 2], [1.0, 2, 2]]
        fake_backward = np.array(
            [[2.0, 2.0, 2.0], [1.5, 2.0, 2.0], [1.0, 2.0, 2.0]], dtype=np.float32
        )

        mock_propagate.side_effect = [fake_forward, fake_backward]

        full_streamline = track_full_streamline(
            seed=seed,
            pdd_volume=pdd_volume,
            mask=mask,
            fa_volume=fa_volume,
            step_size=0.5,
            max_steps=2,
        )

        # Dimensiones esperadas: 2 (backward invertido sin semilla) + 3 (forward completo) = 5 puntos
        assert full_streamline.shape == (5, 3)

        # Corroborar el ordenamiento anatómico continuo de inicio a fin en el eje X
        expected_trajectory = np.array(
            [
                [1.0, 2.0, 2.0],  # Fin de la punta backward
                [1.5, 2.0, 2.0],
                [2.0, 2.0, 2.0],  # Semilla única (costura central)
                [2.5, 2.0, 2.0],
                [3.0, 2.0, 2.0],  # Fin de la punta forward
            ],
            dtype=np.float32,
        )
        assert np.allclose(full_streamline, expected_trajectory)