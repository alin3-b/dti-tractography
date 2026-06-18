# src/io/save_data.py
from pathlib import Path
import nibabel as nib
import numpy as np

def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    output_path: str,
) -> None:
    """Guarda un volumen NIfTI.

    Parameters:
        data : np.ndarray
            Datos a guardar.
        affine : np.ndarray
            Matriz afín.
        header : nib.Nifti1Header
            Header de referencia.
        output_path : str
            Ruta de salida.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = nib.Nifti1Image(data, affine, header=header)
    nib.save(image, str(output_path))

    print(f'Archivo NIfTI guardado: {output_path}')

def save_seeds(
    seeds: np.ndarray,
    output_path: str,
) -> None:
    """Guarda semillas en formato .npy.

    Parameters:
        seeds : np.ndarray
            Coordenadas de semillas (N,3).
        output_path : str
            Ruta de salida.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(output_path), seeds)
    print(f'Semillas guardadas: {output_path}')

def save_tractogram(
    streamlines: list[np.ndarray],
    reference_path: str,
    output_path: str,
) -> None:
    """Guarda streamlines en formato estándar .tck (MRtrix3).

    Transforma automáticamente los puntos desde el espacio continuo de vóxeles
    al espacio físico milimétrico (RAS+) requerido por la especificación tck.

    Parameters:
        streamlines : list[np.ndarray]
            Lista de streamlines en espacio vóxel.
        reference_path : str
            Ruta del volumen NIfTI de referencia para transformaciones espaciales.
        output_path : str
            Archivo .tck de salida.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_img = nib.load(str(reference_path))

    # Importaciones específicas de la API moderna de DIPY
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram as dipy_save

    # El formato .tck REQUIERE estrictamente que el espacio de los datos esté en milímetros
    # Al pasar Space.VOX, StatefulTractogram utiliza la matriz afín (affine) del volumen de
    # referencia para proyectar los puntos de forma precisa al espacio RAS+ real en el archivo final.
    sft = StatefulTractogram(streamlines, reference_img, Space.VOX)

    # El formato .tck no implementa ni soporta chequeos de Bounding Box (bbox_valid_check)
    # ya que no incrusta la matriz afín del vóxel dentro de su propia cabecera binaria.
    dipy_save(sft, str(output_path))

    print(f'Tractograma guardado exitosamente: {output_path}')