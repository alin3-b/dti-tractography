# src/io/load_data.py
from pathlib import Path
import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs


def load_dwi_data(
    dwi_path: str,
    bval_path: str,
    bvec_path: str,
    mask_path: str | None = None,
) -> dict:
    """Carga todos los archivos necesarios para el ajuste DTI.

    Parameters:
        dwi_path : str
            Archivo NIfTI DW-MRI.
        bval_path : str
            Archivo .bval.
        bvec_path : str
            Archivo .bvec.
        mask_path : str | None
            Máscara cerebral opcional.

    Returns:
        dict:
            Diccionario con datos y metadatos.
    """
    required_files = [dwi_path, bval_path, bvec_path]

    for file_path in required_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f'No se encontró el archivo: {file_path}')

    # DWI
    dwi_img = nib.load(str(dwi_path))
    dwi_data = dwi_img.get_fdata(dtype=np.float32)

    # Gradientes
    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    # Corrección de bvecs como keyword arg para eliminar el UserWarning de DIPY
    gtab = gradient_table(bvals, bvecs=bvecs)

    # Máscara
    if mask_path is not None and Path(mask_path).exists():
        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata(dtype=np.float32)
        mask_data = mask_data > 0
    else:
        print(
            'Aviso: no se encontró máscara. Se procesará todo el volumen.'
        )
        mask_data = np.ones(dwi_data.shape[:3], dtype=bool)

    # Validaciones dimensionales de control
    if dwi_data.ndim != 4:
        raise ValueError('El volumen DWI debe ser 4D.')

    if len(bvals) != dwi_data.shape[3]:
        raise ValueError(
            'Número de b-values incompatible con el número de volúmenes.'
        )

    if bvecs.shape[0] != dwi_data.shape[3]:
        raise ValueError(
            'Número de b-vectors incompatible con el número de volúmenes.'
        )

    return {
        'image': dwi_img,
        'data': dwi_data,
        'header': dwi_img.header.copy(),
        'affine': dwi_img.affine,
        'bvals': bvals,
        'bvecs': bvecs,
        'gtab': gtab,
        'mask': mask_data,
        'n_volumes': dwi_data.shape[3],
    }


def load_nifti(
    nifti_path: str,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Carga un volumen NIfTI genérico.

    Parameters:
        nifti_path : str

    Returns:
        tuple
            (data, affine, header)
    """
    if not Path(nifti_path).exists():
        raise FileNotFoundError(f'No se encontró el archivo: {nifti_path}')

    img = nib.load(str(nifti_path))

    return (
        img.get_fdata(dtype=np.float32),
        img.affine,
        img.header.copy(),
    )


def load_seeds(
    seeds_path: str,
) -> np.ndarray:
    """Carga un archivo .npy de semillas."""
    if not Path(seeds_path).exists():
        raise FileNotFoundError(f'No se encontró el archivo: {seeds_path}')

    return np.load(seeds_path)