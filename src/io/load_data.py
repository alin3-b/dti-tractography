# src/io/load_data.py
import os
import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

def load_dwi_data(dwi_path, bval_path, bvec_path, mask_path=None):
    """
    Carga todos los archivos necesarios para la estimación del tensor de difusión.
    
    Parameters:
        dwi_path (str): Ruta al archivo NIfTI de difusión (.nii o .nii.gz).
        bval_path (str): Ruta al archivo de b-values (.bval).
        bvec_path (str): Ruta al archivo de b-vectors (.bvec).
        mask_path (str, opcional): Ruta al archivo de la máscara NIfTI.
        
    Returns:
        dict: Diccionario con los datos cargados y metadatos.
    """
    # Verificar que los archivos obligatorios existan
    for path in [dwi_path, bval_path, bvec_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
            
    # DWI
    dwi_img = nib.load(dwi_path)
    dwi_data = dwi_img.get_fdata()
    
    # GRADIANTES: cargar b-values y b-vectors usando DIPY
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals, bvecs)
    
    # Máscara: cargar máscara si se proporciona, de lo contrario crear una llena de unos
    if mask_path and os.path.exists(mask_path):
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
    else:
        print("Aviso: No se proporcionó máscara válida. Se procesará todo el volumen.")
        mask_data = np.ones(dwi_data.shape[:3], dtype=bool)
        
    return {
        "image": dwi_img,
        "data": dwi_data,
        "header": dwi_img.header.copy(),
        "affine": dwi_img.affine,
        "bvals": bvals,
        "bvecs": bvecs,
        "gtab": gtab,
        "mask": mask_data,
        "n_volumes": dwi_data.shape[3]
    }