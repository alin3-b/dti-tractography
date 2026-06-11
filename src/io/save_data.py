# src/io/save_data.py
import os
import nibabel as nib

def save_nifti(dt_data, affine, header, output_path):
    """
    Guarda el tensor de difusión estimado (DT) en formato NIfTI.
    
    Parameters:
        dt_data (numpy.ndarray): Matriz de datos del tensor, forma (X, Y, Z, 6).
        affine (numpy.ndarray): Matriz de transformación afín (geometría espacial).
        header (nibabel.spatialimages.SpatialHeader): Header original del volumen.
        output_path (str): Ruta completa donde se guardará el archivo.
    """
    # Crear la carpeta destino si no existe (data/processed/)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Reconstruir la imagen NIfTI
    dt_img = nib.Nifti1Image(dt_data, affine, header=header)
    
    # Guardar en disco
    nib.save(dt_img, output_path)
    print(f"Tensor de difusión guardado exitosamente en: {output_path}")
