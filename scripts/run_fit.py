# scripts/run_fit.py

from src.io.load_data import load_dwi_data
from src.io.save_data import save_nifti
from src.dti.estimation import fit_tensor_model
from src.dti.metrics import calculate_dti_metrics

def main():

    # CONFIGURACIÓN DE USUARIO (Cambia aquí los nombres de tus archivos)
    # Archivos de entrada (Input)
    DWI_FILE   = 'ISMRM_2023_b3000.nii.gz'
    BVAL_FILE  = 'ISMRM_2023_b3000.bval'
    BVEC_FILE  = 'ISMRM_2023_b3000.bvec'
    MASK_FILE  = 'ISMRM_2023_b3000_mask.nii'
    
    # Archivos de salida (Output)
    OUT_DT_FILE = 'ISMRM_2023_b3000_DT.nii.gz'
    OUT_FA_FILE = 'ISMRM_2023_b3000_FA.nii.gz'
    OUT_MD_FILE = 'ISMRM_2023_b3000_MD.nii.gz'

    # Construcción automática de rutas fijas de carpetas
    INPUT_DIR = 'data/raw/'
    OUTPUT_DIR = 'data/processed/'

    print("Pipeline DTI")
    
    # Cargar datos crudos y metadatos
    print("Cargando volúmenes y metadatos...")
    dataset = load_dwi_data(
        dwi_path=INPUT_DIR + DWI_FILE,
        bval_path=INPUT_DIR + BVAL_FILE,
        bvec_path=INPUT_DIR + BVEC_FILE,
        mask_path=INPUT_DIR + MASK_FILE
    )

    # Estimar el Tensor de Difusión (DTI)
    DT = fit_tensor_model(
        dwi_data=dataset["data"],
        mask=dataset["mask"],
        bvals=dataset["bvals"],
        bvecs=dataset["bvecs"],
        s0_index=0
    )

    # Calcular las métricas escalares (Anisotropía Fraccional y Difusividad Media)
    FA, MD = calculate_dti_metrics(DT, dataset["mask"])

    # Guardar los resultados en formato NIfTI
    print("Guardando archivos procesados...")
    
    # Guardar el tensor completo (6 componentes)
    save_nifti(
        dt_data=DT, 
        affine=dataset["affine"], 
        header=dataset["header"], 
        output_path=OUTPUT_DIR + OUT_DT_FILE
    )
    
    # Guardar el mapa de Anisotropía Fraccional
    save_nifti(
        dt_data=FA, 
        affine=dataset["affine"], 
        header=dataset["header"], 
        output_path=OUTPUT_DIR + OUT_FA_FILE
    )
    
    # Guardar el mapa de Difusividad Media
    save_nifti(
        dt_data=MD, 
        affine=dataset["affine"], 
        header=dataset["header"], 
        output_path=OUTPUT_DIR + OUT_MD_FILE
    )
    
    print("Proceso completado con éxito")

if __name__ == "__main__":
    main()