# scripts/run_fit.py

from src.io.load_data import load_dwi_data
from src.io.save_data import save_nifti
from src.dti.estimation import fit_tensor_model
from src.dti.metrics import calculate_dti_metrics

def main():
    print("Pipeline DTI")
    
    # Cargar datos crudos y metadatos
    print("Cargando volúmenes y metadatos...")
    dataset = load_dwi_data(
        dwi_path='data/raw/ISMRM_2023_b3000.nii.gz',
        bval_path='data/raw/ISMRM_2023_b3000.bval',
        bvec_path='data/raw/ISMRM_2023_b3000.bvec',
        mask_path='data/raw/ISMRM_2023_b3000_mask.nii'
    )

    # Estimar el Tensor de Difusión (DTI)
    # Aquí se aplica el método de mínimos cuadrados derivado en tu LaTeX
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
        output_path='data/processed/ISMRM_2023_b3000_DT.nii.gz'
    )
    
    # Guardar el mapa de Anisotropía Fraccional
    save_nifti(
        dt_data=FA, 
        affine=dataset["affine"], 
        header=dataset["header"], 
        output_path='data/processed/ISMRM_2023_b3000_FA.nii.gz'
    )
    
    # Guardar el mapa de Difusividad Media
    save_nifti(
        dt_data=MD, 
        affine=dataset["affine"], 
        header=dataset["header"], 
        output_path='data/processed/ISMRM_2023_b3000_MD.nii.gz'
    )
    
    print("Proceso completado con éxito")

if __name__ == "__main__":
    main()