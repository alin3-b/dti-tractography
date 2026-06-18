# scripts/run_seeding.py
from pathlib import Path
import nibabel as nib
import numpy as np
from src.tractography.seeds import generate_seeds
from src.io.save_data import save_seeds

def main() -> None:
    print('Generación de Semillas para Tractografía')
    
    data_dir = Path('data')

    mask_path = data_dir / 'raw' / 'ISMRM_2023_b3000_mask.nii'
    fa_path = data_dir / 'processed' / 'metrics' / 'ISMRM_2023_b3000_FA.nii.gz'

    seeds_dir = data_dir / 'processed' / 'seeds'
    seeds_dir.mkdir(parents=True, exist_ok=True)
    seeds_output_path = seeds_dir / 'seeds.npy'

    # Parámetros de generación organizados de forma limpia
    config = {
        'fa_thresh': 0.20,  # Umbral de FA (materia blanca)
        'method': 'grid',  # 'grid' o 'random'
        'density': 2,  # Para grid: semillas por eje (2x2x2 = 8 por vóxel)
        'n_seeds_random': 8000,  # Solo usado si method='random'
        'random_seed': 42,
    }

    # Validar que los archivos existan antes de procesar
    if not mask_path.exists() or not fa_path.exists():
        print('Error: No se encontraron los mapas de Máscara o FA.')
        print('Asegúrate de ejecutar primero scripts/run_fit.py.')
        return

    print('Cargando FA y máscara desde:')
    print(f'   FA   → {fa_path}')
    print(f'   Mask → {mask_path}')

    # Cargar datos optimizando el uso de memoria RAM
    fa_nii = nib.load(str(fa_path))
    mask_nii = nib.load(str(mask_path))

    fa_volume = fa_nii.get_fdata(dtype=np.float32)
    mask = mask_nii.get_fdata(dtype=np.float32)

    # Generar semillas usando desempaquetado de diccionario
    seeds = generate_seeds(mask=mask, fa_volume=fa_volume, **config)

    # Guardar matriz binaria resultante
    save_seeds(seeds, str(seeds_output_path))

    print(f'\nSemillas generadas exitosamente: {len(seeds):,} semillas')
    print(f"   Método: {config['method']}")
    print(f'   Guardado en: {seeds_output_path}')

if __name__ == '__main__':
    main()