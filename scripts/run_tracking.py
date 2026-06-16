# scripts/run_tracking.py
from pathlib import Path
import numpy as np

from src.io.load_data import load_nifti, load_seeds
from src.io.save_data import save_tractogram
from src.tractography.tracking import track_multiple_seeds


def main() -> None:
    print("Tractografía Determinística desde Tensor de Difusión")
    
    data_dir = Path("data")
    
    #  RUTAS 
    pdd_path = data_dir / "processed" / "metrics" / "ISMRM_2023_b3000_PDD.nii.gz"
    fa_path = data_dir / "processed" / "metrics" / "ISMRM_2023_b3000_FA.nii.gz"
    mask_path = data_dir / "raw" / "ISMRM_2023_b3000_mask.nii"
    seeds_path = data_dir / "processed" / "seeds" / "seeds.npy"
    
    tracts_dir = data_dir / "processed" / "tracts"
    tractogram_path = tracts_dir / "tractogram.tck"
    
    #  CONFIGURACIÓN 
    config = {
        'step_size': 0.5,
        'max_steps': 1200,
        'angle_threshold': 45.0,
        'fa_threshold': 0.15,
        'min_length': 15.0,        # longitud mínima aproximada (voxels)
        'verbose': True
    }
    
    # Validación de archivos
    required_files = [pdd_path, fa_path, mask_path, seeds_path]
    for f in required_files:
        if not f.exists():
            print(f"Error: Archivo no encontrado → {f}")
            print("\nAsegúrate de ejecutar en orden:")
            print("   1. python -m scripts.run_fit")
            print("   2. python -m scripts.run_seeding")
            return
    
    print("Cargando volúmenes necesarios (PDD, FA, Mask)...")
    pdd_volume, _, _ = load_nifti(str(pdd_path))
    fa_volume, _, _ = load_nifti(str(fa_path))
    mask_volume, _, _ = load_nifti(str(mask_path))
    
    print("Cargando semillas...")
    seeds = load_seeds(str(seeds_path))
    
    print(f"\nIniciando tractografía con {len(seeds):,} semillas...")
    streamlines = track_multiple_seeds(
        seeds=seeds,
        pdd_volume=pdd_volume,
        mask=mask_volume,
        fa_volume=fa_volume,
        **config
    )
    
    # Guardar tractograma
    print("\nGuardando tractograma en formato .trk...")
    save_tractogram(
        streamlines=streamlines,
        reference_path=str(mask_path),      # referencia espacial
        output_path=str(tractogram_path)
    )
    
    print("\nTractografía completada exitosamente!")
    print(f"   Streamlines válidas: {len(streamlines)}")
    print(f"   Archivo guardado: {tractogram_path}")


if __name__ == "__main__":
    main()