# Diffusion MRI and Tractography
<p align="center">
  <img src="docs/images/tractography.png" width="85%">
</p>

Implementación desde cero de un pipeline de procesamiento de imágenes de resonancia magnética por difusión (DW-MRI), incluyendo:

- Estimación del Tensor de Difusión (DTI).
- Cálculo de métricas microestructurales.
- Obtención de direcciones principales de difusión (PDD).
- Generación de semillas.
- Tractografía determinística basada en streamlines.
- Exportación de resultados en formatos NIfTI y TRK.

---

# Objetivos

Este proyecto tiene como objetivo implementar de forma explícita y modular las principales etapas de un pipeline clásico de reconstrucción de tractos de sustancia blanca a partir de datos DW-MRI.

El repositorio está dividido en dos bloques principales:

1. **Tensor de Difusión (DTI)**
2. **Tractografía Determinística**

---

# Documentación

## Tensor de Difusión

Contiene:

- Modelo de Stejskal-Tanner.
- Construcción de la matriz de diseño.
- Ajuste por mínimos cuadrados.
- Estimación voxel a voxel del tensor.
- Cálculo de FA.
- Cálculo de MD.
- Obtención de la Dirección Principal de Difusión (PDD).

Consultar:

```text
README_DTI.md
```

---

## Tractografía

Contiene:

- Generación de semillas.
- Interpolación trilineal de la PDD.
- Integración de Euler.
- Criterios de parada.
- Seguimiento bidireccional.
- Construcción del tractograma final.

Consultar:

```text
README_TRACTOGRAPHY.md
```

---

# Estructura del proyecto

```text
.
├── data
│   ├── raw
│   └── processed
│       ├── metrics
│       ├── seeds
│       ├── tensors
│       └── tracts
│
├── docs
│   └── images
│
├── scripts
│   ├── run_fit.py
│   ├── run_seeding.py
│   └── run_tracking.py
│
├── src
│   ├── dti
│   │   ├── design_matrix.py
│   │   ├── tensor.py
│   │   ├── least_squares.py
│   │   ├── estimation.py
│   │   └── metrics.py
│   │
│   ├── tractography
│   │   ├── seeds.py
│   │   ├── interpolation.py
│   │   ├── propagation.py
│   │   └── tracking.py
│   │
│   └── io
│       ├── load_data.py
│       └── save_data.py
│
├── tests
│   ├── dti
│   ├── tractography
│   └── scripts
│
├── README.md
├── README_DTI.md
├── README_TRACTOGRAPHY.md
├── requirements.txt
└── .gitignore
```

---

# Pipeline completo

## 1. Estimación del Tensor de Difusión

```bash
python -m scripts.run_fit
```

Genera:

```text
data/processed/tensors/
└── ISMRM_2023_b3000_DT.nii.gz
```

```text
data/processed/metrics/
├── ISMRM_2023_b3000_FA.nii.gz
├── ISMRM_2023_b3000_MD.nii.gz
└── ISMRM_2023_b3000_PDD.nii.gz
```

---

## 2. Generación de semillas

```bash
python -m scripts.run_seeding
```

Genera:

```text
data/processed/seeds/
└── seeds.npy
```

---

## 3. Tractografía

```bash
python -m scripts.run_tracking
```

Genera:

```text
data/processed/tracts/
└── tractogram.trk
```

---

# Dependencias

- numpy
- nibabel
- dipy
- tqdm
- pytest

Instalación:

```bash
pip install -r requirements.txt
```

---

# Dataset

Los experimentos fueron realizados utilizando el:

**ISMRM 2023 Tractography Challenge Dataset**

Archivos requeridos:

```text
data/raw/
├── ISMRM_2023_b3000.nii.gz
├── ISMRM_2023_b3000.bval
├── ISMRM_2023_b3000.bvec
└── ISMRM_2023_b3000_mask.nii
```

---

# Resultados

El pipeline permite obtener:

- Tensor de difusión.
- Mapas de FA.
- Mapas de MD.
- Campo de direcciones principales (PDD).
- Semillas de tractografía.
- Tractogramas completos en formato `.tck`.

Los detalles matemáticos y de implementación se encuentran en:

```text
README_DTI.md
README_TRACTOGRAPHY.md
```

---

# Referencias

- Stejskal, E. O., & Tanner, J. E. (1965).
- Basser, P. J., Mattiello, J., & LeBihan, D. (1994).
- Descoteaux, M. (2023). Diffusion MRI: Theory, Methods and Applications.
