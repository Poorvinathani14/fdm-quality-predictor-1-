# Data

The dataset is **not** stored in this repository. Download it and place the file in `data/raw/`.

- **Source:** Mendeley Data, *FDM 3D Printing Dataset*
- **DOI:** [10.17632/zd6td6svd6.2](https://doi.org/10.17632/zd6td6svd6.2)
- **Authors:** Aktepe & Ergün (2026)
- **License:** CC BY 4.0. Please cite the authors if you use it.

About 500 specimens, 10 filaments and 15 infill patterns. The Excel sheet is named `Sayfa1`.

```
data/
└── raw/
    └── <downloaded file>.xlsx   # git-ignored
```

The loader handles the quirks of the raw file (zero-width spaces in column names, mixed case text) and drops rows with roughness <= 0.
