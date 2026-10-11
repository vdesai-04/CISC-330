# Question 6 - Marker Localization Results

## Localization at 0.1 mm per pixel

Errors are distances on the detector in mm. One pixel is 0.1 mm.

| Image | Marker | True (u, v) | Segmented (u, v) | Segmented error | Error without background removal | Visual (u, v) | Visual error |
|---|---|---|---|---|---|---|---|
| sphere A | M1 | (22.76, 13.17) | (22.757, 13.165) | 0.0008 | 0.0003 | - | - |
| sphere A | M2 | (3.92, 78.99) | (3.918, 78.990) | 0.0019 | 0.0052 | - | - |
| sphere A | M3 | (-12.40, 73.09) | (-12.402, 73.086) | 0.0024 | 0.0061 | - | - |
| sphere B | M1 | (22.14, -66.89) | (22.140, -66.894) | 0.0009 | 0.0045 | - | - |
| sphere B | M2 | (4.17, 42.73) | (4.170, 42.732) | 0.0032 | 0.0020 | - | - |
| sphere B | M3 | (-12.44, -67.44) | (-12.438, -67.436) | 0.0027 | 0.0035 | - | - |
| vertebra A | M1 | (22.76, 13.17) | (22.757, 13.167) | 0.0013 | 0.0106 | - | - |
| vertebra A | M2 | (3.92, 78.99) | (3.919, 78.990) | 0.0005 | 0.0057 | - | - |
| vertebra A | M3 | (-12.40, 73.09) | (-12.403, 73.087) | 0.0012 | 0.0064 | - | - |
| vertebra B | M1 | (22.14, -66.89) | (22.139, -66.894) | 0.0021 | 0.0112 | - | - |
| vertebra B | M2 | (4.17, 42.73) | (4.170, 42.733) | 0.0025 | 0.0077 | - | - |
| vertebra B | M3 | (-12.44, -67.44) | (-12.438, -67.436) | 0.0029 | 0.0056 | - | - |

Visual columns are empty until `python visual_localization.py` has been run by hand.

## Effect of pixel size

| Pixel size (mm) | Phantom | Mean error (mm) | Max error (mm) | Max error (pixels) |
|---|---|---|---|---|
| 1.0 | sphere | 0.0541 | 0.0879 | 0.088 |
| 1.0 | vertebra | 0.0640 | 0.0895 | 0.090 |
| 0.5 | sphere | 0.0168 | 0.0304 | 0.061 |
| 0.5 | vertebra | 0.0178 | 0.0310 | 0.062 |
| 0.25 | sphere | 0.0059 | 0.0111 | 0.044 |
| 0.25 | vertebra | 0.0059 | 0.0126 | 0.050 |
| 0.1 | sphere | 0.0020 | 0.0032 | 0.032 |
| 0.1 | vertebra | 0.0017 | 0.0029 | 0.029 |

## 3D reconstruction from the segmented centres (Q7)

MRE = distance between the reconstructed and true marker in CK. REM = distance between the two back-projection lines.

| Phantom | Marker | MRE (mm) | REM (mm) |
|---|---|---|---|
| sphere | M1 | 0.0004 | 0.0006 |
| sphere | M2 | 0.0012 | 0.0020 |
| sphere | M3 | 0.0015 | 0.0003 |
| vertebra | M1 | 0.0006 | 0.0014 |
| vertebra | M2 | 0.0010 | 0.0009 |
| vertebra | M3 | 0.0013 | 0.0006 |
