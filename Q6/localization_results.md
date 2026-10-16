# Question 6 - Marker Localization Results

## Localization at 0.1 mm per pixel

Errors are distances on the detector in mm. One pixel is 0.1 mm.

| Image | Marker | True (u, v) | Segmented (u, v) | Segmented error | Error without background removal | Visual (u, v) | Visual error |
|---|---|---|---|---|---|---|---|
| sphere A | M1 | (22.76, 13.17) | (22.756, 13.166) | 0.0003 | 0.0003 | (22.96, 14.82) | 1.666 |
| sphere A | M2 | (3.92, 78.99) | (3.919, 78.990) | 0.0001 | 0.0058 | (4.67, 78.16) | 1.121 |
| sphere A | M3 | (-12.40, 73.09) | (-12.403, 73.088) | 0.0008 | 0.0050 | (-12.89, 72.57) | 0.711 |
| sphere B | M1 | (22.14, -66.89) | (22.140, -66.894) | 0.0010 | 0.0058 | (22.75, -68.35) | 1.576 |
| sphere B | M2 | (4.17, 42.73) | (4.166, 42.736) | 0.0022 | 0.0034 | (4.24, 44.23) | 1.493 |
| sphere B | M3 | (-12.44, -67.44) | (-12.439, -67.439) | 0.0008 | 0.0060 | (-13.04, -69.14) | 1.803 |
| vertebra A | M1 | (22.76, 13.17) | (22.757, 13.166) | 0.0004 | 0.0147 | (22.77, 11.79) | 1.375 |
| vertebra A | M2 | (3.92, 78.99) | (3.921, 78.991) | 0.0012 | 0.0055 | (4.21, 80.43) | 1.466 |
| vertebra A | M3 | (-12.40, 73.09) | (-12.404, 73.088) | 0.0011 | 0.0089 | (-13.12, 74.95) | 1.997 |
| vertebra B | M1 | (22.14, -66.89) | (22.138, -66.893) | 0.0028 | 0.0106 | (22.62, -68.59) | 1.763 |
| vertebra B | M2 | (4.17, 42.73) | (4.166, 42.736) | 0.0022 | 0.0155 | (4.30, 44.74) | 2.015 |
| vertebra B | M3 | (-12.44, -67.44) | (-12.440, -67.439) | 0.0010 | 0.0056 | (-13.25, -69.13) | 1.877 |

## Effect of pixel size

| Pixel size (mm) | Phantom | Mean error (mm) | Max error (mm) | Max error (pixels) |
|---|---|---|---|---|
| 1.0 | sphere | 0.0408 | 0.0629 | 0.063 |
| 1.0 | vertebra | 0.0393 | 0.0614 | 0.061 |
| 0.5 | sphere | 0.0087 | 0.0197 | 0.039 |
| 0.5 | vertebra | 0.0088 | 0.0183 | 0.037 |
| 0.25 | sphere | 0.0054 | 0.0089 | 0.036 |
| 0.25 | vertebra | 0.0056 | 0.0107 | 0.043 |
| 0.1 | sphere | 0.0009 | 0.0022 | 0.022 |
| 0.1 | vertebra | 0.0015 | 0.0028 | 0.028 |

## 3D reconstruction from the segmented centres (Q7)

MRE = distance between the reconstructed and true marker in CK. REM = distance between the two back-projection lines.

| Phantom | Marker | MRE (mm) | REM (mm) |
|---|---|---|---|
| sphere | M1 | 0.0003 | 0.0004 |
| sphere | M2 | 0.0008 | 0.0007 |
| sphere | M3 | 0.0005 | 0.0000 |
| vertebra | M1 | 0.0008 | 0.0014 |
| vertebra | M2 | 0.0007 | 0.0014 |
| vertebra | M3 | 0.0006 | 0.0002 |
