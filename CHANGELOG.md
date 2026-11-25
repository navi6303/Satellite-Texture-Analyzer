# Changelog

## Preprocessing matrix walkthrough

- Added subset-band previews and a real-pixel calculation example for weighted color grayscale or native raster band averaging.
- Added matched input and equalized matrices, the full-subset histograms, cumulative counts, and the exact intensity lookup table used in processing.


## College presentation view

- Replaced the long results screen with a one-patch walkthrough, a texture comparison and a separate checks and downloads tab. Each direction has its measured counts, normalized frequencies and optional count audit.
- Replaced overlapping spectra and shaded bands with bar charts for class means, individual patches and the six unique standardized distances. Three patches fit on each patch chart page. Class means use all patches.
- Kept the separate λ1–λ5 plots, sampling options, scientific checks, CSVs and existing figure filenames. The Markdown report now opens with a short walkthrough and comparison charts before the technical tables.

## Per-patch GLCM occurrences

- Traced confirmed native boxes to their Rasterio windows. Added exact-shape checks on equalized, quantized, PAN and validity arrays. An input mask mismatch now raises an error.
- Kept the integer directed count matrix for each patch and offset throughout analysis, demonstration and export. Listed every occupied pair and computed percentage frequencies from the raw count. Corrected displayed row and column offsets to match the calculation.
- Added one audit row per patch and offset, including coordinates in both frames, native dimensions, valid pixels, possible and valid pairs, raw and symmetric sums, and normalized sum. The canvas shows native pixel area before confirmation.
- Verified the real TCI in two adjacent small boxes, the 12-patch export, and the coordinate transform. The automatic run checks independently reconstruct the integer counts as well as the normalized matrix.

## Original-image subset and GLCM validation

- Added a zoomable original-image rectangle with move, resize, confirm and numeric entry. The review shows the original footprint, extracted composite, dimensions, PAN and equalized PAN. Selecting a new subset discards the previous results.
- Copy all bands from native source windows before averaging. Preserve sample values, dimensions, joint validity, CRS and translated transform. ENVI headers with a companion array, float Radiance HDR and multiband GeoTIFFs are supported.
- The draw path formerly tested an uncomputed horizontal-pair field and reported its failure using the valid-pixel count. The coordinate-list path bypassed it. Both now count valid neighboring pairs in the actual patch for every selected offset. Failures give the count, angle, distance and required minimum of 10 directed pairs.
- Retained the four-class, three-or-more-box workflow, intermediate matrices and fixed-bin default. Fixed switching demonstration patches and regenerating the export. Added original and subset-local coordinates to metadata.
- Added regression tests for masks, HDR, subset-only averaging and equalization, the reported valid-pixel case, angle-specific thresholds and canvas interactions. See `TEST_RESULTS.txt` for the real TCI workflow run.

## Earlier audit fixes

| Audit problem | Change |
| --- | --- |
| 1. Tone dominates the eigenvalues | Added local percentile/min–max and raw-pan bins, a transition-only comparison, and measured λ1-to-diagonal gaps. The global fixed-bin method remains available. |
| 2. λ1 interpreted as smooth water and range called variance | The report computes its findings from feature ranks, distances and held-out predictions. It documents the diagonal-limit occupancy interpretation. |
| 3. Suspect Water label | Added native RGB and equalized statistics, occupancies, two warning rules, thumbnails and mapped footprints. The TCI batch run separately audits the four original full rectangles. |
| 4. Single sample per class | The canvas and coordinate lists accept replicated windows. Seeded region sampling supplies missing samples. Class tables and plots include sample SD. |
| 5. One direction | Four normalized symmetric directional GLCMs are averaged by default. Angle subsets and distances 1–3 are selectable. |
| 6. Unscaled distances | Added matched raw and standardized distances, Fisher ratios and leakage-free training-fold scaling. Trace and diagonal share contribute one model dimension. |
| 7. Overstated discrimination | Added blocked nearest-centroid and RBF SVM comparisons, fold-level predictions, confusion matrices, accuracy, precision and recall. The report gives both directions of Urban/Bare Soil confusion and reports ties. |
| 8. Figure 5 scale and labels | Each eigenvalue has a mean/SD figure. Non-negative data start at zero, signed data retain their SD range, and value labels use point offsets. Regenerated the combined spectrum and added classic-feature and confusion plots. |

Every analysis executes pytest on its computed patches and writes the outcome to the report. Raster processing uses tiled intermediates. The ZIP retains the original result filenames and adds metadata for sampling, quantization, spatial folds, software versions and the source checksum.

Windows startup fix: scikit-image is optional. The NumPy implementation counts the same pixel pairs when the accelerator is unavailable.


## Handbook and user-drawn boxes

- The default analyzes each complete native-pixel rectangle from the canvas or coordinate list. The rectangular drawing tool preserves independent width and height. Region sampling is opt-in.
- Pipeline A diagonalizes normalized P, keeps signed eigenvalues in descending magnitude order, and exports a box/class audit, directional symmetric counts and denominators. Pipeline B separately diagonalizes within-box directional feature covariance.
- Corrected energy to √ASM and entropy to bits. Added dissimilarity, correlation, marginal mean and variance, and maximum probability.
- Figures identify the matrix, bands, window sizes, offsets and sample counts. Labels sit above the upper SD caps. Single-box classes have no sample error bars. Replication-dependent results are marked unavailable where needed.

## Shared five-bin default

The GUI and batch runner now use identical five bins on the globally equalized PAN by default. Patch-wise rescaling remains selectable. The TCI reproduction coordinates now come from the latest user export. Reports and chart captions identify the selected quantization, and a run-specific test checks the fixed-bin output against the equalized source pixels.
