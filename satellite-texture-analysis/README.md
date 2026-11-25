# Satellite Texture Analyzer

Launch the Streamlit workflow from either the project root or this directory. Use Python 3.10 or newer.

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload the original raster, inspect its bands and choose the display composite. Draw a rectangle **on the original image** and drag it or its handles to adjust its position and size. Click **Confirm Box**, or use **Enter subset coordinates instead**. Review the outline, extracted subset, dimensions and band count. Click **Confirm Subset and Generate Grayscale Image** or **Confirm Subset and Generate Panchromatic Image**, then **Enhance Contrast**. **Select a different subset** clears the previous patches and results.

The subset contains all native bands at their original resolution. Display-channel choices affect only the preview. For ordinary decoded color images such as PNG or JPEG, the app uses Gray = 0.114 × B + 0.587 × G + 0.299 × R. A single-band image keeps its native intensities. Satellite GeoTIFFs, ENVI rasters and HDR rasters retain band averaging. The app then computes the 8-bit stretch and equalization histogram from the full set of jointly valid subset pixels. GeoTIFF and ENVI are read in windows. For ENVI, upload the header and matching binary data file. Standalone Radiance HDR is also supported.

To compare four crops for the assignment, choose **One crop per class** under **Analysis settings → Sampling and gray levels → Pixels to analyze**. Confirm one non-overlapping rectangle on the equalized image for each class. To equalize the entire input image before selecting those four crops, select **Use the whole image** in the subset step.

For the multi-patch workflow, confirm at least three non-overlapping rectangles for each class. The canvas and manual entry both use subset-local coordinates and the same validation. **Complete user bounding boxes** measures every valid pair in the confirmed windows. **Sample subpatches within regions** draws reproducible non-overlapping crops inside the confirmed regions. A patch needs at least 10 directed neighboring pairs for **each** selected angle and distance, counted before symmetrizing. Both endpoints must be valid. This is a floor for computing the matrix; small samples can still give unstable texture estimates.

The default bins the globally equalized subset on a shared 8-bit scale. The five inclusive intervals are 0–51, 52–102, 103–153, 154–204 and 205–255. Each interval has the same meaning in every patch. The sidebar also offers per-patch percentile or min–max stretching and fixed raw-pan bins. The analyzer averages normalized symmetric GLCMs over 0°, 45°, 90° and 135° by default, at d=1 and Ng=5.

Inspect the footprint map and thumbnails before accepting the labels. Every patch has five-bin occupancy, raw RGB and equalized mean/SD, and warnings. Bright Water or low gray-level diversity requires review. The true-color patch statistics use native values; display contrast is separate. A single-band input has no RGB statistics.

## Reproduce the supplied TCI analysis

The supplied verification run uses the real three-band TCI and a subset with original origin (600, 625), width 9,450 and height 8,450. The twelve verification windows in `sample_data/tci_validation_boxes.json` lie inside the four original labeled regions in `tci_regions.json`. They are example windows chosen for this test. The UI always uses the rectangles confirmed in the current session.

From this directory, run:

```sh
python run_analysis.py "/path/to/TCI (3).tif" --subset 600 625 9450 8450 --regions sample_data/tci_validation_boxes.json --output satellite_texture_analysis_package.zip
python -m pytest -q
python tests/verify_tci_export.py "/path/to/TCI (3).tif" satellite_texture_analysis_package.zip
python tests/verify_tci_patch_ui.py "/path/to/TCI (3).tif"
```

Batch coordinates for both `--subset` and `--regions` are zero-based **original** pixel coordinates, with width and height in pixels. If the subset is omitted, the batch runner uses the smallest rectangle enclosing the supplied windows. The output metadata gives the subset origin and both coordinate systems for every patch.

To reproduce the separate within-site analysis on five seeded subpatches per original user box, run the following from this directory. Every sampled rectangle lies entirely inside its corresponding user box.

```sh
python run_analysis.py "/path/to/TCI (3).tif" --regions tci_regions.json --settings within_box_settings.json --output within_box_run.zip
```

Use `--settings settings.json` to override any field in `modules.workflow.Settings`, for example `{"exact_boxes": false, "n": 6, "block_size": 256, "distance": 2, "angles": [45, 135], "transition": false}`. Only sampled mode draws windows with the seeded generator. Coordinates, configuration and package versions are also recorded in the result ZIP.

## Inspect preprocessing

Open **Original subset pixel values** to inspect a native band. The preview defaults to the top-left 10 × 10 pixels and can expand to 20 × 20 or 30 × 30. **How the subset becomes one intensity per pixel** shows the conversion, including B, G and R tabs for color images and a worked calculation with real pixel values. After **Enhance Contrast**, the next section shows the 8-bit input, before and after histograms, cumulative histogram, the applied lookup table, and the equalized values. A worked lookup shows how the intensity of one preview pixel changes. The preview size never changes the processing footprint.

## Present the results

**Follow one patch** lets you choose a class, a confirmed box and a direction. Read from left to right through the equalized patch, its five-level image, directed counts and normalized frequencies. The last two steps show the matrix averaged across the selected directions and its five eigenvalues. The direction control changes which counts you inspect. The final matrix continues to use all directions selected for the analysis.

**Compare textures** opens with grouped bars and a table of the five mean eigenvalues by class. Choose a class to show its individual patches or its mean. The patch chart shows up to three boxes at a time, with additional pages for larger collections. The class means always include every measured box. For the separation chart, the app averages each class's normalized patch GLCMs, computes the five eigenvalues of each combined matrix and measures Euclidean distance directly between each of the six pairs of class vectors. The chart caption names the closest and farthest pairs from the current run. A separate selector gives a bar chart for each eigenvalue, and all five charts are included in the results ZIP.

**Checks and downloads** contains the patch map, warnings, full count audit, classifiers, feature rankings and pytest output. The download includes all detailed CSVs, charts and metadata. To check a particular count, open **Inspect exact counts** in the walkthrough. For example, a fully valid 12 × 8 box at horizontal distance one has 88 pairs. Adding the reverse pairs gives a total of 176 before normalization.

## Read the outputs

- `analysis_report.md` contains the matrix/label audit, raw and z-scored eigenvalue distances, class means and sample SD, model-specific confusion matrices, precision and recall, and the run-specific pytest output.
- `tables/patch_sanity.csv` gives patch diagnostics. `tables/eigenvalues_summary.csv` gives native pixel counts, d=1 horizontal pairs, and five signed eigenvalues. “Valid Pairs” totals directional observations over the selected offsets, before symmetrizing. `tables/cv_predictions.csv` lists every held-out prediction when CV is feasible.
- `plots/` contains the five individual eigenvalue charts, grouped class means, per-class patch bars, the six-pair separation chart, classic features, confusion matrix, thumbnails, footprint map and equalization histograms. If several configurations tie, the report lists every tie; the figure identifies the displayed configuration.
- The first sampled patch of each class retains its legacy basename. Subsequent samples add a numbered suffix. Equalized patch PNGs and quantized display PNGs retain their separate roles across all modes.
- `tables/matrix_audit.csv` binds every window to its class, matrix shape, sum, diagonal and spectrum. `tables/directional_glcms.csv` contains the symmetric counts and twice the valid directional pair count as denominator. `tables/directional_feature_pca.csv` reports the separate within-box covariance eigenvalues when at least two offsets are selected.
- Trace and diagonal share are the same statistic. Both are exported, and models count it once. Energy is √ASM; entropy is in bits. Signed eigenvalues of P are ordered by absolute magnitude. The optional transition-only comparison keeps the three primary feature sets unchanged.

The multi-patch modes require at least three boxes per class. Multiple boxes can still fall in a single block group. With one crop per class, the results describe the four selected crops and held-out classification is unavailable.

Validation holds out every occupied geographic block group once. Patches touching a common grid cell are grouped together, including transitive connections. Classifiers use training-fold scaling; descriptive distances use pooled scaling. CV is unavailable if a held-out block would remove a whole class from training. Multiple subpatches in one labeled site measure within-site variability. Independent sites are needed to evaluate transfer to new locations.

A nearly diagonal globally equalized GLCM can have an eigenvalue spectrum governed largely by gray-level occupancy. A high λ1 alone does not identify smoothness or water. Symmetric probability matrices can have negative eigenvalues; plots preserve observed negative values and their signed SD range.
