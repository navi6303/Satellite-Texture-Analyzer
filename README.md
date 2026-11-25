# Satellite-Texture-Analyzer

Satellite-Texture-Analyzer is a Streamlit app that analyzes satellite images and compares the textures of land-use and land-cover classes such as water, vegetation, built-up areas, and bare soil. It describes each texture using a **gray-level co-occurrence matrix (GLCM)**, which counts how often pairs of gray levels occur at a chosen pixel offset. The eigenvalues of these matrices provide features for comparing the selected classes.

## Problem Statement

A satellite image can contain areas with similar brightness but different patterns. A water surface may look smooth, while a built-up area may show repeated edges and variations in brightness. Measuring these patterns gives a consistent way to compare selected areas and study differences in land cover.

## Objective

Extract texture features from selected satellite image patches and compare four land-cover classes using five class-level GLCM eigenvalues and the Euclidean distances between them.

## About the Methodology

1. **Upload an image.** Load a satellite raster and inspect its bands and preview.
2. **Select a region.** Confirm the part of the image that contains the land-cover classes you want to compare.
3. **Generate an intensity image.** For a multiband satellite raster, the app averages the bands to create a panchromatic-like image. It uses weighted grayscale for standard RGB images and keeps the intensity values of single-band images.
4. **Enhance the contrast.** Apply histogram equalization across the selected region.
5. **Select texture patches.** Label four land-cover classes and confirm three representative, non-overlapping patches for each class.
6. **Reduce to five gray levels.** With the default settings, the app maps each equalized patch to levels 0 through 4 using the same intensity bins.
7. **Build the patch GLCMs.** For each patch, the app counts valid pixel pairs at the selected offsets and forms a symmetric, normalized 5 × 5 GLCM across the selected directions.
8. **Combine the GLCMs by class.** Average the three normalized patch matrices element by element, giving each patch equal weight.
9. **Extract the class eigenvalues.** Compute the five signed eigenvalues of each combined matrix to form one texture feature vector per class.
10. **Compare the classes.** Calculate the six Euclidean distances directly between the four class vectors. The app displays eigenvalue charts and a class-distance chart that identifies the closest and farthest pairs for the current image.

## Tech Stack

- **Python 3.12 and Streamlit** for the application.
- **NumPy and pandas** for matrix calculations, eigenvalues, and result tables.
- **Rasterio and Affine** for raster input and spatial coordinates.
- **OpenCV and Pillow** for image operations and previews.
- **Matplotlib** for charts, and **HTML, CSS, and JavaScript** for the region-selection canvas.
- **scikit-learn** for optional validation checks and **pytest** for automated tests. The pinned dependencies are listed in [requirements.txt](requirements.txt).

## Installation

Install Python 3.12 and Git, then run these commands from a terminal:

```bash
git clone https://github.com/navi6303/Satellite-Texture-Analyzer.git
cd Satellite-Texture-Analyzer
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use Python 3.12 for both commands. On Windows, you can replace `python` with `py -3.12`.

## Deployment

Open the app on [Streamlit Community Cloud](https://satellite-texture-analyzer.streamlit.app/). See the [deployment guide](DEPLOYMENT.md) for setup details.

## Usage

1. Upload a satellite image, such as a GeoTIFF. For ENVI input, upload the header and its matching binary file. The app allows up to 400 MB per file.
2. Draw and confirm a region on the original image, or choose **Use the whole image**.
3. Select **Confirm Subset and Generate Panchromatic Image** or the corresponding grayscale button, then select **Enhance Contrast**.
4. Name your four texture classes. Under **Sampling and gray levels**, keep **Complete user bounding boxes** selected. Draw and confirm three representative boxes per class.
5. Select **Calculate GLCM & Eigenvalues**.
6. Open **Follow one patch** to inspect the five-level image and the GLCM calculation. Use **Compare textures** to view the eigenvalue and distance charts, or **Checks and downloads** to save the results.

## Screenshots

### Application home page

<img width="1917" height="782" alt="image" src="https://github.com/user-attachments/assets/052209d7-9cc0-4804-aaa6-364376b85a3a" />

### Panchromatic image and histogram equalization

<img width="1757" height="815" alt="image" src="https://github.com/user-attachments/assets/86e705a5-ee26-4818-a3b6-5688904078c7" />

### Uploaded satellite image and selected region

<img width="1096" height="783" alt="image" src="https://github.com/user-attachments/assets/360de036-2688-4185-8f7d-24201535fcac" />

### Texture patches and 5-level quantization

<img width="1182" height="476" alt="image" src="https://github.com/user-attachments/assets/073eba1b-3324-4dfc-8867-08078579b1de" />


### GLCM calculation

<img width="2000" height="679" alt="image" src="https://github.com/user-attachments/assets/90e3b6d6-c1a5-491e-85ee-b7797cb25a02" />


### Eigenvalue and class-distance charts

<img width="1565" height="796" alt="image" src="https://github.com/user-attachments/assets/715e60d4-21e8-4eaa-9aa6-c20938d22223" />
<img width="1702" height="761" alt="image" src="https://github.com/user-attachments/assets/4282790b-9197-433c-a3f0-a00a0f39286a" />

## Conclusion

The project demonstrates how satellite image textures can be measured and compared. Combining patch GLCMs gives each selected land-cover class a five-value texture signature, and the distance chart shows which classes have the most similar or different signatures in the analyzed image.
