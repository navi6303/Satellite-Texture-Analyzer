# Sample Data Directory

This folder contains demonstration datasets and information for testing the **Satellite Texture Analyzer** application.

## Files

1. `synthetic_multispectral_sample.tif`
   - A deterministic 4-band GeoTIFF raster ($256 \times 256$ pixels).
   - **Bands:**
     - Band 1: Blue ($\sim 490$ nm)
     - Band 2: Green ($\sim 560$ nm)
     - Band 3: Red ($\sim 665$ nm)
     - Band 4: NIR ($\sim 842$ nm)
   - **Spatial Reference:** EPSG:32632 (WGS 84 / UTM Zone 32N)
   - **Land-Cover Quadrants:**
     - Top-Left: **Water Body** (smooth, low reflectance, very low NIR, subtle ripple)
     - Top-Right: **Vegetation / Dense Forest** (high NIR, low Red, organic fine canopy texture)
     - Bottom-Left: **Agricultural Fields** (rectangular plots with directional crop furrows)
     - Bottom-Right: **Urban / Built-up Area** (high-frequency street grid, contrasting rooftops)

## Acquiring Real Satellite Imagery

You can download real Sentinel-2 multispectral imagery free of charge from:
- **Copernicus Data Space Browser:** https://dataspace.copernicus.eu/browser/
- **USGS EarthExplorer:** https://earthexplorer.usgs.gov/

When downloading Sentinel-2 products:
- Level-2A (Bottom-of-Atmosphere surface reflectance) is recommended.
- You can upload GeoTIFFs containing B02 (Blue), B03 (Green), B04 (Red), and B08 (NIR) directly into the application.
