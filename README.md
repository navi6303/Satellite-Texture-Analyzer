# Satellite Texture Analyzer

See [DEPLOYMENT.md](DEPLOYMENT.md) for the GitHub and Streamlit Community Cloud setup.

Run `python -m pip install -r requirements.txt` and `python -m streamlit run app.py` from this directory. Run the suite with `python -m pytest -q`.

Upload the original image, confirm a subset, inspect its native pixel matrix, generate grayscale or panchromatic intensities, then equalize them. Draw or enter at least three texture windows per class on that subset.

After analysis, open **Follow one patch** for the calculation, **Compare textures** for the bar charts, or **Checks and downloads** for the full report.

The [analysis guide](satellite-texture-analysis/README.md) documents the controls, output filenames, and command to reproduce the supplied TCI run. [CHANGELOG.md](CHANGELOG.md) maps the changes to the eight audit findings.


On Windows with Python 3.12, use the same interpreter to install the dependencies and launch Streamlit.

```powershell
cd C:\SIP_PROJECT
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run app.py
```

The app uses NumPy for GLCM pair counting when scikit-image is absent. To enable the optional accelerator, run `py -3.12 -m pip install scikit-image`.
