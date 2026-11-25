# Deploy SIP on Streamlit Community Cloud

SIP is a Python Streamlit app. The repository's `app.py` launches the implementation in `satellite-texture-analysis/app.py`. The rectangle-selection canvas ships with the project and needs no frontend build.

## 1. Put the project on GitHub

Extract the ZIP. Put the **contents** of the extracted `SIP_PROJECT` folder at the root of your GitHub repository, so the repository looks like this:

```text
app.py
requirements.txt
.streamlit/config.toml
satellite-texture-analysis/
DEPLOYMENT.md
```

Make sure GitHub shows `.streamlit/config.toml`. Windows may hide folders with a leading dot in some tools. Keep the application subfolder intact so the canvas, modules, sample data and bundled analysis checks are available. Upload your own satellite raster through the running app.

## 2. Create the Streamlit app

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub. Give Streamlit access to your repository.
2. Choose **Create app → Yup, I have an app**.
3. Select your repository and the branch you pushed, typically `main`. Enter **`app.py`** as the main file path.
4. In **Advanced settings**, choose **Python 3.12**, then save and deploy. SIP needs no API keys or secrets.
5. For a first check, upload the bundled `satellite-texture-analysis/sample_data/synthetic_multispectral_sample.tif`. Select a subset or use the whole image, then generate and equalize the intensity image. Choose **One crop per class** in the sidebar, confirm one box for each class and calculate the GLCMs and eigenvalues. Open **Compare textures** to check the class-distance chart, then download the results ZIP from **Checks and downloads**.

## Run locally with the same setup

From the extracted `SIP_PROJECT` folder in a Python 3.12 environment:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

To run the test suite from the same directory:

```sh
python -m pytest -q
```

## Troubleshooting

- If the app cannot import a module or load the selection canvas, confirm that the GitHub branch contains the root requirements and the entire application subfolder.
- The app accepts up to 400 MiB per file (419,430,400 bytes). The ENVI header and its binary companion each inherit this limit. The limit controls file acceptance. Raster size, dimensions, band count and simultaneous sessions determine processing memory. If the cloud instance runs out of resources on a large scene, upload a smaller raster or use the local app. A smaller analysis subset limits subsequent processing.
- Analysis outputs are provided as downloads. Download your results before closing a hosted session.

[Streamlit deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) · [Community Cloud resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-resources-and-limits)
