# spherical-ml-biodiversity — pedagogy + worked spherical-ML head-to-head
# on synthetic and real Copernicus / NOAA SST data, with a real marine
# heatwave (2011 Ningaloo Niño) plus GBIF marine biodiversity overlay.
#
# CPU-only image. Reproduces all six notebooks end-to-end via Snakemake.
# External data (NOAA OISST v2.1 SST + climatology slice, GBIF marine
# occurrences) is fetched at first run by notebook 05; the GBIF cache
# CSV is shipped in the repo for offline / CI reproducibility, the
# small NetCDF caches are rebuilt on first run.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        libgeos-dev \
        proj-bin \
        proj-data \
    && rm -rf /var/lib/apt/lists/*

# Pin top-level dependencies; mirrors environment.yml. Update both
# files together — environment.yml is the source of truth for CI,
# this Dockerfile is the source of truth for reproducible local
# Docker runs and the GHCR image.
RUN pip install --no-cache-dir \
        "numpy>=2.3" \
        "healpy==1.19.0" \
        "healpix-plot==0.1.1" \
        "cartopy==0.24.*" \
        "matplotlib==3.10.*" \
        "scipy" \
        "pyproj" \
        "xarray==2025.4.*" \
        "netcdf4" \
        "dask" \
        "pandas" \
        "cftime" \
        "scikit-learn" \
        "h3" \
        "pygbif" \
        "requests" \
        "jupytext" \
        "nbclient" \
        "ipykernel" \
        "jupyter" \
        "snakemake" \
        "zenodo-get"

WORKDIR /app
COPY . /app

CMD ["snakemake", "--cores", "1", "all"]
