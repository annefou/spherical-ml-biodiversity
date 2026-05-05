# spherical-ml-biodiversity — pedagogy + worked replication of
# DeepSphere on Copernicus data, tied to biodiversity outcomes.
#
# CPU-only image. Reproduces all current notebooks end-to-end via
# Snakemake. Datasets (ERA5 Copernicus C3S, Copernicus Marine SST,
# GBIF biodiversity occurrences) are fetched at first run by the
# relevant notebooks, with caches in data/.
#
# When notebooks 04–08 land, this Dockerfile will need DeepSphere
# (graph-based spherical CNN) and possibly additional pip dependencies
# pinned for that work. Kept minimal for now to match the pedagogy
# phase.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        libgeos-dev \
        proj-bin \
        proj-data \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "numpy>=2.2,<2.3" \
        healpy \
        cartopy \
        matplotlib \
        scipy \
        pyproj \
        h3 \
        requests \
        jupytext \
        nbclient \
        ipykernel \
        jupyter \
        snakemake

WORKDIR /app
COPY . /app

CMD ["snakemake", "--cores", "1", "all"]
