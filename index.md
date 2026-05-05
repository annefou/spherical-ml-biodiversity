# Spherical ML for biodiversity, EO, and Copernicus

> What spherical machine learning is, why it matters for global Earth science, and a worked replication on Copernicus data tied to biodiversity outcomes.

Follow-up to [**dggs-biodiversity-bias**](https://github.com/annefou/dggs-biodiversity-bias) (concept DOI: [10.5281/zenodo.19848749](https://doi.org/10.5281/zenodo.19848749)). That repository established the **HEALPix family** as the right common DGGS substrate for the integration of biodiversity with high-resolution Copernicus EO and Destination Earth climate models — partly because *the climate-model and spherical-ML sides already live on HEALPix (DeepSphere, foscat, sphere-harmonic transforms)*. This repository **operationalises the spherical-ML claim**.

## The argument, in three steps

1. **What a flat CNN sees on a global raster is a projection-distorted view of the sphere.** Notebook 01 makes the failure mode visible — the same physical feature on the sphere becomes a wildly different *pixel-shape* depending on its latitude.
2. **Spherical ML respects the geometry of the sphere.** Notebook 02 demonstrates rotation equivariance: a translation on the lat-lon raster is *not* a rotation on the sphere (50.6% pixel disagreement at 60°N), and a CNN filter trained at the equator fails to detect the same physical feature at higher latitudes (response collapses 100% → 55% from 0°N to 80°N). Notebook 03 explains why HEALPix is the right substrate for the fix — equal-area + iso-latitude rings + NESTED bit-shift hierarchical refinement + native sphere-harmonic transforms.
3. **Worked example on Copernicus Marine data tied to biodiversity.** Notebooks 04–05 (forthcoming) detect marine heatwaves on Copernicus Marine SST aggregated to HEALPix (Hobday et al. 2016 definition) and connect them to documented marine-biodiversity impacts (Smale 2019 global, Wernberg 2016 Australian kelp regime shift). Notebook 06 stacks Copernicus Marine SST + GBIF biodiversity occurrences + atmospheric-river detection outputs from the **separate `deepsphere-ar-replication`** repository on the common HEALPix substrate. The DeepSphere AR replication itself — a paper-rooted replication of Defferrard et al. 2020 on ClimateNet — lives in its own repo so its FORRT chain can be paper-rooted while the chain for this repo is question-rooted (PICO).

## What this Jupyter Book contains

Three pedagogy notebooks shipped today:

1. **{doc}`What a flat CNN sees on a global raster <notebooks/01_flat_cnn_failure_mode>`** — same spherical cap (10° angular radius) at 0°N, 40°N, 70°N, rendered on a lat-lon raster (top row, distorts) vs orthographic centred on the cap (bottom row, preserves shape). The contrast is the case for spherical ML.
2. **{doc}`Rotation equivariance — what spherical CNNs preserve <notebooks/02_rotation_equivariance>`** — quantitative demos: rotation on the sphere is not translation on the raster (50.6% pixel disagreement at 60°N), and an "equator-cap detector" matched filter fails on the same physical feature at higher latitudes (100% → 55% peak response collapse from 0°N → 80°N). Closes with what a spherical CNN does differently and where the technique genuinely shines vs where it doesn't.
3. **{doc}`Why HEALPix is the right substrate for spherical ML <notebooks/03_why_healpix>`** — four properties: equal-area cells, iso-latitude rings (vs H3 hexagonal which breaks them), NESTED bit-shift hierarchical refinement (parent = `pix >> 2` verified against `healpy`), and graph structure for DeepSphere-style graph CNNs. Closes with the spherical-ML ecosystem that lives on HEALPix: `healpy`, `healpix-geo` (GRID4EARTH path), DeepSphere, DLWP-HEALPix, `foscat` (FIESTA), DISCO.

Notebooks 04 and 05 are also shipped:

4. **{doc}`Spherical-ML head-to-head on synthetic global SST <notebooks/04_mhw_detection_copernicus_marine>`** — both pipelines train on samples with marine-heatwave caps only at |lat| ≤ 20° and test at four latitude bands. The flat lat-lon matched-filter baseline collapses from 100 % accuracy in-distribution to 50 % (chance, F1 = 0) at 70–80°N because the spherical-cap shape distorts under the lat-lon projection. The sphere-aware HEALPix `Cl` features hold 71–91 % across all latitudes.
5. **{doc}`A real marine heatwave and the biodiversity it touched <notebooks/05_real_mhw_and_biodiversity>`** — the documented 2011 Western Australian "Ningaloo Niño" MHW (Wernberg et al. 2016) on NOAA OISST v2.1, projected onto the same HEALPix-NESTED substrate. GBIF marine biodiversity occurrences from 2011 in the same region are overlaid on the MHW footprint to count cell-level exposure. Demonstrates that the spherical substrate carries the climate signal and the biodiversity signal together — without re-projecting onto a distorting lat-lon raster in the process.

Notebook 06 (cross-discipline transfer demonstration: a sphere-aware model trained in one discipline applied without retraining to another on the same HEALPix substrate) is forthcoming.

## Connection to ESA GRID4EARTH

This repository is the spherical-ML side of the case [**ESA GRID4EARTH**](https://www.grid4earth.eu) makes for **Ellipsoidal HEALPix as a Common DGGS** for Copernicus EO and Destination Earth — bridging spherical climate models and ellipsoidal Earth-observation data on a single ellipsoidally-correct, hierarchical, scalable DGGS, with sphere-aware ML on top.

## Citation

If this material is useful in your own work please cite the repository (DOI on first release, concept DOI minted by Zenodo via the GitHub integration) and the foundational references in `CITATION.cff` — Górski et al. 2005 (HEALPix), Cohen et al. 2018 (Spherical CNNs), Defferrard et al. 2020 (DeepSphere), Karlbauer et al. 2024 (DLWP-HEALPix).
