# spherical-ml-biodiversity

> What spherical machine learning is, why it matters for global Earth science, and a worked replication on Copernicus data tied to biodiversity outcomes.

Follow-up to [**dggs-biodiversity-bias**](https://github.com/annefou/dggs-biodiversity-bias) (concept DOI: [10.5281/zenodo.19848749](https://doi.org/10.5281/zenodo.19848749)). That repository established that the **HEALPix family** is the right common DGGS substrate for the integration of biodiversity with high-resolution Copernicus EO and Destination Earth climate models — partly *because the climate-model and spherical-ML sides already live on HEALPix (DeepSphere, foscat, sphere-harmonic transforms)*. This repository **operationalises** the spherical-ML claim with a worked replication and a biodiversity-impact follow-up.

## The argument, in three steps

1. **What a flat CNN sees on a global raster is a projection-distorted view of the sphere.** Notebook 01 makes the failure mode visible: the same physical feature on the sphere becomes a wildly different *pixel-shape* depending on its latitude. CNNs trained on lat-lon-projected global data inherit this distortion silently.
2. **Spherical ML respects the geometry of the sphere.** Notebook 02 demonstrates rotation equivariance — what spherical convolutions preserve and flat CNNs don't. Notebook 03 explains why HEALPix is the right substrate: equal-area + iso-latitude rings + NESTED bit-shift hierarchical refinement + native sphere-harmonic transforms.
3. **Worked example on Copernicus Marine data tied to biodiversity.** Notebooks 04–05 (forthcoming) detect marine heatwaves on Copernicus Marine SST aggregated to HEALPix (Hobday et al. 2016 definition) and connect them to documented marine-biodiversity impacts (Smale 2019 global, Wernberg 2016 Australian kelp regime shift). Notebook 06 stacks Copernicus Marine SST + GBIF biodiversity occurrences + atmospheric-river detection outputs from the **separate `deepsphere-ar-replication`** repository on the common HEALPix substrate. The DeepSphere AR replication itself — a paper-rooted replication of Defferrard et al. 2020 on ClimateNet — lives in its own repo so its FORRT chain can be paper-rooted while the chain for this repo is question-rooted (PICO).

## Status

Work in progress.

- **Notebooks 01–03** — pedagogy (failure mode of flat CNNs on a global raster, rotation equivariance, why HEALPix). Shipped and CI-verified.
- **Notebook 04** — spherical-ML head-to-head on synthetic global SST. A flat lat-lon matched-filter detector trained on equator-shape MHWs collapses from 100 % accuracy in-distribution to 50 % (chance, F1 = 0) at 70–80°N, while sphere-aware HEALPix Cl features hold 71–91 % across all latitudes. Shipped.
- **Notebook 05** — real Ningaloo-Niño-2011 marine heatwave on NOAA OISST → HEALPix substrate, with GBIF marine biodiversity overlay. Shipped.
- **Notebook 06** — cross-discipline spherical-ML transfer: a sphere-aware model trained in one discipline (astrophysics, where the HEALPix ML stack is most mature) applied without retraining to a climate / biodiversity task on the same HEALPix substrate, vs a flat baseline trained from scratch. The *positive* spherical-ML payoff complementing notebook 04's failure-mode demo. Forthcoming.
- **FORRT chain** + **Research Software nanopub** — to be published once the worked example is complete.

## License

Code: **MIT** (see `LICENSE`).
Generated figures and notebook prose: **CC-BY 4.0**.
