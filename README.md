# spherical-ml-biodiversity

> What spherical machine learning is, why it matters for global Earth science, and a worked replication on Copernicus data tied to biodiversity outcomes.

Follow-up to [**dggs-biodiversity-bias**](https://github.com/annefou/dggs-biodiversity-bias) (concept DOI: [10.5281/zenodo.19848749](https://doi.org/10.5281/zenodo.19848749)). That repository established that the **HEALPix family** is the right common DGGS substrate for the integration of biodiversity with high-resolution Copernicus EO and Destination Earth climate models — partly *because the climate-model and spherical-ML sides already live on HEALPix (DeepSphere, foscat, sphere-harmonic transforms)*. This repository **operationalises** the spherical-ML claim with a worked replication and a biodiversity-impact follow-up.

## The argument, in three steps

1. **What a flat CNN sees on a global raster is a projection-distorted view of the sphere.** Notebook 01 makes the failure mode visible: the same physical feature on the sphere becomes a wildly different *pixel-shape* depending on its latitude. CNNs trained on lat-lon-projected global data inherit this distortion silently.
2. **Spherical ML respects the geometry of the sphere.** Notebook 02 demonstrates rotation equivariance — what spherical convolutions preserve and flat CNNs don't. Notebook 03 explains why HEALPix is the right substrate: equal-area + iso-latitude rings + NESTED bit-shift hierarchical refinement + native sphere-harmonic transforms.
3. **Worked head-to-head on synthetic and real Copernicus / NOAA data, plus a cross-discipline transfer demonstration.** Notebook 04 quantifies the negative case — a flat lat-lon matched filter trained on equator-shape marine-heatwaves collapses from 1.000 detection accuracy to 0.500 (chance, F1 = 0) at the 70–80° test band, while a sphere-harmonic band-pass matched filter on HEALPix-NESTED holds at 1.000 across every test latitude band. Notebook 05 grounds the argument in real data — the documented 2011 Western Australian "Ningaloo Niño" marine heatwave on NOAA OISST aggregated to HEALPix-NESTED at WGS84 ellipsoidal projection, with 94.0 percent of 765 marine GBIF records from the same year and region sitting on cells that experienced MHW conditions during the event. Notebook 06 makes the positive case — the same sphere-aware pipeline trained on a cosmology-like domain transfers without retraining to a climate-like domain at 1.000 accuracy on the shared HEALPix substrate, while the lat-lon flat baseline drops from 1.000 in-domain to 0.845. The DeepSphere atmospheric-river replication and AR-biodiversity-Europe follow-up live in the **separate `deepsphere-ar-replication`** repository so its FORRT chain can be paper-rooted (Defferrard et al. 2020 on ClimateNet) while the chains for this repo are question-rooted (PICO).

## Status

Work in progress.

- **Notebooks 01–03** — pedagogy (failure mode of flat CNNs on a global raster, rotation equivariance, why HEALPix). Shipped and CI-verified.
- **Notebook 04** — spherical-ML head-to-head on synthetic global SST. A flat lat-lon matched filter trained on equator-shape MHWs collapses from 1.000 accuracy in-distribution to 0.500 (chance, F1 = 0) at 70–80°N. The sphere-harmonic band-pass matched filter on HEALPix-NESTED — applied via `aₗₘ → aₗₘ · fₗ · bₗ` with high-pass `fₗ` and Gaussian beam `bₗ` at the cap-diameter scale — holds at 1.000 accuracy across every test latitude band. Shipped.
- **Notebook 05** — real-data Ningaloo-Niño-2011 marine heatwave on NOAA OISST aggregated to HEALPix-NESTED nside=128 at WGS84 ellipsoidal projection (via `healpix-plot` and `healpix-geo`), with 719 of 765 marine GBIF records (94.0 percent) across 113 unique species sitting on MHW-exposed cells during the documented event window. Shipped.
- **Notebook 06** — cross-discipline transfer on the spherical substrate. The same sphere-harmonic matched filter trained on a cosmology-like domain classifies a climate-like domain at 1.000 accuracy without retraining, while the lat-lon flat baseline drops to 0.845 on the same transfer. The *positive* spherical-ML payoff complementing notebook 04's failure-mode demo. Shipped.

## FORRT nanopublication chains

Three parallel atomic FORRT chains on Science Live, plus a Research Software nanopub and a Research Synthesis nanopub. Publication ongoing; concept DOI for the Research Software nanopub mints when `v0.1.0` is tagged.

- **Chain A — within-discipline latitude invariance** (notebook 04). Published 2026-05-06.
  - PICO question: <https://w3id.org/sciencelive/np/RAwXnEfd93PfqsvVkYm4XsZ4CJTjgQ_kF8NfExx_S_Xsk>
  - AIDA: <https://w3id.org/sciencelive/np/RAqWXTLUYI99UpQw4mWyGCeRMj2a02vbKaRDf-hqbFYDI>
  - FORRT Claim: <https://w3id.org/sciencelive/np/RAnfkEJBlKQ6KzlvlVdbMvN2px_Oz3Jth9dr5DAs-4XJ4>
  - FORRT Replication Study: <https://w3id.org/sciencelive/np/RAYkg28oVS9Ns_4iDocjaKYGJXQ3VifTt96CyAy56Qrko>
  - FORRT Replication Outcome: <https://w3id.org/sciencelive/np/RAydqzcPo3ZNMYU2Gk9wd4u4OgITCaXwL01IQtxyoBloA>
  - CiTO: <https://w3id.org/sciencelive/np/RABbVtjiCRIhjzf3G0oVKhYD2NyEV6LJMswjWUgezDW9I>
- **Chain C — cross-discipline transfer** (notebook 06). Published 2026-05-08.
  - PICO question: <https://platform.sciencelive4all.org/np/?uri=https://w3id.org/np/RAn4ChpN8nXUYA_z1gDSMA_pN95nm5CmmKksIpOLM3L74>
  - AIDA: <https://w3id.org/sciencelive/np/RAfdjia40JYeC3mMR2_Tc8x6JLHDwC0nUSS5ndfp5TyhE>
  - FORRT Claim: <https://w3id.org/sciencelive/np/RAnu4xXt4BmdzWmOsFRAp4XftQJuw0VAdr3xGyfoJYZZs>
  - FORRT Replication Study: <https://w3id.org/sciencelive/np/RA9MwZOzZVZeWrHRR-aEnZll1rGP_WK-CQYEOPgEHmzHU>
  - FORRT Replication Outcome: <https://w3id.org/sciencelive/np/RAvIzcWGL89mxdBXTTjgRRd0QJBBAdu7wUqkdHRCssSqs>
  - CiTO: <https://w3id.org/sciencelive/np/RAPIqs6m96yf2zaw052KE4SxnGDtrXtzEJo9WnUw5PG6c>
- **Chain B — real-data Ningaloo-Niño-2011 biodiversity exposure** (notebook 05, paper-rooted off Wernberg et al. 2016 `10.1126/science.aad8745`). Published 2026-05-08.
  - Quote-with-comment: <https://w3id.org/sciencelive/np/RAjXiO3nmUGCW-r1PMKSXKDcbsX4gQYzzHffGfCYGGFjc>
  - AIDA: <https://w3id.org/sciencelive/np/RAeUBLkmMl9AVRxnZlbInZNtYB6QL300kQIVV7wEc6OB0>
  - FORRT Claim: <https://w3id.org/sciencelive/np/RAzgbEenbZg6Xb1bDVIZmiKPc6MOSH5GdyHaTQTEutFcE>
  - FORRT Replication Study: <https://w3id.org/sciencelive/np/RAlhXM-tU0wG-SSReXZLiyzQ1sJvEP7_FCqGoXIMmNADs>
  - FORRT Replication Outcome: <https://w3id.org/sciencelive/np/RAoW3q1q1Wyt5DXbFl2PI3woyhuYZuU8HYtJ3m0LyrP9M>
  - CiTO: <https://w3id.org/sciencelive/np/RAUuipi5julhQ_M3pYhg_8UW-m2XmqXBfoHPeRfkNYzoc>
- **Research Software nanopub.** Anchors the `v0.1.0` Zenodo concept DOI (`10.5281/zenodo.20082933`) to chain A's PICO question and back-links to all three chain Outcomes. Published 2026-05-08.
  - <https://w3id.org/sciencelive/np/RA0TakYbwjs9vdc2AXyKxaCj54u5vr8zIrNhebEEskWRc>
- **Research Synthesis nanopub.** Names the cross-cutting property all three chains exercise — that the HEALPix-NESTED substrate makes sphere-aware ML latitude-invariant, discipline-transferable, and biodiversity-attribution-ready. Published 2026-05-08.
  - <https://w3id.org/sciencelive/np/RA6r8sefdZHemsSWVZVo7nXdvydNqCm-VfrQpnmmRBrfA>

## License

Code: **MIT** (see `LICENSE`).
Generated figures and notebook prose: **CC-BY 4.0**.
