# FORRT chain drafts — spherical-ml-biodiversity

This document is the field-by-field draft of the **20 nanopublications** that
formalise the spherical-ml-biodiversity findings as FORRT chains on Science
Live.

The structure: three parallel atomic FORRT chains (one per notebook), one
Research Software nanopub for the repository, and one Research Synthesis
nanopub binding the three chains together.

| Chain | Anchor template | Notebook | Atomic AIDA | Claim type |
|---|---|---|---|---|
| **A** — Within-discipline latitude invariance | PICO (Effectiveness) | 04 | flat collapses 1.000 → 0.500 across latitudes; sphere holds at 1.000 | model performance |
| **B** — Real-data Ningaloo Niño biodiversity exposure | Quote-with-comment from Wernberg et al. 2016 (`10.1126/science.aad8745`) | 05 | 719 of 765 (94.0 percent) of marine GBIF records on MHW-exposed cells | descriptive pattern |
| **C** — Cross-discipline transfer | PICO (Effectiveness) | 06 | sphere transfers cosmology → climate at 1.000; flat drops to 0.845 | model performance |

Each chain has 6 nanopubs in order: anchor → AIDA → FORRT Claim → FORRT
Replication Study → FORRT Replication Outcome → CiTO Citation. 18 nanopubs
total in the chains; plus 1 Research Software nanopub for the repo and 1
Research Synthesis nanopub on top.

## Publication order

1. Tag `v0.1.0` release on GitHub → Zenodo mints the concept DOI
2. **Chain A** — PICO → AIDA → Claim → Study → Outcome → CiTO  *(6 nanopubs)*
3. **Chain C** — PICO → AIDA → Claim → Study → Outcome → CiTO  *(6 nanopubs)*
4. **Chain B** — Quote → AIDA → Claim → Study → Outcome → CiTO  *(6 nanopubs;
   B last because conceptually it is the real-data anchor that ties the
   synthesis together)*
5. **Research Software nanopub** *(Related Publications back-link to all
   three Outcomes)*
6. **Research Synthesis nanopub** *(supporting sources = three Outcomes +
   Research Software)*

URIs of previously-published nanopubs in the chain are referenced as
`{X1-uri}`, `{X2-uri}`, etc. — replace each placeholder with the actual
`https://w3id.org/sciencelive/np/RA…` URI returned by Science Live as you
publish.

---

# Chain A — Within-discipline latitude invariance (notebook 04)

PICO-rooted. Question Type: Effectiveness. Claim type: model performance.

## A1 / 6 — PICO Research Question

| Field | Value |
|---|---|
| **Short ID** | `spherical-ml-latitude-invariance-2026` |
| **Research Question Title** | Does sphere-aware ML on HEALPix recover detection accuracy at high latitudes that lat-lon flat ML loses? |
| **Complete Research Question** | For globally-distributed marine-heatwave detection on synthetic global SST samples on a HEALPix-NESTED grid (Population), does a sphere-harmonic band-pass matched filter applied via `aₗₘ × fₗ × bₗ` (Intervention) recover the high-latitude detection accuracy lost by an equivalent lat-lon flat matched filter trained on equator-shape features (Comparator), measured by per-test-latitude-band detection accuracy and F1 score (Outcome)? |
| **Question Type** | Effectiveness |
| **Population (P)** | Synthetic global daily sea-surface-temperature samples on a 1° lat-lon raster with marine-heatwave (MHW) events injected as 10° angular-radius spherical caps with +4 °C anomaly placed at uniform-random longitude and a chosen latitude band. 600 training samples (300 with MHW at \|lat\| ≤ 20°, 300 without) and 200 samples per test latitude band (100 with MHW, 100 without) at four bands centred at 10°, 35°, 55°, 75°. |
| **Intervention (I)** | A sphere-harmonic band-pass matched filter on HEALPix-NESTED nside=64 — aggregate the lat-lon SST to HEALPix, mean-subtract, transform via `healpy.map2alm` at lmax=64, multiply by `fₗ × bₗ` where `fₗ = 0` for `ℓ < 5` (high-pass to remove the cosine-of-latitude SST baseline) and `bₗ` is a Gaussian beam with `FWHM = 2 × 10° = 20°` (matched to the cap diameter), inverse SHT via `healpy.alm2map`, return `(max, mean, std)` of the response field, fed to a logistic-regression classifier head. |
| **Comparison (C)** | A lat-lon flat matched filter on the same data — cross-correlate the SST anomaly raster with an equator-shape 10° spherical-cap template via `scipy.signal.correlate2d` (mode="same", boundary="wrap"), return `(max, mean, std)` of the response, fed to the same logistic-regression head. |
| **Outcome (O)** | Detection accuracy and F1 score by test latitude band at four bands 0–20° / 30–40° / 50–60° / 70–80°, plus training accuracy on the in-distribution low-latitude set. |

## A2 / 6 — AIDA sentence

| Field | Value |
|---|---|
| **AIDA sentence** | On the HEALPix-NESTED substrate, a sphere-harmonic band-pass matched filter trained on low-latitude marine-heatwave events maintains 1.000 detection accuracy at the 70-80° latitude band while an equivalent lat-lon flat matched filter trained identically collapses to 0.500 detection accuracy at the same band. |
| **Select related topics/tags** | machine learning |
| **Relates to this nanopublication** | `{A1-uri}` |
| **Supported by datasets** | (skip — optional; data is synthetic, generated in-notebook) |
| **Supported by other publications** | `https://doi.org/10.1086/427976`; `https://arxiv.org/abs/1801.10130`; `https://arxiv.org/abs/2012.15000`; `https://doi.org/10.1029/2023MS004021` |

## A3 / 6 — FORRT Claim

| Field | Value |
|---|---|
| **Short URI suffix** | `spherical-ml-latitude-invariance-claim-2026` |
| **Label of the claim** | Sphere-harmonic matched filter on HEALPix maintains detection accuracy at high latitudes; lat-lon flat matched filter does not |
| **Search for an AIDA sentence** | `{A2-uri}` |
| **Type of FORRT claim** | model performance |
| **Source URI** | (skip — first-party question-rooted chain, no original paper) |

## A4 / 6 — FORRT Replication Study

| Field | Value |
|---|---|
| **Short URI suffix** | `spherical-ml-latitude-invariance-study-2026` |
| **Label/name of replication study** | Sphere vs flat matched-filter pipelines for marine-heatwave detection on synthetic global SST on HEALPix-NESTED |
| **Study type** | Replication Study — replication with different methodology or conditions |
| **Search for a FORRT claim** | `{A3-uri}` |
| **Describe what part of the claim is reproduced/replicated** | The within-discipline latitude-invariance property of the sphere-harmonic band-pass matched filter on synthetic global SST samples with marine-heatwave events injected at varying latitude bands, against an equivalent lat-lon flat matched filter trained on equator-shape features. The two pipelines share identical `(max, mean, std)` feature structure and identical logistic-regression heads; the only thing that differs is the substrate the convolution lives on. |
| **Describe how the claim is reproduced/replicated** | 600 training samples (300 with MHW at \|lat\| ≤ 20°, 300 without) plus 200 samples per test latitude band (100 with MHW, 100 without) at four bands centred 10°/35°/55°/75°. Sphere features via `healpy.map2alm` / `almxfl` / `alm2map` at nside=64 lmax=64 with `fₗ = 0` for `ℓ < 5` and Gaussian beam `bₗ` FWHM=20°. Flat features via `scipy.signal.correlate2d` with mode="same" boundary="wrap" against an equator-cap template. Same `sklearn.linear_model.LogisticRegression(max_iter=2000, C=1.0)` head on top of each feature triple. Reproducible end-to-end via the repository's `environment.yml` + `Snakefile`; CI executes the notebook on every push. |
| **Describe any deviations from original methodology** | First end-to-end demonstration in this repository of the substrate-dependence with this matched-filter pair; no prior single-paper implementation. The within-discipline failure mode is documented qualitatively in DeepSphere and DLWP-HEALPix follow-ups but not quantified at this granularity with this minimal feature pair. |
| **Search keywords (Wikidata)** | spherical machine learning, rotation equivariance, HEALPix, marine heatwave, sea surface temperature, sphere-harmonic transform, matched filter |
| **Search discipline (Wikidata)** | machine learning, oceanography, Earth observation |

## A5 / 6 — FORRT Replication Outcome

| Field | Value |
|---|---|
| **Short URI suffix** | `spherical-ml-latitude-invariance-outcome-2026` |
| **Plain-text label** | Sphere matched filter holds 1.000 accuracy at every test latitude band; flat collapses to 0.500 at 70–80° |
| **Search for a FORRT replication study** | `{A4-uri}` |
| **Repository URL** | `https://github.com/annefou/spherical-ml-biodiversity` |
| **Completion date** | 2026-05-06 |
| **Validation status** | Validated |
| **Confidence level** | High |
| **Describe the overall conclusion about the original claim** | The sphere-harmonic band-pass matched filter on the HEALPix-NESTED substrate maintains 1.000 detection accuracy and 1.000 F1 across all four test latitude bands while the equivalent lat-lon flat matched filter trained on the same data collapses progressively as the test latitude moves poleward, dropping to 0.500 accuracy and 0.000 F1 (chance, all-negative classification) at 70-80°. Both pipelines use identical feature structure and identical logistic-regression heads; the only thing that differs is the substrate. The sphere-harmonic operation `aₗₘ → aₗₘ × fₗ × bₗ` is exactly rotation-equivariant on the sphere by construction, so the response peak for the same physical 10° spherical cap is the same value regardless of where on the sphere it sits; the lat-lon flat operation is only translation-equivariant in pixel space, so the same physical spherical cap renders to a 3× longitudinally-stretched shape at 70°N and the matched-filter response degrades. |
| **Describe the evidence that supports your conclusion** | Numerical results from notebook 04 — flat lat-lon matched filter at test bands {0–20°: accuracy 1.000, F1 1.000; 30–40°: 1.000, 1.000; 50–60°: 0.915, 0.907; 70–80°: 0.500, 0.000} versus sphere-harmonic band-pass matched filter at the same bands {0–20°: 1.000, 1.000; 30–40°: 1.000, 1.000; 50–60°: 1.000, 1.000; 70–80°: 1.000, 1.000}. Training accuracy 1.000 for both pipelines on the in-distribution \|lat\| ≤ 20° set. 200 test samples per band (100 positive + 100 negative). |
| **Describe what limits the conclusions of the study** | (i) Synthetic SST with controlled feature physics (10° fixed-radius cap, +4 °C fixed-amplitude anomaly) — this isolates the substrate effect from the model class but does not characterise real-data noise patterns. (ii) The cosine-of-latitude SST baseline is an idealised tropical-to-polar gradient; real SST climatologies have basin-scale and seasonal structure the high-pass `fₗ` filter would also remove if extended to lower multipoles. (iii) The `(max, mean, std)` feature triple is the minimal possible matched-filter readout; richer learned representations from DeepSphere graph CNNs or `foscat` scattering networks would deliver substantially higher feature dimensionality on real data. (iv) Four test latitude bands sample the failure mode at four points; intermediate latitudes were not separately tested. |

## A6 / 6 — Citation with CiTO

| Field | Value |
|---|---|
| **Identifier for the citing creative work** | `{A5-uri}` |
| Citation 1 — type | `confirms` |
| Citation 1 — DOI/URL | `https://arxiv.org/abs/1801.10130` (Cohen et al. 2018, Spherical CNNs) |
| Citation 2 — type | `confirms` |
| Citation 2 — DOI/URL | `https://arxiv.org/abs/2012.15000` (Defferrard et al. 2020, DeepSphere) |
| Citation 3 — type | `extends` |
| Citation 3 — DOI/URL | `https://doi.org/10.1086/427976` (Górski et al. 2005, HEALPix) |
| Citation 4 — type | `extends` |
| Citation 4 — DOI/URL | `https://doi.org/10.1029/2023MS004021` (Karlbauer et al. 2024, DLWP-HEALPix) |

---

# Chain B — Real-data biodiversity exposure for the 2011 Ningaloo Niño (notebook 05)

Paper-rooted off Wernberg et al. 2016 (*Science* 353:169–172). Claim type: descriptive pattern.

## B1 / 6 — Quote-with-comment

| Field | Value |
|---|---|
| **Cited DOI** | `10.1126/science.aad8745` |
| **Mode** | Quote whole text (less than 500 characters) |
| **Quoted Text** | An extreme marine heat wave in 2011 forced a reorganization of the temperate Australian kelp forest community to one dominated by tropical and subtropical species (i.e., regime shift), with degradation in ecosystem services. |
| **Comment** | We extend Wernberg et al.'s documented regime-shift event into a substrate-aware multi-taxon biodiversity-exposure overlay. NOAA OISST v2.1 daily SST for 2011 is aggregated onto a HEALPix-NESTED nside=128 substrate at WGS84 ellipsoidal projection via healpix-geo, MHW-detected with a Hobday-style rule (anomaly above 2008–2010 daily climatology + 1.5 °C threshold + ≥5-day persistence), and overlaid with GBIF marine biodiversity occurrences from the same region and year, restricted to taxonKey-marine-only taxa via OISST-derived ocean masking. We replicate the spatial-temporal correspondence at the multi-taxon level: 719 of 765 marine occurrences (94.0 percent) from 113 unique species sat on cells flagged as MHW-exposed during the same event window. |

## B2 / 6 — AIDA sentence

| Field | Value |
|---|---|
| **AIDA sentence** | On the HEALPix-NESTED substrate, 719 of 765 (94.0 percent) marine biodiversity occurrences from GBIF in the Western Australian region during 2011 sat on cells that experienced marine-heatwave conditions during the documented Ningaloo Niño event of February-March 2011. |
| **Select related topics/tags** | (pick one — likely "marine biology" or "climate change" from the dropdown — open it and choose closest match) |
| **Relates to this nanopublication** | `{B1-uri}` |
| **Supported by datasets** | `https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html`; `https://www.gbif.org/` |
| **Supported by other publications** | `https://doi.org/10.1126/science.aad8745`; `https://doi.org/10.1016/j.pocean.2015.12.014` |

## B3 / 6 — FORRT Claim

| Field | Value |
|---|---|
| **Short URI suffix** | `ningaloo-2011-biodiversity-exposure-claim-2026` |
| **Label of the claim** | 94 percent of marine GBIF records in the Western Australian region during 2011 sat on HEALPix cells that experienced MHW conditions during the documented Ningaloo Niño event |
| **Search for an AIDA sentence** | `{B2-uri}` |
| **Type of FORRT claim** | descriptive pattern |
| **Source URI** | `https://doi.org/10.1126/science.aad8745` |

## B4 / 6 — FORRT Replication Study

| Field | Value |
|---|---|
| **Short URI suffix** | `ningaloo-2011-biodiversity-exposure-study-2026` |
| **Label/name of replication study** | Substrate-aware multi-taxon biodiversity-exposure overlay for the 2011 Western Australian Ningaloo Niño event on HEALPix-NESTED nside=128 |
| **Study type** | Replication Study — replication with different methodology or conditions |
| **Search for a FORRT claim** | `{B3-uri}` |
| **Describe what part of the claim is reproduced/replicated** | Replication of Wernberg et al. 2016's spatial-temporal correspondence between the 2011 Ningaloo Niño marine heatwave and biodiversity impacts in the Western Australian temperate-reef community, extended from kelp-only diver-transect surveys to all marine taxa available in GBIF for the same region and year, on a shared HEALPix-NESTED substrate. |
| **Describe how the claim is reproduced/replicated** | Region: lat ∈ [-35°, -20°], lon ∈ [108°, 118°] (Western Australian coast and adjacent eastern Indian Ocean). Period: Jan–Apr 2011. SST: NOAA OISST v2.1 daily SST via PSL OPeNDAP. Climatology: per-day-of-year mean from 2008–2010 yearly OISST files (3-year baseline; the canonical Hobday 2016 30-year 1991–2020 baseline is unreliable via the precomputed PSL OPeNDAP file at full-region slicing for this analysis box). Anomaly: 2011 daily SST minus 2008–2010 per-DOY mean. MHW detection: cell flagged as MHW-day on day d if anomaly > 1.5 °C for at least 5 consecutive days including d (Hobday-style minimum-duration rule with simplified threshold). HEALPix aggregation: nside=128 NESTED ordering, area-weighted mean over OISST sea-cells only (OISST land mask via numpy.isfinite at the product's native sea/land boundary). GBIF: pygbif.occurrences.search by taxonKey for class Elasmobranchii (121), class Cephalopoda (136), phylum Cnidaria (43), phylum Echinodermata (50), phylum Porifera (105). Phylum-level Mollusca taxonKey (52) excluded because it includes terrestrial gastropods and freshwater bivalves. Records dropped if their HEALPix cell has no OISST sea coverage. Visualisation via healpix-plot with WGS84 ellipsoidal HEALPix from healpix-geo. |
| **Describe any deviations from original methodology** | Wernberg et al. 2016 sampled the kelp-forest community via diver-transect surveys at fixed sites. This study uses GBIF marine occurrence records as the data source (different scope — broader marine taxa, not just kelp; different sampling — opportunistic occurrence vs structured transect), and a shared HEALPix-NESTED substrate for spatial overlap with the MHW footprint (different aggregation method). The MHW threshold uses 2008–2010 daily climatology with 1.5 °C anomaly + 5-day persistence (a simplified Hobday-style rule); the canonical Hobday 2016 90th-percentile-per-day-of-year over 30 years is approximated. |
| **Search keywords (Wikidata)** | marine heatwave, Ningaloo Niño, sea surface temperature, marine biodiversity, GBIF, HEALPix, Western Australia, kelp, Elasmobranchii, Cephalopoda, Anthozoa |
| **Search discipline (Wikidata)** | marine biology, oceanography, biodiversity science, climate change adaptation |

## B5 / 6 — FORRT Replication Outcome

| Field | Value |
|---|---|
| **Short URI suffix** | `ningaloo-2011-biodiversity-exposure-outcome-2026` |
| **Plain-text label** | 719 of 765 marine GBIF records (94.0 percent) on MHW-exposed HEALPix cells; 113 unique species exposed |
| **Search for a FORRT replication study** | `{B4-uri}` |
| **Repository URL** | `https://github.com/annefou/spherical-ml-biodiversity` |
| **Completion date** | 2026-05-06 |
| **Validation status** | Validated |
| **Confidence level** | High |
| **Describe the overall conclusion about the original claim** | Wernberg et al. 2016's documented Ningaloo Niño regime-shift event has a much broader marine-biodiversity exposure footprint than the kelp-forest community alone. On the HEALPix-NESTED nside=128 substrate at WGS84 ellipsoidal projection, 719 of 765 marine GBIF records (94.0 percent) from the WA region during 2011 sat on cells that experienced MHW conditions during the same Jan–Apr 2011 event window, spanning 113 unique species across Elasmobranchii (sharks and rays), Cephalopoda (octopus and squid), Anthozoa (corals), Hydrozoa (jellies), Demospongiae (sponges), Echinoidea (urchins), Ophiuroidea (brittle stars), Asteroidea (sea stars), Bivalvia (clams), Scyphozoa (jellyfish). The shared HEALPix substrate enables direct spatial-temporal overlap of climate-event fields and biodiversity occurrence data on a single equal-area grid without re-projection — the same substrate used by climate models and sphere-aware ML. This is the operational case for HEALPix as the common DGGS for biodiversity-impact attribution: the climate side and the biodiversity side meet on one substrate. |
| **Describe the evidence that supports your conclusion** | The 2011 SST anomaly relative to the 2008–2010 per-DOY mean climatology peaked at +7.20 °C globally over the WA region in Jan–Apr 2011 with mean Feb–Mar anomaly of +2.36 °C — consistent with the documented Ningaloo Niño event peak. Maximum number of MHW-days at a single 0.25° lat-lon cell over Jan–Apr 2011 was 109 of 120 days. After OISST-substrate-aware ocean masking, 765 marine GBIF records were retained out of 3134 phylum/class-restricted hits; the 2369 dropped records were on land or outside the OISST sea cells (terrestrial molluscs from a too-loose phylum filter, GBIF coordinate fuzzing, and records with museum/institution-coded coordinates). Of the 765 ocean records, 719 (94.0 percent) sat on HEALPix cells flagged as MHW-exposed by the Hobday-style detection. 113 unique species on MHW-exposed cells. |
| **Describe what limits the conclusions of the study** | (i) MHW threshold simplification — fixed +1.5 °C anomaly relative to a 3-year (2008–2010) per-DOY climatology approximates the canonical Hobday 2016 30-year 1991–2020 90th-percentile threshold; the qualitative spatial footprint matches the documented Ningaloo Niño event but the per-cell MHW-day count is approximate. (ii) Spatial-overlap statistic is exposure-only (records on MHW cells), not causal attribution of biodiversity change to MHW conditions; the linkage to Wernberg et al. 2016 kelp regime shift is via spatial-temporal coincidence with the documented event, not via direct measurement of population change. (iii) GBIF taxonomic filter restricted to phylum/class taxonKeys for Elasmobranchii, Cephalopoda, Cnidaria, Echinodermata, Porifera; phylum-level Mollusca excluded because it includes terrestrial gastropods and freshwater bivalves. Some marine Mollusca records other than Cephalopoda are therefore not included, which under-counts marine biodiversity exposure modestly. (iv) GBIF coordinate precision is variable; the OISST sea-mask catches the worst inland-coordinate cases but not all coordinate-precision issues. (v) Period restricted to Jan–Apr 2011; the actual Ningaloo Niño peak intensity was Feb–Mar 2011 but warm anomalies persisted into mid-2011. |

## B6 / 6 — Citation with CiTO

| Field | Value |
|---|---|
| **Identifier for the citing creative work** | `{B5-uri}` |
| Citation 1 — type | `confirms` |
| Citation 1 — DOI/URL | `https://doi.org/10.1126/science.aad8745` (Wernberg et al. 2016, kelp regime shift) |
| Citation 2 — type | `usesMethodIn` |
| Citation 2 — DOI/URL | `https://doi.org/10.1016/j.pocean.2015.12.014` (Hobday et al. 2016, MHW definition) |
| Citation 3 — type | `obtainsBackgroundFrom` |
| Citation 3 — DOI/URL | `https://doi.org/10.1086/427976` (Górski et al. 2005, HEALPix) |
| Citation 4 — type | `obtainsBackgroundFrom` |
| Citation 4 — DOI/URL | `https://doi.org/10.5281/zenodo.19848749` (dggs-biodiversity-bias concept DOI) |

---

# Chain C — Cross-discipline transfer (notebook 06)

PICO-rooted. Question Type: Effectiveness. Claim type: model performance.

## C1 / 6 — PICO Research Question

| Field | Value |
|---|---|
| **Short ID** | `spherical-ml-cross-discipline-transfer-2026` |
| **Research Question Title** | Does sphere-aware ML on HEALPix transfer cleanly across discipline pairs without retraining? |
| **Complete Research Question** | For two synthetic discipline regimes on the HEALPix-NESTED substrate sharing the same physical feature physics — Domain A cosmology-like with `Cl ∝ (l+1)^-1.5` and features at uniformly-random sphere locations, Domain B climate-like with `Cl ∝ (l+1)^-3` plus a cosine-of-latitude SST baseline and features confined to high latitudes (Population) — does a sphere-harmonic band-pass matched filter trained on Domain A (Intervention) transfer to Domain B without retraining at higher accuracy than an equivalent lat-lon flat matched filter trained identically on Domain A and applied to Domain B (Comparator), measured by cross-domain transfer accuracy and in-domain B-trained upper-bound accuracy (Outcome)? |
| **Question Type** | Effectiveness |
| **Population (P)** | Two synthetic global random-field regimes on HEALPix-NESTED nside=64. Domain A is cosmology-like (Gaussian random field with steep `Cl ∝ (l+1)^-1.5`, optional 12° angular-radius spherical-cap feature with +6.0 amplitude at uniformly-random sphere location). Domain B is climate-like (smoother `Cl ∝ (l+1)^-3`, additive cosine-of-latitude SST baseline `1.5 cos²(lat)`, optional 12° feature with +6.0 amplitude restricted to \|lat\| ≥ 50°). 200 + 200 train + 100 + 100 test samples per class per domain. |
| **Intervention (I)** | Sphere-harmonic band-pass matched filter on HEALPix — `aₗₘ → aₗₘ · fₗ · bₗ` with `fₗ = 0` for `ℓ < 5` and Gaussian beam `bₗ` with `FWHM = 24°` matched to the 12°-radius cap diameter, returning `(max, mean, std)` of the inverse-SHT response field, fed to a logistic-regression classifier head. |
| **Comparison (C)** | Lat-lon flat matched filter — cross-correlate the lat-lon raster with an equator-shape 12° spherical-cap template via `scipy.signal.correlate2d`, returning `(max, mean, std)` of the response, fed to the same logistic-regression head. |
| **Outcome (O)** | In-domain accuracy on Domain A (sanity check), cross-domain transfer accuracy on Domain B (the headline), and in-domain Domain-B-trained-and-tested accuracy as the upper bound. |

## C2 / 6 — AIDA sentence

| Field | Value |
|---|---|
| **AIDA sentence** | On the HEALPix-NESTED substrate, a sphere-harmonic band-pass matched filter trained on a cosmology-like Domain A with features at uniformly-random sphere locations classifies a climate-like Domain B with features confined to high latitudes at 1.000 accuracy without retraining, while an equivalent lat-lon flat matched filter trained identically drops from 1.000 in-domain to 0.845 on the same cross-domain transfer. |
| **Select related topics/tags** | machine learning |
| **Relates to this nanopublication** | `{C1-uri}` |
| **Supported by datasets** | (skip — optional; data is synthetic, generated in-notebook) |
| **Supported by other publications** | `https://doi.org/10.1086/427976`; `https://arxiv.org/abs/1801.10130`; `https://arxiv.org/abs/2012.15000`; `https://doi.org/10.1029/2023MS004021` |

## C3 / 6 — FORRT Claim

| Field | Value |
|---|---|
| **Short URI suffix** | `spherical-ml-cross-discipline-transfer-claim-2026` |
| **Label of the claim** | Sphere-harmonic matched filter on HEALPix transfers cross-discipline at 1.000 without retraining; lat-lon flat matched filter drops to 0.845 on the same transfer |
| **Search for an AIDA sentence** | `{C2-uri}` |
| **Type of FORRT claim** | model performance |
| **Source URI** | (skip — first-party question-rooted chain, no original paper) |

## C4 / 6 — FORRT Replication Study

| Field | Value |
|---|---|
| **Short URI suffix** | `spherical-ml-cross-discipline-transfer-study-2026` |
| **Label/name of replication study** | Cross-discipline transfer of sphere-harmonic vs lat-lon-flat matched filters between cosmology-like and climate-like synthetic domains on HEALPix-NESTED |
| **Study type** | Replication Study — replication with different methodology or conditions |
| **Search for a FORRT claim** | `{C3-uri}` |
| **Describe what part of the claim is reproduced/replicated** | Test whether the substrate-rotation-equivariance property demonstrated within-discipline in chain A also delivers cross-discipline transfer between domains with different background spectra and different feature-location distributions, on the same HEALPix-NESTED substrate. |
| **Describe how the claim is reproduced/replicated** | HEALPix nside=64, lmax=64. Domain A samples synthesised via `healpy.synfast` with `Cl ∝ (l+1)^-1.5` power spectrum. Domain B samples synthesised via `healpy.synfast` with `Cl ∝ (l+1)^-3` plus an additive `1.5 cos²(lat)` baseline representing the SST equator-pole gradient. With-feature samples have a 12° angular-radius spherical cap with +6.0 amplitude added at a uniformly-random location (Domain A) or a uniformly-cos-restricted high-latitude location (Domain B, \|lat\| ≥ 50°). Both pipelines extract the same `(max, mean, std)` feature triple — sphere via `aₗₘ × fₗ × bₗ` with `fₗ = 0` for `ℓ < 5` and Gaussian beam `bₗ` FWHM=24°, flat via `scipy.signal.correlate2d` against an equator-cap template — and feed identical `sklearn.linear_model.LogisticRegression(max_iter=4000, C=1.0)` heads. Train on Domain A only, evaluate on (i) Domain A held-out test set, (ii) Domain B test set as cross-domain transfer, (iii) Domain B trained classifier on Domain B test set as in-domain B upper bound. |
| **Describe any deviations from original methodology** | First in-repository demonstration of cross-discipline transfer with this matched-filter pair; no prior single-paper precedent for this specific experiment. The two synthetic domain regimes are constructed to share feature physics across different background spectra so the substrate effect is isolated from the model class. |
| **Search keywords (Wikidata)** | spherical machine learning, geometric deep learning, rotation equivariance, HEALPix, transfer learning, cross-domain transfer, cosmology, climate model |
| **Search discipline (Wikidata)** | machine learning, cosmology, climate science |

## C5 / 6 — FORRT Replication Outcome

| Field | Value |
|---|---|
| **Short URI suffix** | `spherical-ml-cross-discipline-transfer-outcome-2026` |
| **Plain-text label** | Sphere matched filter transfers A→B at 1.000 without retraining; flat drops from 1.000 to 0.845 on the same transfer |
| **Search for a FORRT replication study** | `{C4-uri}` |
| **Repository URL** | `https://github.com/annefou/spherical-ml-biodiversity` |
| **Completion date** | 2026-05-06 |
| **Validation status** | Validated |
| **Confidence level** | High |
| **Describe the overall conclusion about the original claim** | The substrate-rotation-equivariance property the within-discipline test (chain A) demonstrated also delivers cross-discipline transfer at the same magnitude on the HEALPix-NESTED substrate. The sphere-harmonic band-pass matched filter trained on Domain A (cosmology-like, features at uniformly-random sphere locations, steep background spectrum) classifies Domain B (climate-like, features confined to high latitudes, smoother background spectrum + cosine-of-latitude baseline) at 1.000 accuracy without retraining. The equivalent lat-lon flat matched filter trained identically drops from 1.000 in-domain on Domain A to 0.845 on the cross-domain transfer to Domain B because the equator-shape template under-responds to polar-stretched features in lat-lon space. Both pipelines reach 1.000 / 0.995 in-domain on Domain B when trained directly on Domain B, so the test is fair — the asymmetry shows up only in the cross-discipline transfer column. This is the operational claim that investments in sphere-aware models from one discipline (cosmology DeepSphere, foscat, healpy) carry over to other disciplines (climate, biodiversity, EO) on the shared HEALPix substrate. |
| **Describe the evidence that supports your conclusion** | Numerical results from notebook 06 — sphere {A→A 0.990; A→B transfer 1.000; B→B upper-bound 1.000} versus flat {A→A 1.000; A→B transfer 0.845; B→B upper-bound 0.995}. 200 train + 100 test samples per class per domain. Sphere band-pass matched filter via healpy at nside=64 lmax=64 with `fₗ = 0` for `ℓ < 5` and Gaussian beam FWHM=24°. Flat matched filter via `scipy.signal.correlate2d` against an equator-cap template with mode="same" boundary="wrap". Identical `(max, mean, std)` features and identical logistic-regression classifier heads. Reproducible end-to-end via the repository's `environment.yml` + `Snakefile`. |
| **Describe what limits the conclusions of the study** | (i) Synthetic domain regimes constructed to share feature physics across different background spectra; true cross-discipline transfer from real cosmology data (e.g., Planck CMB on HEALPix) to real climate data (e.g., DLWP-HEALPix forecasts on HEALPix) would require integrating with `foscat` scattering networks or a DeepSphere graph CNN as future work. (ii) The substrate effect is isolated from the model class via the minimal `(max, mean, std)` feature triple; richer learned representations would deliver substantially different absolute accuracy numbers but the substrate-dependence is the geometric mechanism the experiment captures. (iii) The latitude restriction in Domain B (features at \|lat\| ≥ 50°) is the climate-motivated regime where lat-lon projection distortion bites hardest; the cross-discipline transfer test would yield different numbers for differently-distributed feature regimes. (iv) Two domains tested; the transfer-between-pairs claim generalises naturally to N-way transfer but was not separately tested with three or more domains. |

## C6 / 6 — Citation with CiTO

| Field | Value |
|---|---|
| **Identifier for the citing creative work** | `{C5-uri}` |
| Citation 1 — type | `confirms` |
| Citation 1 — DOI/URL | `https://arxiv.org/abs/1801.10130` (Cohen et al. 2018, Spherical CNNs) |
| Citation 2 — type | `confirms` |
| Citation 2 — DOI/URL | `https://arxiv.org/abs/2012.15000` (Defferrard et al. 2020, DeepSphere) |
| Citation 3 — type | `extends` |
| Citation 3 — DOI/URL | `https://doi.org/10.1086/427976` (Górski et al. 2005, HEALPix) |
| Citation 4 — type | `extends` |
| Citation 4 — DOI/URL | `https://doi.org/10.1029/2023MS004021` (Karlbauer et al. 2024, DLWP-HEALPix) |

---

# Research Software nanopub (after v0.1.0 release)

| Field | Value |
|---|---|
| **URI of published software** | `https://doi.org/10.5281/zenodo.{concept-DOI}` *(insert after Zenodo mints the concept DOI on v0.1.0 release)* |
| **Software Title** | spherical-ml-biodiversity — pedagogy and worked head-to-head of sphere-aware vs flat ML on HEALPix, with a real Ningaloo-Niño-2011 biodiversity overlay and a cross-discipline transfer demonstration |
| **Repository URL** | `https://github.com/annefou/spherical-ml-biodiversity` |
| **Research Project** | `{A1-uri}` *(PICO question URI from chain A — chain-anchor convention for question-rooted chains)* |
| **License** | `https://spdx.org/licenses/MIT.html` |
| **Related Datasets** | `https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html` (NOAA OISST v2.1 daily SST); `https://www.gbif.org/` (Global Biodiversity Information Facility) |
| **Related Publications** | `{A5-uri}` (chain A Outcome); `{B5-uri}` (chain B Outcome); `{C5-uri}` (chain C Outcome); `https://doi.org/10.1086/427976` (Górski et al. 2005, HEALPix); `https://arxiv.org/abs/1801.10130` (Cohen et al. 2018, Spherical CNNs); `https://arxiv.org/abs/2012.15000` (Defferrard et al. 2020, DeepSphere); `https://doi.org/10.1029/2023MS004021` (Karlbauer et al. 2024, DLWP-HEALPix); `https://doi.org/10.1016/j.pocean.2015.12.014` (Hobday et al. 2016, MHW definition); `https://doi.org/10.1126/science.aad8745` (Wernberg et al. 2016, kelp regime shift) |

---

# Research Synthesis nanopub (after Research Software)

| Field | Value |
|---|---|
| **Short URI suffix** | `spherical-ml-substrate-synthesis-2026` |
| **Label of the synthesis** | The HEALPix-NESTED substrate makes sphere-aware ML latitude-invariant, discipline-transferable, and biodiversity-attribution-ready |
| **Conclusion of the synthesis** | Three independent tests on the HEALPix-NESTED substrate jointly establish that sphere-aware operators recover detection accuracy lat-lon flat operators lose, transfer across discipline pairs without retraining, and integrate with marine biodiversity occurrence data on a single shared substrate — without re-projection at any step. The within-discipline test (chain A, notebook 04) shows the lat-lon-flat matched filter collapsing from 1.000 to 0.500 chance at 70–80° latitude while the sphere-harmonic band-pass matched filter holds at 1.000 across all four test bands. The cross-discipline test (chain C, notebook 06) shows the same sphere-aware pipeline transferring at 1.000 from a cosmology-like training domain to a climate-like test domain without retraining, while the flat baseline drops to 0.845 on the same transfer. The real-data test (chain B, notebook 05) shows that when the climate-event field and the biodiversity occurrence field meet on the same HEALPix substrate, 94.0 percent of 765 marine GBIF records during the documented 2011 Ningaloo Niño event sat on cells that experienced marine-heatwave conditions in the same window Wernberg et al. 2016 documented the kelp regime shift in. The shared property — that sphere-harmonic convolution is exactly rotation-equivariant on the sphere, while lat-lon convolution is only translation-equivariant in pixel space — is what makes the substrate the right common DGGS for Copernicus EO, Destination Earth climate models, and biodiversity-impact attribution to interoperate. |
| **Recommendations** | (1) When training ML detection or classification models on globally-distributed Earth-observation, climate, or biodiversity data, render features on a HEALPix-NESTED substrate (spherical for cosmology / synthetic experiments, WGS84-ellipsoidal via `healpix-geo` for geoscience) before applying convolutions; the substrate choice is what separates "works at the equator only" from "works at every latitude". (2) When integrating biodiversity occurrence data with Copernicus Marine SST, NOAA OISST, ERA5, or DestinE climate-model output for impact-attribution work, co-locate all four sources on a shared HEALPix-NESTED grid (NESTED ordering for hierarchical bit-shift refinement) at the resolution that matches the coarsest input, then perform overlap statistics there; do not perform per-source lat-lon aggregation followed by raster-level joins. (3) When evaluating sphere-aware versus flat ML pipelines, report accuracy at multiple test latitude bands and on at least one cross-discipline transfer regime; in-distribution-only metrics under-state the substrate effect. (4) Investments in sphere-aware models from one discipline (cosmology DeepSphere, foscat scattering networks, DLWP-HEALPix global weather forecasting) carry directly over to the other disciplines on the same HEALPix substrate; budget integration work as a feature-extractor port rather than a from-scratch retrain. |
| **Conditions under which the synthesis applies** | Scope: global ML detection / classification tasks where features can be expressed on a HEALPix-NESTED grid, including but not limited to sea-surface-temperature anomaly fields (NOAA OISST v2.1, Copernicus Marine SST, ERA5), tropospheric or stratospheric atmospheric fields (DLWP-HEALPix forecast outputs, ClimateNet), marine biodiversity occurrences (GBIF, OBIS) at coarsest-feature resolution, and synthetic Gaussian-random-field samples with compact features. Methods: sphere-harmonic transforms via `healpy.map2alm` / `alm2map`, sphere-harmonic-domain convolutions via `aₗₘ → aₗₘ · fₗ · bₗ`, equal-area cell aggregation via `numpy.bincount` on `healpy.ang2pix(..., nest=True)`. Domains: cosmology, climate, Earth observation, marine biodiversity, atmospheric science. The latitude-invariance and cross-discipline-transfer claims hold for any compact-feature detection task at fixed angular scale on the HEALPix substrate; the biodiversity-attribution claim is documented for the 2011 Ningaloo Niño event in the Western Australian region but the substrate-and-method combination generalises to any documented MHW or atmospheric-event case with available occurrence data. |
| **Limitations of the synthesis** | (1) The within-discipline and cross-discipline tests use synthetic data with controlled feature physics; the substrate-dependence is demonstrated via a minimal `(max, mean, std)` matched-filter feature triple to isolate the substrate effect from the model class, not via a deep learned representation. The numerical magnitudes (1.000 vs 0.500, 1.000 vs 0.845) reflect the geometric mechanism cleanly but are upper bounds on what richer learned representations like DeepSphere graph CNNs or `foscat` scattering networks would deliver on real data. (2) The real-data biodiversity-attribution case (chain B) uses a 3-year-baseline (2008–2010) simplification of the canonical Hobday et al. 2016 30-year 1991–2020 climatology and a fixed +1.5 °C anomaly threshold rather than the per-day-of-year 90th-percentile threshold; the qualitative spatial footprint matches the documented Ningaloo Niño but the per-cell MHW-day count is approximate. (3) The biodiversity-overlap statistic is exposure-only (records on MHW cells), not a causal attribution of biodiversity change to MHW conditions; the linkage to Wernberg et al. 2016 kelp regime shift is via spatial-temporal coincidence with the documented event. (4) The cross-discipline transfer test uses synthetic discipline regimes constructed to share feature physics across different background spectra; true cross-discipline transfer from real cosmology data (e.g., Planck CMB on HEALPix) to real climate data (e.g., DLWP-HEALPix forecasts on HEALPix) would require integrating with `foscat` or DeepSphere as future work. (5) All HEALPix work in this synthesis uses NESTED ordering throughout per the project-wide convention; results for RING-ordered HEALPix at the same resolution are expected to be equivalent but were not separately tested. |
| **Completion date** | 2026-05-06 |
| **Supporting sources** | `{A5-uri}`; `{B5-uri}`; `{C5-uri}`; `{RS-uri}` *(the Research Software nanopub URI from the previous step)* |
| **Search topics (Wikidata)** | spherical machine learning, geometric deep learning, rotation equivariance, HEALPix, marine heatwave, Ningaloo Niño, sea surface temperature, biodiversity, GBIF, Copernicus Earth observation, Destination Earth, GRID4EARTH, FAIR data |

---

# Cross-references

- Repo memory: `project_spherical_ml_biodiversity.md`
- Cross-discipline angle memory: `project_spherical_ml_cross_discipline.md`
- Plan summary memory: `project_spherical_ml_nanopubs_plan.md`
- Form structure reference: `reference_nanopub_form_fields.md` (for PICO, Quote-with-comment, AIDA, FORRT Claim, FORRT Replication Study, FORRT Replication Outcome, Research Software, Research Synthesis, Citation with CiTO field-by-field)
- Claim type vocabulary: `reference_forrt_claim_types.md`
- Chain start types (paper-rooted vs question-rooted): `feedback_chain_starts.md`
- Spell out percentage points: `feedback_spell_out_pp.md`
- Verify code before drafting Study/Outcome: `feedback_verify_code_before_drafting.md`
- Layered FORRT-vs-RS architecture: `feedback_forrt_vs_research_software_layers.md`
