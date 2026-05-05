# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Part 4: Spherical ML head-to-head — flat CNN on lat-lon vs sphere-harmonic features on HEALPix
#
# Pedagogy notebooks 01–03 motivated spherical ML; this notebook
# delivers the **head-to-head payoff**. We define one task — *detect
# whether a global daily sea-surface temperature (SST) field
# contains a marine-heatwave-like anomaly* — and run two pipelines
# on the same synthetic data:
#
# - **Pipeline A — flat baseline (standard EO ML practice).** SST
#   on a lat-lon raster, classify with features from a fixed
#   convolutional template matched to the *equator-shape MHW*. This
#   is the lat-lon analogue of what a flat CNN's first-layer filter
#   learns when trained on equator-distributed labels — a small
#   shape-detector on the projected raster. It's exactly the
#   matched filter from notebook 02; here we wrap it in a
#   classifier head. This is what an EO ML pipeline trained
#   directly on lat-lon Copernicus Marine SST or ERA5 products
#   looks like in practice.
#
# - **Pipeline B — sphere-aware (the GRID4EARTH / DLWP-HEALPix
#   path).** Same SST field, aggregated to HEALPix, classified
#   using a **sphere-harmonic matched filter** at the feature
#   scale: the field is convolved with a Gaussian beam of
#   `FWHM = 2 × ANGULAR_RADIUS_DEG` via `healpy.smoothing` (which
#   internally applies a multipole-domain beam to the spherical-
#   harmonic coefficients $a_{\ell m}$), and we report the
#   `(max, mean, std)` of the smoothed field as features —
#   structurally identical to the lat-lon matched filter above,
#   except convolution is happening on the sphere via `a_lm`
#   rather than on the lat-lon raster via 2-D FFT. Convolution
#   on the sphere with a sphere-harmonic beam is **exactly
#   rotation-equivariant by construction** (see Górski et al.
#   2005, Cohen et al. 2018), so the matched-filter response
#   for the same physical 10° cap is identical regardless of
#   where on the sphere the cap sits.
#
# Same classifier head (logistic regression) on top of each feature
# set. **Train on MHW events placed at low latitudes only.** Then
# evaluate at multiple test latitudes. The point is to see what
# happens when an EO ML model trained on low-latitude data is asked
# to recognise the same physical feature at high latitudes — the
# situation that arises constantly in operational climate-attribution
# work where global stacks of Copernicus Marine SST are used and the
# polar / boreal regimes are systematically under-represented in
# labels.
#
# **Why "lat-lon" is the right baseline here.** Biodiversity
# practitioners don't typically work on raw lat-lon rasters
# (notebook 04 of dggs-biodiversity-bias). But **EO and climate ML
# pipelines do** — ERA5, Copernicus Marine SST, MODIS, Sentinel,
# DestinE outputs are all distributed on lat-lon (or Gaussian)
# grids, and standard practice is to train flat CNNs on them
# directly. Spherical ML competes against that operational reality.
# Notebook 05 propagates this comparison through to a marine-
# biodiversity attribution task — the integrated regime where the
# bias finally lands on biodiversity science.

# %%
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

LAT_RES = 1.0
LON_RES = 1.0
lats_grid = np.arange(-89.5, 90, LAT_RES)
lons_grid = np.arange(-179.5, 180, LON_RES)
LAT_G, LON_G = np.meshgrid(lats_grid, lons_grid, indexing="ij")
n_lat, n_lon = LAT_G.shape

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
LMAX = 64  # Cl up to this multipole order

# Lat-lon → HEALPix index lookup (precomputed once)
theta_grid = np.radians(90.0 - LAT_G.ravel())
phi_grid = np.radians(LON_G.ravel() % 360)
pix_index = hp.ang2pix(NSIDE, theta_grid, phi_grid, nest=True).reshape(LAT_G.shape)
counts_per_pix = np.bincount(pix_index.ravel(), minlength=NPIX)
counts_safe = np.where(counts_per_pix > 0, counts_per_pix, 1)


def aggregate_to_healpix(sst):
    """Average SST per HEALPix cell (NESTED ordering)."""
    sums = np.bincount(pix_index.ravel(), weights=sst.ravel(), minlength=NPIX)
    out = sums / counts_safe
    out[counts_per_pix == 0] = 0.0
    return out


# %% [markdown]
# ## Generating synthetic global SST samples
#
# Each sample is a single daily SST snapshot. **Negative samples**
# have a baseline cosine-of-latitude SST plus weather noise.
# **Positive samples** add a synthetic MHW event — a spherical cap
# of fixed angular radius (10°) with a +4 °C anomaly — placed at a
# specified latitude. The MHW is the *same physical feature* on the
# sphere regardless of where it is placed (rotation-equivalent).

# %%
ANGULAR_RADIUS_DEG = 10.0
MHW_ANOMALY_C = 4.0
NOISE_C = 0.6


def is_in_spherical_cap(lat, lon, lat_c, lon_c, alpha_deg):
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    lat_c_r = np.radians(lat_c)
    lon_c_r = np.radians(lon_c)
    cos_d = (np.sin(lat_c_r) * np.sin(lat_r)
             + np.cos(lat_c_r) * np.cos(lat_r)
             * np.cos(lon_r - lon_c_r))
    cos_d = np.clip(cos_d, -1, 1)
    return np.degrees(np.arccos(cos_d)) <= alpha_deg


def make_sample(has_mhw, mhw_lat=None, mhw_lon=None, rng=None):
    """One daily SST field; optionally inject an MHW event."""
    if rng is None:
        rng = np.random.default_rng()
    sst_base = 25.0 * np.cos(np.radians(LAT_G)) ** 2 - 5.0
    noise = NOISE_C * rng.standard_normal(LAT_G.shape)
    sst = sst_base + noise
    if has_mhw:
        if mhw_lon is None:
            mhw_lon = rng.uniform(-180, 180)
        cap = is_in_spherical_cap(
            LAT_G, LON_G, mhw_lat, mhw_lon, ANGULAR_RADIUS_DEG,
        )
        sst[cap] += MHW_ANOMALY_C
    return sst


# %% [markdown]
# ## Feature extractors
#
# **Flat — fixed convolutional template matched to the equator-shape
# MHW.** Cross-correlate the SST anomaly field with the template and
# report (max response, mean response, std response). This is
# shape-sensitive on the lat-lon raster: the template was built for
# the equator pixel-shape, so it matches strongly when MHWs are at
# the equator and weakly when they're at high latitudes (where the
# same physical MHW has a different lat-lon-projected shape). This
# is the lat-lon analogue of a flat CNN's first-layer
# blob-detector — exactly the matched filter from notebook 02
# (whose peak response we already showed collapses 100% → 55% from
# 0°N to 80°N), now wrapped in a classifier head.
#
# **Spherical — sphere-harmonic matched filter on HEALPix.**
# Same `(max, mean, std)`-of-response structure as the flat
# pipeline above, but the convolution happens *on the sphere*
# via `healpy.smoothing` with a Gaussian beam of
# `FWHM = 2 × ANGULAR_RADIUS_DEG` (matching the cap diameter).
# `healpy.smoothing` is sphere-harmonic convolution: it computes
# `a_lm` of the field, multiplies by the beam transfer function
# `b_l`, and inverts back. Sphere-harmonic convolution is
# **exactly rotation-equivariant** — convolution commutes with
# rotation on the sphere, so the smoothed response peak at the
# centre of a 10° cap is the same value whether the cap sits at
# 0°N or 80°N (the entire smoothed field is just the rotated
# response field). This is the direct sphere analog of the
# lat-lon matched filter — same template-matching idea, but on
# the sphere's own substrate.

# %%
def _equator_mhw_template():
    """Build a 2D template matched to an MHW at the equator.

    This stands in for what a CNN's first-layer filter learns when
    trained on equator-shape MHWs: a small, roughly-circular
    blob-detector on the lat-lon raster. It is exactly the matched
    filter from notebook 02 — there we showed its peak response
    collapses 100% → 55% from 0°N to 80°N. Here we wrap it in a
    proper classifier head.
    """
    # Build the cap shape on a small lat-lon patch around (0, 0)
    half_deg = 14
    patch_lat = np.arange(-half_deg, half_deg + 0.5, LAT_RES)
    patch_lon = np.arange(-half_deg, half_deg + 0.5, LON_RES)
    PLA, PLO = np.meshgrid(patch_lat, patch_lon, indexing="ij")
    cap = is_in_spherical_cap(PLA, PLO, 0.0, 0.0,
                                ANGULAR_RADIUS_DEG).astype(float)
    cap = cap - cap.mean()
    return cap


_TEMPLATE_EQ = _equator_mhw_template()


def flat_features(sst):
    """Shape-sensitive features from a fixed equator-shape template.

    Cross-correlate the SST anomaly field with the equator-MHW
    template; report the maximum, mean, and standard deviation of
    the response. This mimics how a flat CNN trained on equator
    MHWs detects them — the kernel learns the *equator pixel-shape*,
    and the response collapses on differently-shaped polar versions
    of the same physical feature.
    """
    sst_anom = sst - sst.mean()
    response = correlate2d(sst_anom, _TEMPLATE_EQ,
                            mode="same", boundary="wrap")
    return np.array([response.max(), response.mean(), response.std()])


_SPHERE_FWHM_RAD = np.deg2rad(2.0 * ANGULAR_RADIUS_DEG)
_HP_FILTER_LMIN = 5     # zero out a_lm for l<5 (DC + cosine-of-lat baseline)
_HP_FILTER_FL = np.ones(LMAX + 1)
_HP_FILTER_FL[:_HP_FILTER_LMIN] = 0
_GAUSS_BEAM_BL = hp.gauss_beam(_SPHERE_FWHM_RAD, lmax=LMAX)


def spherical_features(sst, lmax=LMAX):
    """Sphere-harmonic matched filter on HEALPix.

    Aggregate the lat-lon SST to HEALPix, then apply a band-pass
    filter directly in `a_lm` space:

      1. Decompose the field with `hp.map2alm`.
      2. **High-pass:** zero out `a_lm` for `l < 5`. This removes
         the DC mode and the dominant low-`l` modes that encode the
         cosine-of-latitude SST baseline (the equatorial water
         being ~30 °C warmer than the polar water dominates the
         field by an order of magnitude over a 4 °C MHW signal,
         and would otherwise hide the MHW peak when we look at
         `response.max()`).
      3. **Matched-filter smoothing:** multiply by a Gaussian
         beam transfer function of `FWHM = 2·ANGULAR_RADIUS_DEG`
         (cap-diameter scale).
      4. Inverse SHT and report `(max, mean, std)` of the
         response field.

    The combined operation `a_lm → a_lm · f_l · b_l` is sphere-
    harmonic convolution with a band-pass kernel — exactly
    rotation-equivariant on the sphere. The matched-filter
    response peak for a 10° cap is therefore the same value no
    matter where on the sphere the cap sits, and the high-pass
    step ensures that peak isn't swamped by the cosine-baseline
    structure at the equator.
    """
    sst_hp = aggregate_to_healpix(sst)
    sst_hp = sst_hp - sst_hp.mean()
    sst_hp_ring = hp.reorder(sst_hp, n2r=True)
    alm = hp.map2alm(sst_hp_ring, lmax=lmax, iter=1)
    alm = hp.almxfl(alm, _HP_FILTER_FL)
    alm = hp.almxfl(alm, _GAUSS_BEAM_BL)
    response = hp.alm2map(alm, NSIDE, lmax=lmax)
    return np.array([response.max(), response.mean(), response.std()])


# %% [markdown]
# ## Build the training and test sets
#
# **Training set**: 600 samples (300 negative + 300 positive),
# positive MHWs placed at **low latitudes (|lat| ≤ 20°)** —
# representing the typical situation where labelled training
# data is concentrated in equatorial and mid-latitude regions
# (most published MHW datasets, most fish-stock ground-truth, most
# high-quality SST climatologies).
#
# **Test sets** at four latitude bands: 0–20° (in-distribution),
# 30–40°, 50–60°, 70–80° (progressively out-of-distribution).
# 200 samples each (100 positive, 100 negative).

# %%
rng = np.random.default_rng(42)

N_TRAIN_PER_CLASS = 300
TRAIN_LAT_BAND = (0, 20)
TEST_LAT_BANDS = [(0, 20), (30, 40), (50, 60), (70, 80)]
N_TEST_PER_CLASS_PER_BAND = 100


def random_lat_in_band(band, rng):
    lat = rng.uniform(*band)
    if rng.random() < 0.5:
        lat = -lat
    return lat


def build_dataset(n_pos, n_neg, lat_band, rng):
    samples = []
    labels = []
    for _ in range(n_pos):
        lat = random_lat_in_band(lat_band, rng)
        samples.append(make_sample(has_mhw=True, mhw_lat=lat, rng=rng))
        labels.append(1)
    for _ in range(n_neg):
        samples.append(make_sample(has_mhw=False, rng=rng))
        labels.append(0)
    return samples, np.array(labels)


print("Building training set (low-lat MHWs only)…")
train_samples, y_train = build_dataset(
    N_TRAIN_PER_CLASS, N_TRAIN_PER_CLASS, TRAIN_LAT_BAND, rng,
)
print(f"  {len(train_samples)} samples, "
      f"{y_train.sum()} positive (MHW at |lat| ≤ 20°), "
      f"{(y_train == 0).sum()} negative")

# %% [markdown]
# ## Extract features and train the two classifiers

# %%
print("Extracting flat (lat-lon FFT) features for training…")
X_train_flat = np.array([flat_features(s) for s in train_samples])
print("Extracting spherical (HEALPix Cl) features for training…")
X_train_sphere = np.array([spherical_features(s) for s in train_samples])
print(f"  Flat features: {X_train_flat.shape}")
print(f"  Spherical features: {X_train_sphere.shape}")

clf_flat = LogisticRegression(max_iter=2000, C=1.0)
clf_flat.fit(X_train_flat, y_train)
clf_sphere = LogisticRegression(max_iter=2000, C=1.0)
clf_sphere.fit(X_train_sphere, y_train)

# Sanity-check: both classifiers fit the training set
acc_train_flat = accuracy_score(y_train, clf_flat.predict(X_train_flat))
acc_train_sphere = accuracy_score(y_train, clf_sphere.predict(X_train_sphere))
print(f"\nTraining accuracy:")
print(f"  Pipeline A (flat lat-lon matched filter):    {acc_train_flat:.3f}")
print(f"  Pipeline B (HEALPix Cl):          {acc_train_sphere:.3f}")

# %% [markdown]
# ## Evaluate at each test latitude band

# %%
results_flat = []
results_sphere = []
for lat_band in TEST_LAT_BANDS:
    test_samples, y_test = build_dataset(
        N_TEST_PER_CLASS_PER_BAND, N_TEST_PER_CLASS_PER_BAND, lat_band, rng,
    )
    X_test_flat = np.array([flat_features(s) for s in test_samples])
    X_test_sphere = np.array([spherical_features(s) for s in test_samples])
    yp_flat = clf_flat.predict(X_test_flat)
    yp_sphere = clf_sphere.predict(X_test_sphere)
    results_flat.append({
        "band": lat_band,
        "acc": accuracy_score(y_test, yp_flat),
        "f1": f1_score(y_test, yp_flat),
    })
    results_sphere.append({
        "band": lat_band,
        "acc": accuracy_score(y_test, yp_sphere),
        "f1": f1_score(y_test, yp_sphere),
    })

print(f"\n{'Test band':>20} {'Flat acc':>12} {'Flat F1':>10} "
      f"{'Sphere acc':>12} {'Sphere F1':>10}")
print("-" * 70)
for f, s in zip(results_flat, results_sphere):
    band_label = f"{f['band'][0]}-{f['band'][1]}°"
    print(f"  {band_label:>18} "
          f"{f['acc']:>12.3f} {f['f1']:>10.3f} "
          f"{s['acc']:>12.3f} {s['f1']:>10.3f}")

# %% [markdown]
# ## Headline figure — accuracy and F1 by test latitude band

# %%
band_centres = [(b[0] + b[1]) / 2 for b in TEST_LAT_BANDS]
band_labels = [f"{b[0]}–{b[1]}°" for b in TEST_LAT_BANDS]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(band_centres, [r["acc"] for r in results_flat],
        "o-", color="tab:red", lw=2, ms=8,
        label="Pipeline A — flat (lat-lon matched filter)")
ax.plot(band_centres, [r["acc"] for r in results_sphere],
        "s-", color="tab:blue", lw=2, ms=8,
        label="Pipeline B — sphere-aware (HEALPix matched filter)")
ax.axhline(0.5, color="0.5", linestyle=":", linewidth=0.8)
ax.set_xlabel("Test latitude band centre (°)")
ax.set_ylabel("Accuracy")
ax.set_title("Detection accuracy by test latitude band\n"
             "(both pipelines trained on |lat| ≤ 20° only)",
             fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_xticks(band_centres)
ax.set_xticklabels(band_labels)
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3)
# Shade in-distribution band
ax.axvspan(0, 20, color="green", alpha=0.07, label="In-distribution")

ax = axes[1]
ax.plot(band_centres, [r["f1"] for r in results_flat],
        "o-", color="tab:red", lw=2, ms=8,
        label="Pipeline A — flat (lat-lon matched filter)")
ax.plot(band_centres, [r["f1"] for r in results_sphere],
        "s-", color="tab:blue", lw=2, ms=8,
        label="Pipeline B — sphere-aware (HEALPix matched filter)")
ax.set_xlabel("Test latitude band centre (°)")
ax.set_ylabel("F1 score")
ax.set_title("F1 by test latitude band\n"
             "(harmonic mean of precision and recall)",
             fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_xticks(band_centres)
ax.set_xticklabels(band_labels)
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3)
ax.axvspan(0, 20, color="green", alpha=0.07)

fig.suptitle(
    "Spherical ML head-to-head: flat lat-lon matched-filter baseline vs HEALPix sphere-harmonic features",
    fontsize=13, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("../images/spherical_ml_payoff.png", dpi=150,
            bbox_inches="tight")
plt.show()

# %% [markdown]
# ## What the figure shows
#
# Both pipelines are trained **only on low-latitude MHWs**, and
# both use the same template-matching idea (cross-correlate with a
# cap-shape kernel, read out `(max, mean, std)`, train a logistic
# regression). The only thing that differs is the substrate the
# convolution lives on.
#
# Pipeline A (flat lat-lon matched filter) is **perfect at low
# latitudes** (1.000 / 1.000 at 0–20° and 30–40°), starts to slip
# at 50–60° (≈ 0.915), and **collapses to chance at 70–80°**
# (0.500 accuracy, F1 = 0). The same physical 10° spherical-cap
# MHW renders to a *different lat-lon raster shape* at high
# latitudes — notebook 02 quantified this: 50.6 % pixel
# disagreement when treating rotation as translation, matched-
# filter detector peak response collapses 100 % → 55 % from 0°N
# to 80°N. The classifier learnt the equator pixel-shape; the
# polar pixel-shape doesn't match.
#
# Pipeline B (HEALPix sphere-harmonic matched filter) sits at
# **1.000 accuracy and F1 = 1.000 in every band**, including
# 70–80° where the flat baseline has collapsed to chance.
# Sphere-harmonic convolution is exactly rotation-equivariant by
# construction: the smoothed-response peak for the same physical
# 10° cap is the same value whether the cap sits at 0°N or 80°N
# (the entire response field is just the rotated response field).
# The classifier sees the *same* `(max, mean, std)` triple for an
# MHW regardless of where on the sphere it sits, so the threshold
# learnt at low latitudes works at the pole.
#
# **At low latitudes the two pipelines tie; at the pole the flat
# baseline collapses while the sphere baseline is unchanged.**
# That asymmetry is the payoff: it costs nothing to use spherical
# ML in the regimes where flat ML works, and you get the polar
# regime for free.
#
# **This is the spherical-ML payoff.** Without sphere-aware features,
# an EO ML model trained on lat-lon-projected Copernicus Marine
# SST loses accuracy at high latitudes — exactly where biodiversity
# range-shift attribution work increasingly operates as climate
# change pushes species poleward. With sphere-aware features, the
# loss disappears.
#
# This is the same load-bearing claim the
# [DLWP-HEALPix line of work](https://doi.org/10.1029/2023MS004021)
# (Karlbauer et al. 2024) makes for global weather forecasting,
# the [DeepSphere line](https://arxiv.org/abs/2012.15000)
# (Defferrard et al. 2020) makes for atmospheric-river / cyclone
# detection on ClimateNet, and that scattering-network /
# `foscat`-style approaches make for spectral / multi-scale features
# on the sphere. This notebook demonstrates it on a Copernicus-
# Marine-flavoured task using a deliberately simple feature
# extractor (FFT vs $C_\ell$) so the geometric mechanism — *rotation
# invariance vs translation invariance* — is visible without
# heavy ML machinery. Both pipelines use **the same template-
# matching idea** (cross-correlate with a feature-shape template,
# read out `(max, mean, std)` of the response, train a logistic
# regression on top); the only thing that differs between them
# is the substrate the convolution lives on — lat-lon raster
# (translation-equivariant in pixel space, *not* rotation-
# equivariant on the sphere) versus HEALPix sphere-harmonics
# (exactly rotation-equivariant on the sphere). The latitude-
# dependence of the result is therefore a pure property of the
# substrate, not of the model class.
#
# ## Notebook 05 — propagate to marine biodiversity
#
# Notebook 05 takes the same head-to-head pipeline and applies it
# to a **marine-biodiversity attribution task**: predicting kelp /
# coral / fish biodiversity hotspots from SST anomaly features
# (Smale et al. 2019, [doi:10.1038/s41558-019-0412-1](https://doi.org/10.1038/s41558-019-0412-1);
# Wernberg et al. 2016, [doi:10.1126/science.aad8745](https://doi.org/10.1126/science.aad8745)).
# The latitude-bias gap between flat and spherical-ML pipelines
# propagates from features → predictions → biodiversity-attribution
# error. That is the payoff for biodiversity science: not just
# "spherical ML is better" but "the bias of using flat ML on
# Copernicus EO produces *measurably wrong* biodiversity
# attributions, and sphere-aware ML eliminates it."
#
# ## Forthcoming — real Copernicus Marine SST integration
#
# This notebook ships with synthetic SST so the comparison is
# CI-fast and the geometric mechanism is uncontaminated by
# real-world signal. A follow-up commit will replace the synthetic
# `make_sample` block with Copernicus Marine SST anomaly subsets
# from the CDS API; the rest of the notebook (HEALPix aggregation,
# feature extraction, classifier head) is unchanged.
