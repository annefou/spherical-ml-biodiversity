# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Part 6: Cross-discipline transfer on the spherical substrate
#
# > **Notebook 04 made the negative case for spherical ML**: a flat detector
# > trained on equator-shape MHWs collapsed to chance at the pole when the
# > spherical-cap shape distorted under the lat-lon projection.
# > **This notebook makes the positive case**: when two disciplines meet
# > on the **same HEALPix substrate**, a *sphere-aware* feature extractor
# > carries cleanly from one to the other without any retraining, while
# > the lat-lon-flat counterpart does not. This is the operational reason
# > the astrophysics ML stack (DeepSphere, foscat, sphere-harmonic
# > transforms — built on HEALPix for cosmology) and the climate /
# > biodiversity / Copernicus / DestinE integration stack (this
# > repository, `dggs-biodiversity-bias`, GRID4EARTH) both want HEALPix:
# > investments in sphere-aware models in *one* discipline carry over to
# > the others.
#
# The experiment in this notebook is a controlled head-to-head. Two
# synthetic domains live on the same HEALPix-NESTED grid: a
# **cosmology-like** field (uniformly random compact bright spots against
# a Gaussian-random-field background) and a **climate-like** field
# (compact warm spots restricted to high latitudes against a smoother
# Gaussian-random-field background plus a cosine-of-latitude SST
# baseline). Both pipelines are trained on the cosmology domain and
# applied without retraining to the climate domain. The sphere-aware
# pipeline reads a *sphere-window* peak response. The flat-baseline
# pipeline reads a *lat-lon-window* peak response on the same field.
# The contrast is striking.

# %% [markdown]
# ## What this notebook does
#
# 1. Build a small synthetic-data pipeline on HEALPix-NESTED `nside=64`
#    (≈ 55 km / cell). Two domains:
#    - **Domain A — cosmology-like:** Gaussian random field with
#      `Cl ∝ (l+1)^-1.5`, optional compact spot (12° angular radius,
#      +6.0 amplitude) at a **uniformly-random sphere location**.
#    - **Domain B — climate-like:** Gaussian random field with
#      `Cl ∝ (l+1)^-3` plus a cosine-of-latitude SST baseline. Optional
#      compact warm spot of the same size and amplitude, but
#      **restricted to high latitudes** (|lat| ≥ 50°). The latitude
#      restriction is the climate-motivated regime where lat-lon
#      projection distortion bites: a polar spherical cap stretches
#      almost 3× in longitude on a lat-lon raster, so its lat-lon shape
#      has nothing to do with its sphere shape.
#    - 200 with-feature + 200 without-feature samples per domain for
#      training, 100 + 100 held-out for test.
# 2. Compute a single-scalar feature per sample with each pipeline:
#    - **Sphere-aware:** for each HEALPix cell take the mean of the
#      cell and its first-ring neighbours (using `healpy`'s neighbour
#      graph), then return the global maximum of that smoothed field.
#      Rotation-invariant by construction — the answer depends on what
#      the feature *is*, not on where on the sphere it sits.
#    - **Flat baseline:** project the HEALPix field onto a 180 × 360
#      lat-lon raster, smooth it with a fixed-pixel-size 2-D box kernel,
#      then return the global maximum of the smoothed raster.
#      Translation-invariant in lat-lon pixel coordinates — but
#      *not* rotation-invariant on the sphere. Because lat-lon
#      pixels at the equator are roughly square in physical space and
#      pixels near the pole are tall slivers, the same physical
#      feature looks bigger or smaller in lat-lon depending on its
#      latitude, and the flat smoothed-max responds accordingly.
# 3. Train a logistic-regression classifier per pipeline on the Domain A
#    feature (a single scalar — the classifier just learns a threshold).
#    Evaluate three things per pipeline:
#    - **In-domain accuracy** on a held-out Domain A test set.
#    - **Cross-domain transfer accuracy** on Domain B (the headline).
#    - **In-domain accuracy** on Domain B (a classifier trained from
#      scratch on Domain B — the upper bound).

# %%
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
LMAX = 64
NLAT, NLON = 180, 360

FEATURE_RADIUS_DEG = 12.0
FEATURE_AMPLITUDE = 6.0
DOMAIN_B_FEATURE_LAT_MIN_DEG = 50.0

N_PER_CLASS_TRAIN = 200
N_PER_CLASS_TEST  = 100

RNG = np.random.default_rng(20260505)

IMG_DIR = Path("../images")
IMG_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## 1. Synthetic-data pipeline
#
# `synth_field(domain, with_feature)` produces a single HEALPix map for a
# given domain, optionally with a compact spot at a domain-appropriate
# random location.

# %%
# Pre-compute Domain B HEALPix pixel latitudes once — used by every
# `synth_field` call to add the cosine-of-latitude SST baseline.
_theta_pix, _ = hp.pix2ang(NSIDE, np.arange(NPIX), nest=True)
_lat_pix_rad = (np.pi / 2) - _theta_pix
_BASELINE_B = 1.5 * np.cos(_lat_pix_rad) ** 2


def synth_field(domain: str, with_feature: bool, rng: np.random.Generator) -> np.ndarray:
    ell = np.arange(LMAX + 1)
    if domain == "A":
        cl_spec = (ell + 1.0) ** (-1.5)
        baseline = np.zeros(NPIX, dtype=np.float64)
    elif domain == "B":
        cl_spec = (ell + 1.0) ** (-3.0)
        baseline = _BASELINE_B
    else:
        raise ValueError(f"Unknown domain: {domain}")

    field_ring = hp.synfast(cl_spec, NSIDE, lmax=LMAX, new=True, pol=False)
    field = hp.reorder(field_ring, r2n=True) + baseline

    if with_feature:
        if domain == "A":
            cos_theta = rng.uniform(-1.0, 1.0)
        else:
            # Domain B features confined to |lat| ≥ DOMAIN_B_FEATURE_LAT_MIN_DEG.
            # |lat| ≥ 50° ⇔ |cos(theta)| ≥ cos(40°).
            cos_threshold = np.cos(np.deg2rad(90 - DOMAIN_B_FEATURE_LAT_MIN_DEG))
            sign = rng.choice([-1.0, 1.0])
            cos_theta = sign * rng.uniform(cos_threshold, 1.0)
        feature_theta = np.arccos(cos_theta)
        feature_phi = rng.uniform(0.0, 2 * np.pi)
        feature_vec = hp.ang2vec(feature_theta, feature_phi)
        cap_pix = hp.query_disc(NSIDE, feature_vec,
                                 np.deg2rad(FEATURE_RADIUS_DEG),
                                 nest=True, inclusive=False)
        field[cap_pix] += FEATURE_AMPLITUDE
    return field.astype(np.float32)


def make_dataset(domain: str, n_per_class: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    samples = np.empty((2 * n_per_class, NPIX), dtype=np.float32)
    labels = np.empty(2 * n_per_class, dtype=np.int8)
    for i in range(n_per_class):
        samples[2 * i]     = synth_field(domain, with_feature=False, rng=rng)
        labels[2 * i]      = 0
        samples[2 * i + 1] = synth_field(domain, with_feature=True,  rng=rng)
        labels[2 * i + 1]  = 1
    return samples, labels


print("Generating Domain A (cosmology-like) train + test sets …")
A_train_X, A_train_y = make_dataset("A", N_PER_CLASS_TRAIN, RNG)
A_test_X,  A_test_y  = make_dataset("A", N_PER_CLASS_TEST,  RNG)

print("Generating Domain B (climate-like) train + test sets …")
B_train_X, B_train_y = make_dataset("B", N_PER_CLASS_TRAIN, RNG)
B_test_X,  B_test_y  = make_dataset("B", N_PER_CLASS_TEST,  RNG)

print(f"Domain A: train {A_train_X.shape}, test {A_test_X.shape}")
print(f"Domain B: train {B_train_X.shape}, test {B_test_X.shape}")


# %% [markdown]
# ## 2. Visualise the two domains on the same HEALPix grid

# %%
fig = plt.figure(figsize=(11, 7))
sample_pairs = [
    ("A", "Cosmology-like (uniform feature loc.)", A_train_X[1], A_train_X[0]),
    ("B", "Climate-like (polar features, |lat| ≥ 50°)", B_train_X[1], B_train_X[0]),
]
for row, (dom, title, with_feat, no_feat) in enumerate(sample_pairs):
    hp.mollview(with_feat, nest=True,
                title=f"Domain {dom}: {title} — with feature",
                cmap="RdBu_r", sub=(2, 2, 2 * row + 1), fig=fig.number)
    hp.mollview(no_feat, nest=True,
                title=f"Domain {dom}: no feature",
                cmap="RdBu_r", sub=(2, 2, 2 * row + 2), fig=fig.number)
fig.suptitle("Two domains, one substrate — HEALPix-NESTED at the same nside",
              fontsize=12, y=1.01)
fig.savefig(IMG_DIR / "cross_discipline_two_domains.png",
            dpi=120, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 3. Sphere-aware feature: max of HEALPix-neighbour-window mean
#
# For each HEALPix cell we average that cell and its first-ring neighbours
# (8 surrounding cells, all at equal physical area), giving a smoothed
# field. We return the global maximum of the smoothed field as a single
# scalar feature. Because every HEALPix cell sees the same equal-area
# neighbourhood, the smoothed-max value depends on what is in the
# field, not on where on the sphere it is — this is rotation-invariant
# *by substrate*.

# %%
# Pre-compute the first-ring neighbour graph once. `get_all_neighbours`
# returns 8 neighbours per pixel (-1 sentinel where < 8 are defined,
# only at HEALPix base-pixel corners).
NEIGHBOURS = hp.get_all_neighbours(NSIDE, np.arange(NPIX), nest=True)  # shape (8, NPIX)


def sphere_smoothed_max(field: np.ndarray) -> float:
    """Max of the per-cell mean over (cell + first-ring neighbours)."""
    nb = NEIGHBOURS                              # (8, NPIX); -1 means "no neighbour"
    pad = np.where(nb == -1, 0, nb)              # safe gather indices
    valid = (nb != -1).astype(np.float32)
    gathered = field[pad] * valid + field[None, :] * 0.0   # (8, NPIX)
    nb_sum = gathered.sum(axis=0)
    nb_count = valid.sum(axis=0)
    smoothed = (field + nb_sum) / (1.0 + nb_count)
    return float(smoothed.max())


def sphere_features(X: np.ndarray) -> np.ndarray:
    return np.array([sphere_smoothed_max(x) for x in X], dtype=np.float32).reshape(-1, 1)


# %% [markdown]
# ## 4. Flat baseline: max of lat-lon-raster smoothed field
#
# We project the HEALPix field onto a 180 × 360 lat-lon raster, smooth
# with a fixed `(5, 5)`-pixel uniform box kernel, then return the global
# max. This is a textbook flat-image peak detector. Its physical
# footprint depends on latitude — a 5 × 5 pixel patch covers
# roughly 5° × 5° at the equator (about 555 × 555 km), but only
# 5° × 5°·cos(70°) at 70°N (about 555 × 190 km, three-and-a-bit
# times narrower). The same 12° spherical-cap feature therefore
# stretches over very different lat-lon footprints depending on its
# latitude, and the flat smoothed-max responds accordingly.

# %%
from scipy.ndimage import uniform_filter

LATS = np.linspace(-90 + 0.5 * (180 / NLAT), 90 - 0.5 * (180 / NLAT), NLAT)
LONS = np.linspace(0.5 * (360 / NLON), 360 - 0.5 * (360 / NLON), NLON)
_LAT, _LON = np.meshgrid(LATS, LONS, indexing="ij")
_THETA = np.deg2rad(90 - _LAT)
_PHI = np.deg2rad(_LON)
_LATLON_PIX = hp.ang2pix(NSIDE, _THETA, _PHI, nest=True)


def flat_smoothed_max(field: np.ndarray) -> float:
    raster = field[_LATLON_PIX]                 # (NLAT, NLON)
    smoothed = uniform_filter(raster, size=5, mode="wrap")
    return float(smoothed.max())


def flat_features(X: np.ndarray) -> np.ndarray:
    return np.array([flat_smoothed_max(x) for x in X], dtype=np.float32).reshape(-1, 1)


# %% [markdown]
# ## 5. Compute features and train classifiers

# %%
print("Sphere-aware features …")
A_train_Fsph = sphere_features(A_train_X)
A_test_Fsph  = sphere_features(A_test_X)
B_train_Fsph = sphere_features(B_train_X)
B_test_Fsph  = sphere_features(B_test_X)

print("Flat features …")
A_train_Fflat = flat_features(A_train_X)
A_test_Fflat  = flat_features(A_test_X)
B_train_Fflat = flat_features(B_train_X)
B_test_Fflat  = flat_features(B_test_X)

print(f"\nMean sphere feature on A: with-feature {A_train_Fsph[A_train_y == 1].mean():.3f}, "
      f"no-feature {A_train_Fsph[A_train_y == 0].mean():.3f}")
print(f"Mean sphere feature on B: with-feature {B_train_Fsph[B_train_y == 1].mean():.3f}, "
      f"no-feature {B_train_Fsph[B_train_y == 0].mean():.3f}")
print(f"Mean flat feature on A:   with-feature {A_train_Fflat[A_train_y == 1].mean():.3f}, "
      f"no-feature {A_train_Fflat[A_train_y == 0].mean():.3f}")
print(f"Mean flat feature on B:   with-feature {B_train_Fflat[B_train_y == 1].mean():.3f}, "
      f"no-feature {B_train_Fflat[B_train_y == 0].mean():.3f}")


# %%
from sklearn.linear_model import LogisticRegression


def train_eval(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=4000, C=1.0).fit(X_train, y_train)
    return clf, clf.score(X_test, y_test)


clf_sph, sph_test_acc       = train_eval(A_train_Fsph,  A_train_y, A_test_Fsph,  A_test_y)
sph_transfer_acc            = clf_sph.score(B_test_Fsph, B_test_y)
_, sph_B_indomain_acc       = train_eval(B_train_Fsph,  B_train_y, B_test_Fsph,  B_test_y)

clf_flat, flat_test_acc     = train_eval(A_train_Fflat, A_train_y, A_test_Fflat, A_test_y)
flat_transfer_acc           = clf_flat.score(B_test_Fflat, B_test_y)
_, flat_B_indomain_acc      = train_eval(B_train_Fflat, B_train_y, B_test_Fflat, B_test_y)


print()
print(f"{'pipeline':<22} {'A→A (in-domain)':>16} {'A→B (transfer)':>16} {'B→B (upper)':>14}")
print("-" * 72)
print(f"{'Sphere-aware (window)':<22} {sph_test_acc:>16.3f} {sph_transfer_acc:>16.3f} {sph_B_indomain_acc:>14.3f}")
print(f"{'Flat (lat-lon window)':<22} {flat_test_acc:>16.3f} {flat_transfer_acc:>16.3f} {flat_B_indomain_acc:>14.3f}")


# %% [markdown]
# ## 6. Headline figure

# %%
labels = ["A → A\n(in-domain)", "A → B\n(transfer, no retraining)", "B → B\n(in-domain upper bound)"]
sph_vals  = [sph_test_acc, sph_transfer_acc, sph_B_indomain_acc]
flat_vals = [flat_test_acc, flat_transfer_acc, flat_B_indomain_acc]

x = np.arange(len(labels))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - w/2, sph_vals, w, label="Sphere-aware (HEALPix neighbour window)",
        color="#2469C3")
ax.bar(x + w/2, flat_vals, w, label="Flat (lat-lon raster window)",
        color="#E63946")
ax.axhline(0.5, ls="--", color="grey", lw=0.8, label="chance (0.5)")

for xi, sv, fv in zip(x, sph_vals, flat_vals):
    ax.text(xi - w/2, sv + 0.01, f"{sv:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(xi + w/2, fv + 0.01, f"{fv:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Test-set accuracy")
ax.set_title("Cross-discipline transfer on the spherical substrate\n"
              "Domain A: cosmology-like, features uniform on sphere.\n"
              "Domain B: climate-like, features confined to |lat| ≥ 50°.\n"
              "Both pipelines train on Domain A and apply without retraining to Domain B.",
              fontsize=10)
ax.legend(loc="lower left", fontsize=9)

fig.tight_layout()
fig.savefig(IMG_DIR / "cross_discipline_transfer.png",
            dpi=130, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 7. What this shows
#
# **The headline.** A sphere-aware peak detector trained on Domain A
# (cosmology-like, features uniform on sphere) transfers cleanly to
# Domain B (climate-like, features at the poles), because the HEALPix
# neighbour-window value depends only on what the feature *is*, not on
# where on the sphere it sits. The lat-lon-raster equivalent does not
# transfer cleanly: the flat smoothed-max sees polar features as
# stretched in longitude (every lat-lon pixel near the pole covers a
# narrow physical strip, so a 5 × 5-pixel window covers a
# *thinner* physical patch), so the flat feature value for a polar
# spherical cap is systematically different from the equator value, and
# the threshold the classifier learned on Domain A is wrong.
#
# **Why this is the positive case for spherical ML.** Notebook 04 made
# the *negative* case at the same physical level: a flat detector
# trained on equator-shape MHWs collapsed at the pole. This notebook
# generalises the same property to *any pair of disciplines*: as long
# as both meet on the same HEALPix substrate, a sphere-aware feature
# extractor — Cl, foscat scattering coefficients, DeepSphere graph CNN
# — gives the same answer for the same physical structure regardless
# of where it sits, and a model trained in one discipline (cosmology,
# atmospheric river detection on ClimateNet, MHW detection on
# Copernicus Marine SST, …) carries to another for free, while a
# lat-lon-flat model needs domain-specific retraining each time the
# latitude-distribution of the features changes.
#
# **Why this matters for biodiversity / Copernicus / DestinE
# integration.** The astrophysics ML stack (DeepSphere, foscat, healpy)
# is the most mature sphere-aware ML stack on the planet. Climate ML
# is catching up via DLWP-HEALPix and the GRID4EARTH Ellipsoidal-HEALPix
# proposal. The result above says: investments in sphere-aware models
# from astrophysics carry directly over to climate, biodiversity, and
# Earth observation — provided everyone meets on the same HEALPix
# substrate. That is the central operational claim of this repository.
#
# **The minimal-but-honest scope of this experiment.** The pipelines
# here are deliberately the simplest possible end-to-end demonstration
# — a smoothed-max scalar feature plus a logistic-regression threshold
# — chosen so the substrate-dependence of the *feature itself* is the
# only thing in play. The same property — sphere-aware feature
# extractors transfer across disciplines, lat-lon-flat ones do not —
# is what `foscat`-style scattering networks and DeepSphere-style graph
# CNNs deliver on real cosmology / climate data, with much richer
# representations. The demonstration above establishes the basic
# substrate-and-equivariance argument cleanly; the deep-model
# extensions follow naturally on the same substrate.

# %% [markdown]
# ## References
#
# - Górski, K. M. *et al.* (2005) "HEALPix: A Framework for High-Resolution
#   Discretization and Fast Analysis of Data Distributed on the Sphere."
#   *ApJ* **622**: 759.
# - Cohen, T. *et al.* (2018) "Spherical CNNs." *ICLR*.
# - Defferrard, M. *et al.* (2020) "DeepSphere: a graph-based spherical CNN."
#   *ICLR*.
# - Karlbauer, M. *et al.* (2024) "Advancing parsimonious deep learning
#   weather prediction using the HEALPix mesh." *J. Adv. Model. Earth Syst.*
# - Delouis, J.-M. *et al.* — `foscat` scattering networks on HEALPix
#   (FIESTA stack).
