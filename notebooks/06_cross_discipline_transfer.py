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
# > **Notebook 04 made the negative-direction case for spherical ML**:
# > a flat lat-lon matched filter trained on equator-shape MHWs
# > collapsed from 1.000 accuracy at 0–20° to 0.500 (chance) at 70–80°,
# > while a sphere-harmonic matched filter on HEALPix held at 1.000
# > across every test latitude band.
# >
# > **This notebook makes the cross-discipline case** with the same
# > technique. We swap the within-discipline latitude-band split for
# > a between-discipline split: train on a **cosmology-like** domain
# > (compact bright spots against a Gaussian random field, features
# > placed at uniformly random sphere locations) and apply the same
# > classifier without retraining to a **climate-like** domain
# > (compact warm spots against a smoother random field plus a
# > cosine-of-latitude SST baseline, features confined to high
# > latitudes |lat| ≥ 50°). The sphere-aware matched filter is
# > rotation-equivariant on the sphere by construction, so its
# > response peak doesn't notice the latitude shift. The flat
# > matched filter has an equator-shape template baked in, so its
# > response peak collapses on the polar-stretched features in the
# > climate domain — even though both domains live on the same
# > HEALPix substrate.
# >
# > This is the operational reason the astrophysics ML stack
# > (DeepSphere, foscat, sphere-harmonic transforms — built on
# > HEALPix for cosmology) and the climate / biodiversity /
# > Copernicus / DestinE integration stack (this repository,
# > `dggs-biodiversity-bias`, GRID4EARTH) both want HEALPix:
# > investments in sphere-aware models in *one* discipline carry over
# > to the others through the shared substrate.

# %% [markdown]
# ## What this notebook does
#
# 1. Build two synthetic domains on the same HEALPix-NESTED `nside=64`
#    grid. Both share the same physical feature physics (a 12° angular-
#    radius spherical cap with +6.0 amplitude); the rest is what makes
#    them feel like different disciplines.
#    - **Domain A — cosmology-like:** Gaussian random field with
#      `Cl ∝ (l+1)^-1.5` (steep, ≈ CMB-ish). The optional feature is
#      placed at a **uniformly random** sphere location (cosmological
#      observations have no preferred direction at first order, modulo
#      the galactic plane).
#    - **Domain B — climate-like:** Gaussian random field with
#      `Cl ∝ (l+1)^-3` (smoother) plus a cosine-of-latitude SST
#      baseline. The optional feature is restricted to **high
#      latitudes** (|lat| ≥ 50°) — the climate-motivated regime
#      where lat-lon projection distortion bites: a polar
#      spherical cap stretches almost 3× in longitude on a lat-lon
#      raster (cos⁻¹(70°) ≈ 2.92), so its lat-lon shape has nothing
#      to do with its physical sphere shape.
# 2. Reuse the two feature extractors from notebook 04, unchanged:
#    - **Flat baseline:** cross-correlate the lat-lon raster with an
#      equator-shape 10°-cap template, return `(max, mean, std)` of
#      the response.
#    - **Sphere-aware:** sphere-harmonic band-pass matched filter on
#      HEALPix — `aₗₘ → aₗₘ · fₗ · bₗ` where `fₗ` is a high-pass that
#      zeros `ℓ < 5` (suppressing the cosine-of-latitude baseline) and
#      `bₗ` is a Gaussian beam at the cap scale. Inverse SHT, return
#      `(max, mean, std)` of the response field. Sphere-harmonic
#      convolution is exactly rotation-equivariant, so the response
#      peak for a 12° cap is the same value no matter where on the
#      sphere it sits — and `fₗ` makes it robust to the
#      domain-specific baseline structure.
# 3. Train both pipelines on Domain A only, evaluate on:
#    - **Domain A test set** (in-domain accuracy — sanity).
#    - **Domain B test set** (cross-domain transfer — headline).

# %%
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d

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

# %%
# Lat-lon grid (matches notebook 04 conventions).
LATS_GRID = np.arange(-89.5, 90, 1.0)
LONS_GRID = np.arange(-179.5, 180, 1.0)
LAT_G, LON_G = np.meshgrid(LATS_GRID, LONS_GRID, indexing="ij")

# Lat-lon → HEALPix index lookup (precomputed once).
_theta_grid = np.radians(90.0 - LAT_G.ravel())
_phi_grid = np.radians(LON_G.ravel() % 360.0)
_PIX_INDEX = hp.ang2pix(NSIDE, _theta_grid, _phi_grid, nest=True).reshape(LAT_G.shape)
_COUNTS_PER_PIX = np.bincount(_PIX_INDEX.ravel(), minlength=NPIX)
_COUNTS_SAFE = np.where(_COUNTS_PER_PIX > 0, _COUNTS_PER_PIX, 1)


def aggregate_to_healpix(field_latlon: np.ndarray) -> np.ndarray:
    sums = np.bincount(_PIX_INDEX.ravel(),
                        weights=field_latlon.ravel(),
                        minlength=NPIX)
    out = sums / _COUNTS_SAFE
    out[_COUNTS_PER_PIX == 0] = 0.0
    return out


def is_in_spherical_cap(lat, lon, lat_c, lon_c, alpha_deg):
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    lat_cr, lon_cr = np.radians(lat_c), np.radians(lon_c)
    cos_d = (np.sin(lat_cr) * np.sin(lat_r)
             + np.cos(lat_cr) * np.cos(lat_r) * np.cos(lon_r - lon_cr))
    return np.degrees(np.arccos(np.clip(cos_d, -1, 1))) <= alpha_deg


def synth_field(domain: str, with_feature: bool, rng: np.random.Generator) -> np.ndarray:
    """Generate one global lat-lon field, optionally with a feature."""
    if domain == "A":
        # Cosmology-like: Gaussian random field with steep Cl, no
        # latitudinal baseline. We sample on HEALPix via synfast
        # then project to lat-lon — same field on both substrates.
        ell = np.arange(LMAX + 1)
        cl_spec = (ell + 1.0) ** (-1.5)
        field_hp_ring = hp.synfast(cl_spec, NSIDE, lmax=LMAX, new=True, pol=False)
        field_hp = hp.reorder(field_hp_ring, r2n=True)
        field_latlon = field_hp[_PIX_INDEX]
        baseline = 0.0
    elif domain == "B":
        # Climate-like: smoother Cl + cosine-of-latitude SST baseline.
        ell = np.arange(LMAX + 1)
        cl_spec = (ell + 1.0) ** (-3.0)
        field_hp_ring = hp.synfast(cl_spec, NSIDE, lmax=LMAX, new=True, pol=False)
        field_hp = hp.reorder(field_hp_ring, r2n=True)
        field_latlon = field_hp[_PIX_INDEX]
        baseline = 25.0 * np.cos(np.radians(LAT_G)) ** 2 - 5.0
        field_latlon = field_latlon + baseline
    else:
        raise ValueError(f"Unknown domain: {domain}")

    if with_feature:
        if domain == "A":
            # Uniformly random sphere location.
            lat_c = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0)))
        else:
            # Domain B: |lat| ≥ DOMAIN_B_FEATURE_LAT_MIN_DEG, uniform-cos
            # within band so the per-area distribution is uniform.
            cos_threshold = np.cos(np.radians(90 - DOMAIN_B_FEATURE_LAT_MIN_DEG))
            sign = rng.choice([-1.0, 1.0])
            cos_theta = sign * rng.uniform(cos_threshold, 1.0)
            lat_c = np.degrees(np.arcsin(cos_theta))
        lon_c = rng.uniform(-180.0, 180.0)
        cap = is_in_spherical_cap(LAT_G, LON_G, lat_c, lon_c, FEATURE_RADIUS_DEG)
        field_latlon = field_latlon + cap.astype(np.float64) * FEATURE_AMPLITUDE
    return field_latlon


def make_dataset(domain: str, n_per_class: int, rng: np.random.Generator):
    samples, labels = [], []
    for _ in range(n_per_class):
        samples.append(synth_field(domain, with_feature=True, rng=rng));  labels.append(1)
        samples.append(synth_field(domain, with_feature=False, rng=rng)); labels.append(0)
    return samples, np.array(labels, dtype=np.int8)


print("Generating Domain A (cosmology-like) train + test sets …")
A_train_X, A_train_y = make_dataset("A", N_PER_CLASS_TRAIN, RNG)
A_test_X,  A_test_y  = make_dataset("A", N_PER_CLASS_TEST,  RNG)

print("Generating Domain B (climate-like, polar features) train + test sets …")
B_train_X, B_train_y = make_dataset("B", N_PER_CLASS_TRAIN, RNG)
B_test_X,  B_test_y  = make_dataset("B", N_PER_CLASS_TEST,  RNG)

print(f"Domain A: train {len(A_train_X)} samples, test {len(A_test_X)} samples")
print(f"Domain B: train {len(B_train_X)} samples, test {len(B_test_X)} samples")


# %% [markdown]
# ## 2. Visualise the two domains on the same HEALPix grid

# %%
fig = plt.figure(figsize=(11, 7))
sample_pairs = [
    ("A", "Cosmology-like (uniform feature loc.)", A_train_X[0], A_train_X[1]),
    ("B", "Climate-like (polar features, |lat| ≥ 50°)", B_train_X[0], B_train_X[1]),
]
for row, (dom, title, with_feat, no_feat) in enumerate(sample_pairs):
    hp.mollview(aggregate_to_healpix(with_feat), nest=True,
                title=f"Domain {dom}: {title} — with feature",
                cmap="RdBu_r", sub=(2, 2, 2 * row + 1), fig=fig.number)
    hp.mollview(aggregate_to_healpix(no_feat), nest=True,
                title=f"Domain {dom}: no feature",
                cmap="RdBu_r", sub=(2, 2, 2 * row + 2), fig=fig.number)
fig.suptitle("Two domains, one substrate — HEALPix-NESTED at the same nside",
              fontsize=12, y=1.01)
fig.savefig(IMG_DIR / "cross_discipline_two_domains.png",
            dpi=120, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 3. Feature extractors (reused from notebook 04)
#
# Both extractors are the same template-matching idea — cross-correlate
# the field with a cap-shape kernel, return `(max, mean, std)` of the
# response. The flat extractor cross-correlates a fixed equator-shape
# template against the lat-lon raster (translation-equivariant in
# pixel space, *not* rotation-equivariant on the sphere). The sphere
# extractor multiplies `aₗₘ` by a band-pass (high-pass `fₗ` to remove
# the cosine-of-latitude baseline + Gaussian beam `bₗ` at the cap
# scale), inverts the SHT, and reads the response field. Sphere-
# harmonic convolution commutes with rotation on the sphere, so the
# response peak for a 12° cap is the same value regardless of where
# on the sphere the cap sits.

# %%
def _equator_cap_template():
    half_deg = int(np.ceil(FEATURE_RADIUS_DEG)) + 4
    patch_lat = np.arange(-half_deg, half_deg + 0.5, 1.0)
    patch_lon = np.arange(-half_deg, half_deg + 0.5, 1.0)
    PLA, PLO = np.meshgrid(patch_lat, patch_lon, indexing="ij")
    cap = is_in_spherical_cap(PLA, PLO, 0.0, 0.0,
                                FEATURE_RADIUS_DEG).astype(float)
    return cap - cap.mean()


_TEMPLATE_EQ = _equator_cap_template()

_SPHERE_FWHM_RAD = np.radians(2.0 * FEATURE_RADIUS_DEG)
_HP_FILTER_FL = np.ones(LMAX + 1); _HP_FILTER_FL[:5] = 0
_GAUSS_BEAM_BL = hp.gauss_beam(_SPHERE_FWHM_RAD, lmax=LMAX)


def flat_features(field_latlon: np.ndarray) -> np.ndarray:
    """Lat-lon matched filter against an equator-shape cap template."""
    anom = field_latlon - field_latlon.mean()
    response = correlate2d(anom, _TEMPLATE_EQ, mode="same", boundary="wrap")
    return np.array([response.max(), response.mean(), response.std()])


def sphere_features(field_latlon: np.ndarray) -> np.ndarray:
    """Sphere-harmonic band-pass matched filter on HEALPix."""
    field_hp = aggregate_to_healpix(field_latlon)
    field_hp = field_hp - field_hp.mean()
    field_hp_ring = hp.reorder(field_hp, n2r=True)
    alm = hp.map2alm(field_hp_ring, lmax=LMAX, iter=1)
    alm = hp.almxfl(alm, _HP_FILTER_FL)
    alm = hp.almxfl(alm, _GAUSS_BEAM_BL)
    response = hp.alm2map(alm, NSIDE, lmax=LMAX)
    return np.array([response.max(), response.mean(), response.std()])


# %% [markdown]
# ## 4. Compute features and train classifiers

# %%
print("Sphere-aware matched-filter features …")
A_train_Fsph = np.array([sphere_features(x) for x in A_train_X])
A_test_Fsph  = np.array([sphere_features(x) for x in A_test_X])
B_train_Fsph = np.array([sphere_features(x) for x in B_train_X])
B_test_Fsph  = np.array([sphere_features(x) for x in B_test_X])

print("Flat matched-filter features …")
A_train_Fflat = np.array([flat_features(x) for x in A_train_X])
A_test_Fflat  = np.array([flat_features(x) for x in A_test_X])
B_train_Fflat = np.array([flat_features(x) for x in B_train_X])
B_test_Fflat  = np.array([flat_features(x) for x in B_test_X])


# %%
from sklearn.linear_model import LogisticRegression


def train_eval(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=4000, C=1.0).fit(X_train, y_train)
    return clf, clf.score(X_test, y_test)


clf_sph,  sph_test_acc      = train_eval(A_train_Fsph,  A_train_y, A_test_Fsph,  A_test_y)
sph_transfer_acc            = clf_sph.score(B_test_Fsph, B_test_y)
_,        sph_B_indomain    = train_eval(B_train_Fsph,  B_train_y, B_test_Fsph,  B_test_y)

clf_flat, flat_test_acc     = train_eval(A_train_Fflat, A_train_y, A_test_Fflat, A_test_y)
flat_transfer_acc           = clf_flat.score(B_test_Fflat, B_test_y)
_,        flat_B_indomain   = train_eval(B_train_Fflat, B_train_y, B_test_Fflat, B_test_y)


print()
print(f"{'pipeline':<28} {'A→A (in-domain)':>16} {'A→B (transfer)':>16} {'B→B (upper)':>14}")
print("-" * 78)
print(f"{'Sphere-aware (band-pass MF)':<28} {sph_test_acc:>16.3f} {sph_transfer_acc:>16.3f} {sph_B_indomain:>14.3f}")
print(f"{'Flat (lat-lon matched filter)':<28} {flat_test_acc:>16.3f} {flat_transfer_acc:>16.3f} {flat_B_indomain:>14.3f}")


# %% [markdown]
# ## 5. Headline figure

# %%
labels = ["A → A\n(in-domain)", "A → B\n(transfer, no retraining)", "B → B\n(in-domain upper bound)"]
sph_vals  = [sph_test_acc, sph_transfer_acc, sph_B_indomain]
flat_vals = [flat_test_acc, flat_transfer_acc, flat_B_indomain]

x = np.arange(len(labels))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - w/2, sph_vals, w,
        label="Sphere-aware (HEALPix sphere-harmonic matched filter)",
        color="#2469C3")
ax.bar(x + w/2, flat_vals, w,
        label="Flat (lat-lon equator-shape matched filter)",
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
              "Domain B: climate-like, features confined to |lat| ≥ 50° + cos²-lat baseline.\n"
              "Both pipelines train on Domain A and apply without retraining to Domain B.",
              fontsize=10)
ax.legend(loc="lower left", fontsize=9)

fig.tight_layout()
fig.savefig(IMG_DIR / "cross_discipline_transfer.png",
            dpi=130, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 6. What this shows
#
# **The headline.** A sphere-harmonic matched filter trained on
# Domain A (cosmology-like, features at uniformly random sphere
# locations) transfers to Domain B (climate-like, features at high
# latitudes, cosine-of-latitude SST baseline) with the same accuracy
# it obtains in-domain on Domain A — even though the classifier never
# saw a Domain B sample. The flat lat-lon matched filter does not
# transfer cleanly: its equator-shape template was trained on
# A's mix of latitudes (where the average rendered shape is roughly
# circular), but Domain B's polar features stretch ~3× in longitude
# on the lat-lon raster, so the template-match response collapses
# and the classifier falls toward chance.
#
# **Why this is the positive case for spherical ML.** Notebook 04
# made the *negative* case at the same physical level: a flat
# matched filter trained on equator-shape MHWs collapsed to chance
# (0.500, F1=0) at 70–80°. This notebook generalises the same
# property *across pairs of disciplines*: as long as both meet on
# the same HEALPix substrate, a sphere-aware matched filter — and,
# by extension, sphere-aware deep models like DeepSphere graph CNNs
# or `foscat` scattering networks — gives the same answer for the
# same physical structure regardless of where on the sphere it sits,
# and a model trained in one discipline (cosmology, atmospheric
# river detection on ClimateNet, MHW detection on Copernicus Marine
# SST, …) carries to another *for free* through the substrate.
# Lat-lon-flat models need domain-specific retraining each time the
# latitude-distribution of the features changes.
#
# **Why this matters for biodiversity / Copernicus / DestinE
# integration.** The astrophysics ML stack (DeepSphere, foscat,
# healpy) is the most mature sphere-aware ML stack on the planet.
# Climate ML is catching up via DLWP-HEALPix and the GRID4EARTH
# Ellipsoidal-HEALPix proposal. The result above says: investments
# in sphere-aware models from astrophysics carry directly over to
# climate, biodiversity, and Earth observation — provided everyone
# meets on the same HEALPix substrate. That is the operational
# claim of this repository made concrete at the model-transfer
# level. The same argument applies one rung up the ladder: a
# `foscat`-trained scattering network on cosmological maps, or a
# DeepSphere model trained on ClimateNet, can act as a feature
# extractor for biodiversity-relevant fields like Copernicus Marine
# SST aggregated to HEALPix (notebook 05) without redoing the
# training from scratch.
#
# **Honest scope.** The pipelines here are deliberately the simplest
# end-to-end transfer experiment that isolates the substrate-
# dependence — a sphere-harmonic band-pass matched filter and its
# lat-lon counterpart, both with identical `(max, mean, std)`
# feature structure and a logistic-regression head. This isolates
# the substrate from the model class and lets the geometric
# mechanism — *rotation equivariance vs translation equivariance* —
# do the explanatory work. The same property — sphere-aware
# operators transfer across disciplines on a shared substrate, lat-
# lon-flat operators do not — is exactly what `foscat`-style
# scattering networks and DeepSphere-style graph CNNs deliver on
# real cosmology / climate data, with much richer learned
# representations.

# %% [markdown]
# ## References
#
# - Górski, K. M. *et al.* (2005) "HEALPix: A Framework for High-
#   Resolution Discretization and Fast Analysis of Data Distributed
#   on the Sphere." *ApJ* **622**: 759.
# - Cohen, T. *et al.* (2018) "Spherical CNNs." *ICLR*.
# - Defferrard, M. *et al.* (2020) "DeepSphere: a graph-based
#   spherical CNN." *ICLR*.
# - Perraudin, N. *et al.* (2019) "DeepSphere: efficient spherical
#   convolutional neural network with HEALPix sampling for
#   cosmological applications." *Astronomy and Computing* **27**:
#   130–146.
# - Karlbauer, M. *et al.* (2024) "Advancing parsimonious deep
#   learning weather prediction using the HEALPix mesh." *J. Adv.
#   Model. Earth Syst.*
# - Delouis, J.-M. *et al.* — `foscat` scattering networks on
#   HEALPix (FIESTA stack).
