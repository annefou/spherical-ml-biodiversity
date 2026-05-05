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
# # Part 2: Rotation equivariance — what spherical CNNs preserve and flat CNNs don't
#
# Notebook 01 showed the failure mode: a flat CNN ingests a
# **lat-lon raster** in which the *same physical feature* on the
# sphere appears as a *different pixel-shape* depending on latitude.
# This notebook makes the underlying principle precise — and shows
# what a spherical CNN does differently.
#
# ## The equivariance principle
#
# A function $f$ is **equivariant** under a transformation $T$ when
#
# $$f(T(x)) = T(f(x)).$$
#
# In English: applying $T$ to the input and then computing $f$ is the
# same as computing $f$ first and then applying $T$ to the output.
# Convolutional neural networks work because their convolution layer
# $f_{\text{conv}}$ is **equivariant under translation**: if you shift
# the input image by $(dx, dy)$, the output feature map shifts by the
# same $(dx, dy)$. That is *why* a feature learned at one image
# position generalises to other positions.
#
# **The natural equivariance for global Earth data is rotation, not
# translation.** Two physical features on the sphere that are
# rotations of each other (e.g. an atmospheric river over the
# California coast vs the same physical event hitting Iberia) should
# elicit the same network response. But a translation-equivariant
# flat CNN on a lat-lon raster does not deliver this.
#
# This notebook shows why mechanically — by demonstrating that a
# rotation on the sphere is **not** a translation on the lat-lon
# raster, and that a "feature detector" matched to a low-latitude
# example fails to fire on the same physical feature at high
# latitudes.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d

ANGULAR_RADIUS_DEG = 10.0
LON_C = 0.0


def is_in_spherical_cap(lat, lon, lat_c, lon_c, alpha_deg):
    """True where (lat, lon) is within great-circle distance alpha_deg
    of (lat_c, lon_c). Inputs in degrees."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    lat_c_r = np.radians(lat_c)
    lon_c_r = np.radians(lon_c)
    cos_d = (np.sin(lat_c_r) * np.sin(lat_r)
             + np.cos(lat_c_r) * np.cos(lat_r)
             * np.cos(lon_r - lon_c_r))
    cos_d = np.clip(cos_d, -1, 1)
    return np.degrees(np.arccos(cos_d)) <= alpha_deg


# Lat-lon raster grid — typical EO product resolution
LATS_GRID = np.arange(-90, 90.5, 0.5)
LONS_GRID = np.arange(-180, 180.5, 0.5)
LON_G, LAT_G = np.meshgrid(LONS_GRID, LATS_GRID)


def cap_raster(lat_c, lon_c=LON_C):
    """Boolean lat-lon raster (1 inside the cap, 0 outside)."""
    return is_in_spherical_cap(
        LAT_G, LON_G, lat_c, lon_c, ANGULAR_RADIUS_DEG,
    ).astype(float)


# %% [markdown]
# ## Demonstration 1 — rotation on the sphere is *not* translation on the raster
#
# Take two caps that are simply rotations of each other on the
# sphere: one at (0°N, 0°E), and one at (60°N, 0°E) — the second is
# the first rotated by 60° around the y-axis. On the sphere, these
# are *the same feature in two locations*. On the lat-lon raster,
# they have different shapes. Translating the 0°N cap raster by 60°
# of latitude produces something that does **not** match the actual
# 60°N cap.

# %%
cap_0 = cap_raster(0.0)
cap_60 = cap_raster(60.0)

# Translate the 0°N raster up by 60° (= 120 rows at 0.5° resolution)
shift_rows = int(60.0 / 0.5)
cap_0_translated = np.roll(cap_0, shift_rows, axis=0)

# Where do the actual 60°N cap and the "translated 0°N cap" disagree?
mismatch = np.abs(cap_60 - cap_0_translated)

fig = plt.figure(figsize=(15, 4.5))

ax_a = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
ax_a.set_global()
ax_a.coastlines(linewidth=0.4, color="0.5")
m = cap_0.copy()
m[m == 0] = np.nan
ax_a.pcolormesh(LON_G, LAT_G, m, transform=ccrs.PlateCarree(),
                cmap="Reds", vmin=0, vmax=1, alpha=0.7)
ax_a.set_title("Cap at 0°N\n(reference)", fontsize=10)

ax_b = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
ax_b.set_global()
ax_b.coastlines(linewidth=0.4, color="0.5")
m = cap_0_translated.copy()
m[m == 0] = np.nan
ax_b.pcolormesh(LON_G, LAT_G, m, transform=ccrs.PlateCarree(),
                cmap="Greys", vmin=0, vmax=1, alpha=0.55)
m = cap_60.copy()
m[m == 0] = np.nan
ax_b.pcolormesh(LON_G, LAT_G, m, transform=ccrs.PlateCarree(),
                cmap="Reds", vmin=0, vmax=1, alpha=0.55)
ax_b.set_title("Translated 0°N raster (grey)\n"
               "vs actual 60°N cap (red)\n"
               "— they don't match",
               fontsize=10)

ax_c = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
ax_c.set_global()
ax_c.coastlines(linewidth=0.4, color="0.5")
m = mismatch.copy()
m[m == 0] = np.nan
ax_c.pcolormesh(LON_G, LAT_G, m, transform=ccrs.PlateCarree(),
                cmap="Purples", vmin=0, vmax=1, alpha=0.7)
ax_c.set_title("Mismatch: |translated − actual|\n"
               "(purple = pixels where the two disagree)",
               fontsize=10)

fig.suptitle("Rotation on the sphere is not translation on the raster — "
             "translation-equivariant CNNs cannot capture rotation symmetry",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../images/rotation_not_translation.png", dpi=150,
            bbox_inches="tight")
plt.show()

n_actual = int(cap_60.sum())
n_mismatch = int(mismatch.sum())
print(f"Actual 60°N cap:                 {n_actual:>6,} raster pixels")
print(f"Disagreement with translated 0°N: {n_mismatch:>6,} raster pixels "
      f"({100 * n_mismatch / n_actual:.1f}% of the cap area)")

# %% [markdown]
# Even at a moderate 60°N, the lat-lon raster of the rotated cap
# disagrees with a simple translation of the equator cap by a
# substantial fraction of the cap's pixel count. This disagreement
# grows with latitude. **A translation-equivariant CNN treats these
# as different features**, even though on the sphere they are the
# same.

# %% [markdown]
# ## Demonstration 2 — a "0°N cap detector" cannot detect the same cap at higher latitudes
#
# Imagine training a CNN on a single labelled example — a cap at
# 0°N — and using it as a feature detector via cross-correlation. We
# extract a small lat-lon-raster patch around the 0°N cap and use it
# as a matched filter. Then we apply that filter to lat-lon rasters
# of caps placed at 0°N, 30°N, 60°N, 80°N. The filter's response
# tells us how confidently it recognises the cap at each latitude.
#
# **Spoiler:** the response collapses with latitude — the matched
# filter is fitted to the equator's pixel-shape, and the polar
# pixel-shape is no longer a match.

# %%
# Build the "feature detector" — a small patch around the 0°N cap
PATCH_HALF_DEG = 14  # window half-width in degrees
patch_n = int(2 * PATCH_HALF_DEG / 0.5) + 1  # number of pixels
i_centre_0 = int((0.0 + 90) / 0.5)
j_centre_0 = int((LON_C + 180) / 0.5)
half_n = patch_n // 2
patch_0n = cap_0[
    i_centre_0 - half_n: i_centre_0 + half_n + 1,
    j_centre_0 - half_n: j_centre_0 + half_n + 1,
].copy()

# Centre + zero-mean the filter so cross-correlation reads as similarity
filter_kernel = patch_0n - patch_0n.mean()

# Apply to lat-lon rasters at 0°N, 30°N, 60°N, 80°N
LAT_TEST = [0, 30, 60, 80]
responses = []
for lat_c in LAT_TEST:
    field = cap_raster(lat_c) - patch_0n.mean()
    response = correlate2d(field, filter_kernel, mode="same",
                           boundary="wrap")
    responses.append(response)

response_max_at_lat = [r.max() for r in responses]
response_max_at_lat = np.array(response_max_at_lat)
response_max_normalised = response_max_at_lat / response_max_at_lat[0]

print("Cross-correlation peak response of the 0°N cap detector:")
print(f"  {'lat':>5} {'peak':>10} {'normalised to 0°N':>20}")
for lat_c, peak, norm in zip(LAT_TEST, response_max_at_lat,
                             response_max_normalised):
    print(f"  {lat_c:>5}° {peak:>10.1f} {norm:>20.3f}")

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8),
                         subplot_kw={"projection": ccrs.PlateCarree()})

for col, lat_c in enumerate(LAT_TEST):
    # Top: the cap raster (input)
    ax = axes[0, col]
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.5")
    m = cap_raster(lat_c).copy()
    m[m == 0] = np.nan
    ax.pcolormesh(LON_G, LAT_G, m, transform=ccrs.PlateCarree(),
                  cmap="Reds", vmin=0, vmax=1, alpha=0.6)
    ax.set_title(f"Input — cap at {lat_c}°N", fontsize=10)

    # Bottom: the detector's response (cross-correlation map)
    ax = axes[1, col]
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.5")
    r = responses[col]
    # Only show non-trivial response
    r_show = r.copy()
    r_show[r_show < 0.05 * response_max_at_lat.max()] = np.nan
    ax.pcolormesh(LON_G, LAT_G, r_show, transform=ccrs.PlateCarree(),
                  cmap="viridis",
                  vmin=0, vmax=response_max_at_lat.max(), alpha=0.85)
    ax.set_title(f"0°N detector response\n"
                 f"peak {response_max_at_lat[col]:.0f} "
                 f"(× {response_max_normalised[col]:.2f} of 0°N)",
                 fontsize=10)

fig.suptitle(
    "A flat CNN's filter trained at the equator does not detect the "
    "same physical feature at the pole",
    fontsize=12, fontweight="bold", y=1.00,
)
plt.tight_layout()
plt.savefig("../images/equator_detector_fails_at_pole.png", dpi=150,
            bbox_inches="tight")
plt.show()

# %% [markdown]
# The detector's peak response collapses with latitude. The detector
# *was* the equator pixel-shape; the polar pixel-shape no longer
# matches. **The same physical feature on the sphere has been
# rendered into the model's input space as a fundamentally different
# pattern, and the model has no built-in way to know they are the
# same.**

# %% [markdown]
# ## What a spherical CNN does differently
#
# A spherical CNN — DeepSphere (Defferrard et al. 2020,
# arXiv:2012.15000), Cohen-style spherical convolutions
# (Cohen et al. 2018, arXiv:1801.10130) — replaces the
# translation-equivariant convolution with one that is
# **rotation-equivariant on the sphere**:
#
# $$f_{\text{sphere-conv}}(R \cdot x) = R \cdot f_{\text{sphere-conv}}(x)$$
#
# where $R$ is any rotation in $SO(3)$ (e.g. moving a feature from
# California to Iberia). The filter "slides" over the sphere
# maintaining the rotation symmetry of the underlying geometry. A
# single learned filter recognises the same physical feature wherever
# it appears on the globe, with no per-latitude retraining and no
# pixel-shape distortion.
#
# This is exactly the property a flat CNN does **not** have on global
# Earth data, as the demonstrations above showed quantitatively.
#
# **What it costs.** Spherical convolutions are more expensive per
# operation than flat convolutions (Cohen-style: $O(N^{3/2})$ via
# spherical-harmonic transforms; DeepSphere graph-convolution: $O(N)$
# but loses strict equivariance). The DISCO architecture (Ocampo,
# Price, McEwen 2023, arXiv:2209.13603) tightens the efficiency
# / equivariance trade-off.
#
# **Where it shines.** Global, full-sphere tasks at moderate-to-high
# resolution where rotation symmetry is genuine — atmospheric-river
# detection (Defferrard et al. 2020), tropical-cyclone tracking,
# global weather forecasting (Karlbauer et al. 2024,
# doi:10.1029/2023MS004021; Weyn, Durran & Caruana 2020,
# doi:10.1029/2020MS002109), and cosmological full-sky analysis
# (Perraudin, Defferrard et al. 2019,
# doi:10.1016/j.ascom.2019.03.004).
#
# **Where it doesn't help much.** Regional mid-latitude tasks on a
# small domain (the projection distortion is small), coarse-resolution
# global tasks (the rotation-symmetry gain is eaten by noise), and
# tasks whose labels are heavily mid-latitude-biased (most biodiversity
# occurrence data is — GBIF is heavily Europe / North America biased).
# Honest framing: *spherical ML is a real but bounded win, on the
# right kind of task at the right scale.*
#
# Notebook 03 explains why **HEALPix** is the right substrate for
# spherical convolutions — equal-area + iso-latitude rings + NESTED
# bit-shift hierarchical refinement + native sphere-harmonic
# transforms make it the foundation of the geometric-deep-learning
# stack on which DeepSphere, foscat, and DLWP-HEALPix are all built.
