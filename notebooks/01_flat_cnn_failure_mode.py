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
# # Part 1: What a flat CNN sees on a global raster
#
# **Why this notebook exists.** A convolutional neural network (CNN)
# works because it assumes **translation equivariance**: the same
# learnable kernel sees the same physical feature whether the feature
# is at pixel `(10, 10)` or pixel `(300, 300)`. That assumption holds
# perfectly for a photograph, where every pixel covers the same area
# of the world.
#
# It does **not** hold for a global Earth-observation product
# unrolled onto a regular latitude–longitude raster. The same physical
# feature on the sphere — a region of given angular size — shows up
# as a wildly different *pixel-shape* depending on its latitude.
# Equator pixels are large; polar pixels are small; a 3×3 kernel at
# 70°N covers a tall thin strip of geography while the same kernel at
# the equator covers a near-square. A CNN trained on lat-lon-projected
# global data therefore learns to treat the *same physical feature* as
# different at different latitudes — without anyone telling it to.
#
# This notebook makes that failure mode visible. We construct a single
# physical feature (a spherical cap of fixed angular radius), place it
# at three latitudes — 0°N, 40°N, 70°N — and render it two ways:
#
# 1. As a **lat-lon raster** (the pixels a flat CNN ingests)
# 2. As **the sphere reality** (orthographic projection centred on the
#    feature — what a spherical CNN sees as input)
#
# The contrast is the case for spherical ML.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

LATS_OF_INTEREST = [0, 40, 70]
LON_C = 0.0
ANGULAR_RADIUS_DEG = 10.0

# %% [markdown]
# ## A spherical cap of fixed angular radius
#
# A spherical cap is the region of the sphere within angular distance
# α of a centre point. We use α = 10° at three latitudes. Every cap
# is the **same physical feature**: same area on the sphere
# (~3.83 × 10⁶ km² at α = 10°), same angular shape, same everything
# that a rotation-equivariant model should treat identically.

# %%
def is_in_spherical_cap(lat, lon, lat_c, lon_c, alpha_deg):
    """True where (lat, lon) is within great-circle angular distance
    `alpha_deg` of (lat_c, lon_c). Inputs in degrees, broadcasts."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    lat_c_r = np.radians(lat_c)
    lon_c_r = np.radians(lon_c)
    cos_d = (np.sin(lat_c_r) * np.sin(lat_r)
             + np.cos(lat_c_r) * np.cos(lat_r)
             * np.cos(lon_r - lon_c_r))
    cos_d = np.clip(cos_d, -1, 1)
    return np.degrees(np.arccos(cos_d)) <= alpha_deg


# Reference grid (high-density lat-lon for both renderings)
LATS_GRID = np.arange(-90, 90.5, 0.5)
LONS_GRID = np.arange(-180, 180.5, 0.5)
LON_G, LAT_G = np.meshgrid(LONS_GRID, LATS_GRID)

# %% [markdown]
# ## What the CNN sees vs what the sphere actually is
#
# Top row — **lat-lon raster** (the pixels a flat CNN ingests).
# The cap at 0°N looks roughly circular; at 40°N it's E–W stretched;
# at 70°N it's a near-horizontal strip. Different shape per latitude
# even though the underlying physical feature is identical.
#
# Bottom row — **sphere reality** (orthographic projection centred on
# the cap). Every cap is a true circle. This is what a spherical CNN,
# operating directly on the sphere, sees as its input.

# %%
fig = plt.figure(figsize=(15, 9))

# Row 1 — lat-lon raster (PlateCarree, what a flat CNN ingests)
for col, lat_c in enumerate(LATS_OF_INTEREST):
    ax = fig.add_subplot(2, 3, col + 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.5")
    ax.gridlines(linewidth=0.3, color="0.7", alpha=0.5)
    in_cap = is_in_spherical_cap(
        LAT_G, LON_G, lat_c, LON_C, ANGULAR_RADIUS_DEG,
    ).astype(float)
    in_cap[in_cap == 0] = np.nan
    ax.pcolormesh(LON_G, LAT_G, in_cap, transform=ccrs.PlateCarree(),
                  cmap="Reds", vmin=0, vmax=1, alpha=0.7)
    ax.scatter([LON_C], [lat_c], marker="+", s=80, c="black",
               transform=ccrs.PlateCarree(), zorder=5)
    ax.set_title(
        f"Lat-lon raster — cap at {lat_c}°N\n"
        "(what a flat CNN ingests as pixels)",
        fontsize=10,
    )

# Row 2 — orthographic centred on the cap (sphere reality)
for col, lat_c in enumerate(LATS_OF_INTEREST):
    ax = fig.add_subplot(2, 3, col + 4,
                         projection=ccrs.Orthographic(LON_C, lat_c))
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.5")
    ax.gridlines(linewidth=0.3, color="0.7", alpha=0.5)
    in_cap = is_in_spherical_cap(
        LAT_G, LON_G, lat_c, LON_C, ANGULAR_RADIUS_DEG,
    ).astype(float)
    in_cap[in_cap == 0] = np.nan
    ax.pcolormesh(LON_G, LAT_G, in_cap, transform=ccrs.PlateCarree(),
                  cmap="Reds", vmin=0, vmax=1, alpha=0.7)
    ax.scatter([LON_C], [lat_c], marker="+", s=80, c="black",
               transform=ccrs.PlateCarree(), zorder=5)
    ax.set_title(
        f"Sphere reality — cap at {lat_c}°N\n"
        "(what a spherical CNN ingests)",
        fontsize=10,
    )

fig.suptitle(
    "Same physical feature, two views — flat raster distorts, "
    "sphere preserves shape",
    fontsize=14, fontweight="bold", y=1.00,
)
plt.tight_layout()
plt.savefig("../images/flat_cnn_failure_mode.png", dpi=150,
            bbox_inches="tight")
plt.show()

# %% [markdown]
# ## What this means for training
#
# A CNN trained on lat-lon-projected global data is shown the **same
# physical feature** in radically different shapes depending on
# latitude. The model has three options for handling this:
#
# 1. **Memorise per-latitude shape** — needs much more training data,
#    and the polar / equatorial regimes are systematically
#    under-represented in most globally-distributed labels.
# 2. **Learn the projection-induced distortion as if it were signal**
#    — the model "explains" the latitude-dependent feature variance
#    with phantom dependencies on geography that don't exist
#    physically.
# 3. **Use a sphere-aware architecture** — let the CNN's
#    equivariance match the symmetry the data actually has.
#
# Option (3) is what spherical ML does. Notebook 02 shows what
# rotation equivariance on the sphere actually means; notebook 03
# explains why HEALPix is the right substrate for it.
#
# **Why this matters for biodiversity, EO, and Copernicus.** Every
# global stack — biodiversity occurrences from GBIF, ERA5 reanalysis,
# Copernicus Marine SST, Destination Earth model output — that is
# combined on a flat lat-lon grid inherits this latitude-dependent
# distortion in every layer. ML pipelines built on top inherit it
# silently. The sphere-aware fix is what GRID4EARTH targets and what
# this repository operationalises with a worked replication of
# DeepSphere on Copernicus atmospheric-river data, plus a marine
# heatwave detection on Copernicus Marine SST tied to documented
# biodiversity impacts.
