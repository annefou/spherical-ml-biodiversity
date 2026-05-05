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
# # Part 3: Why HEALPix is the right substrate for spherical ML
#
# Notebook 02 motivated rotation-equivariant convolutions on the
# sphere. To make those convolutions actually work — fast, scalable,
# and consistent with the rest of the geometric-deep-learning stack —
# you need the right *substrate*: a way of tiling the sphere into
# discrete cells over which the convolution operates.
#
# **HEALPix** (Górski et al. 2005, [doi:10.1086/427976](https://doi.org/10.1086/427976))
# is the substrate the modern spherical-ML ecosystem is built on:
# DeepSphere (Defferrard et al. 2020, arXiv:2012.15000), DLWP-HEALPix
# weather forecasting (Karlbauer et al. 2024,
# [doi:10.1029/2023MS004021](https://doi.org/10.1029/2023MS004021)),
# `foscat` scattering networks (Delouis et al.), `healpy` for native
# spherical-harmonic transforms. This notebook explains why.
#
# Four properties that together make HEALPix uniquely suited to
# sphere-aware ML:
#
# 1. **Equal-area cells** — every pixel covers the same physical
#    area on the sphere. Loss functions and aggregation operations
#    are not latitude-biased.
# 2. **Iso-latitude rings** — pixels are organised into rings of
#    constant colatitude, which makes spherical-harmonic transforms
#    fast (O(N · √N) at high resolution).
# 3. **NESTED hierarchical refinement via bit-shift** — multi-scale
#    convolutions and pyramid networks are natural; parent and
#    children are integer bit operations.
# 4. **Graph structure with mostly-uniform neighbour count** —
#    DeepSphere's graph CNN exploits this directly.
#
# The visual demos below illustrate the first three. The fourth is
# best understood in code (see DeepSphere's reference implementation
# at [deepsphere-pytorch](https://github.com/deepsphere/deepsphere-pytorch)).

# %%
import cartopy.crs as ccrs
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


# %% [markdown]
# ## Property 1 — Equal-area cells
#
# Every HEALPix cell covers the same area on the sphere by
# construction: 4π R² / N_pix. There are no high-latitude shrinkage
# or polar-cell pathologies. A density estimate, an aggregation, or
# a loss function summed over HEALPix cells is **not biased by
# latitude**.

# %%
NSIDE_DEMO = 32
NPIX_DEMO = hp.nside2npix(NSIDE_DEMO)
R_KM = 6371.0
HEALPIX_CELL_AREA_KM2 = 4 * np.pi * R_KM ** 2 / NPIX_DEMO

print(f"HEALPix nside={NSIDE_DEMO}: {NPIX_DEMO:,} cells globally")
print(f"Each cell area: {HEALPIX_CELL_AREA_KM2:,.0f} km²")
print(f"Latitude-dependent area variation: 0.0 % (equal-area by construction)")

# Compare against a regular lat-lon grid at comparable nominal resolution
LATLON_RES_DEG = np.sqrt(4 * np.pi / NPIX_DEMO) * 180 / np.pi
print()
print(f"For comparison — a regular lat-lon grid at {LATLON_RES_DEG:.2f}° "
      f"nominal resolution:")
for lat_c in [0, 30, 60, 80, 89]:
    cell_area = (
        2 * np.pi * R_KM ** 2 / 360
        * abs(np.sin(np.radians(lat_c + LATLON_RES_DEG / 2))
              - np.sin(np.radians(lat_c - LATLON_RES_DEG / 2)))
        * LATLON_RES_DEG
    )
    print(f"  lat {lat_c:>2}°N : {cell_area:>9,.0f} km²")
print("  Latitude-dependent area variation: SEVERE (a single 'global "
      "grid' contains cells whose physical areas differ by orders of "
      "magnitude)")

# %% [markdown]
# ## Property 2 — Iso-latitude rings: pixels organised by latitude
#
# HEALPix pixels are organised into **rings of constant colatitude**.
# Every pixel in a given ring sits at exactly the same latitude as
# every other pixel in that ring. This is what makes spherical
# harmonic transforms (`healpy.map2alm` / `alm2map`) computationally
# fast: the ring structure factorises the longitude direction,
# enabling FFTs along each ring.
#
# The visual: render every pixel of HEALPix nside=16 coloured by its
# ring index (= unique latitude). Sharp horizontal bands. Compare
# against a hexagonal DGGS like H3, which does not have this
# property (cells in the "same row" are at slightly different
# latitudes, blurring any latitude-banded analysis).

# %%
NSIDE_RINGS = 16
NPIX_RINGS = hp.nside2npix(NSIDE_RINGS)
ring_thetas, ring_phis = hp.pix2ang(
    NSIDE_RINGS, np.arange(NPIX_RINGS), nest=False,
)
ring_lats = 90.0 - np.degrees(ring_thetas)
ring_lons = np.degrees(ring_phis)
ring_lons = np.where(ring_lons > 180, ring_lons - 360, ring_lons)

unique_thetas = np.unique(np.round(ring_thetas, 10))
ring_id = np.searchsorted(unique_thetas, np.round(ring_thetas, 10))

print(f"HEALPix nside={NSIDE_RINGS}: {len(unique_thetas)} distinct "
      f"iso-latitude rings, {NPIX_RINGS:,} pixels total")
print("All pixels in a given ring share the same colatitude to "
      "machine precision.")

# %%
fig, axes = plt.subplots(
    1, 2, figsize=(13, 5),
    subplot_kw={"projection": ccrs.Robinson()},
)

ax = axes[0]
ax.set_global()
ax.coastlines(linewidth=0.4, color="0.5")
ax.gridlines(linewidth=0.3, color="0.7", alpha=0.4)
ax.scatter(ring_lons, ring_lats, c=ring_id, cmap="viridis",
           s=2.5, transform=ccrs.PlateCarree(), alpha=0.85)
ax.set_title(
    f"HEALPix nside={NSIDE_RINGS}: iso-latitude rings\n"
    "(coloured by ring index — sharp horizontal bands)",
    fontsize=11,
)

# H3 hexagonal comparison: cells in the same "row" are NOT at the
# same latitude (hex tessellation breaks iso-latitude alignment).
import h3

ax = axes[1]
ax.set_global()
ax.coastlines(linewidth=0.4, color="0.5")
ax.gridlines(linewidth=0.3, color="0.7", alpha=0.4)

H3_RES_VIS = 2
h3_cells_res0 = h3.get_res0_cells()
h3_cells_res2 = []
for c in h3_cells_res0:
    h3_cells_res2.extend(h3.cell_to_children(c, H3_RES_VIS))
h3_lats = []
h3_lons = []
for c in h3_cells_res2:
    lat, lon = h3.cell_to_latlng(c)
    h3_lats.append(lat)
    h3_lons.append(lon)
h3_lats = np.array(h3_lats)
h3_lons = np.array(h3_lons)
h3_band_idx = np.floor(h3_lats / 10).astype(int)
ax.scatter(h3_lons, h3_lats, c=h3_band_idx, cmap="viridis",
           s=12, transform=ccrs.PlateCarree(), alpha=0.85)
ax.set_title(
    f"H3 res {H3_RES_VIS}: cells coloured by 10° latitude band\n"
    "(no iso-latitude rings — hex tessellation breaks them)",
    fontsize=11,
)

fig.suptitle(
    "HEALPix iso-latitude rings vs H3 hexagonal tessellation — "
    "why HEALPix wins for spherical-harmonic transforms and "
    "latitude-banded analyses",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("../images/iso_latitude_rings.png", dpi=150,
            bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Property 3 — NESTED bit-shift hierarchical refinement
#
# In HEALPix's NESTED ordering, the parent-child relationship between
# resolutions `nside = N` and `nside = 2N` is **pure integer
# arithmetic**:
#
# ```
# parent(pix)        = pix >> 2
# children(pix, k)   = (pix << 2) | k    for k in {0, 1, 2, 3}
# ```
#
# That makes multi-scale operations — pyramid networks, wavelet
# transforms, image-pyramid analogues, and tile-based pipelines on
# Copernicus Zarr stores — *O(1) per cell*. No projection, no
# resampling, no hash lookup, no coordinate conversion. Every spherical
# CNN that operates at multiple resolutions exploits this property.

# %%
PARENT_NSIDE = 8
CHILD_NSIDE = 2 * PARENT_NSIDE  # 16

parent_pix = 42
children_via_bitshift = [(parent_pix << 2) | k for k in range(4)]

print(f"Parent (nside={PARENT_NSIDE}, NESTED):  pix = {parent_pix} "
      f"(binary {parent_pix:08b})")
print(f"Children at nside={CHILD_NSIDE} via bit-shift:")
for k, c in enumerate(children_via_bitshift):
    print(f"  k={k}:  pix = {c:>4}  (binary {c:010b})")

# Verify against healpy
child_thetas, child_phis = hp.pix2ang(
    CHILD_NSIDE, np.array(children_via_bitshift), nest=True,
)
parent_check = hp.ang2pix(
    PARENT_NSIDE, child_thetas, child_phis, nest=True,
)
print(f"\nVerify by computing parent of each child via healpy: "
      f"{parent_check.tolist()}")
assert all(p == parent_pix for p in parent_check)
print(f"All four children resolve back to parent pix {parent_pix} ✓")

# %% [markdown]
# ## Property 4 — Graph structure for DeepSphere-style graph CNNs
#
# DeepSphere (Defferrard et al. 2020) treats the HEALPix mesh as a
# **graph**: every pixel is a node, every pair of neighbouring pixels
# is an edge. Most HEALPix pixels have **8 neighbours**, with a small
# number of "corner" pixels having 7 — almost-uniform graph degree.
# Spectral graph convolutions on this graph are O(N) and approximate
# rotation equivariance well at scale.
#
# Compared with cubed-sphere or icosahedral grids, HEALPix wins on
# (a) equal-area + (b) fast spherical-harmonic transforms via
# iso-latitude rings + (c) hierarchical bit-shift refinement, in a
# single integrated package. Cubed-sphere and icosahedral grids are
# good in their own right (DLWP-cubed-sphere, Cohen 2019 icosahedral
# CNNs) but lack one or more of these properties.

# %% [markdown]
# ## Putting it together — the spherical-ML ecosystem on HEALPix
#
# - **`healpy`** — Python interface to the HEALPix C library; native
#   `hp.ang2pix`, `hp.pix2ang`, `hp.boundaries`, `hp.map2alm`,
#   `hp.alm2map`. The substrate.
# - **`healpix-geo`** (EOPF-DGGS) — adds WGS84-ellipsoid HEALPix via
#   the authalic-sphere mapping (the **GRID4EARTH "Ellipsoidal
#   HEALPix"** path). Cell vertices computed on the actual Earth
#   shape rather than the unit sphere.
# - **DeepSphere** (Defferrard et al. 2020) — graph-based spherical
#   CNN on HEALPix. The most-cited spherical CNN architecture for
#   global Earth science. Used in the AR replication in notebook 04.
# - **DLWP-HEALPix** (Karlbauer et al. 2024) — global medium-range
#   weather forecasting on HEALPix. Demonstrates that the substrate
#   scales to operational AI weather prediction.
# - **`foscat`** (Delouis et al., FIESTA stack) — scattering networks
#   on HEALPix. Wavelet covariance features for global EO data.
# - **DISCO** (Ocampo, Price, McEwen 2023, arXiv:2209.13603) —
#   discrete-continuous spherical convolutions; tightens the
#   efficiency / equivariance trade-off.
#
# These all share HEALPix as the substrate, the `nest=True` ordering
# convention, and the ability to interoperate at the cell-id level.
# That is the integration story the **dggs-biodiversity-bias**
# repository (concept DOI [10.5281/zenodo.19848749](https://doi.org/10.5281/zenodo.19848749))
# argues for. This notebook is the why-HEALPix layer of that case.
#
# Notebook 04 starts the worked replication: DeepSphere
# atmospheric-river detection on ERA5 / HEALPix.
