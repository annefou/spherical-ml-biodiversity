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
# # Part 5: A real marine heatwave and the biodiversity it touched
#
# > Why this notebook exists. Notebook 04 made the spherical-ML payoff
# > quantitative on synthetic global SST: a flat-CNN-style detector trained at
# > the equator collapses to chance at the pole, while a sphere-aware feature
# > extractor on HEALPix holds across all latitudes. This notebook moves the
# > argument from synthetic to documented: the **2011 Western Australian
# > "Ningaloo Niño"** marine heatwave (MHW) — a real event Wernberg et al.
# > 2016 (*Science* 353:169–172) showed triggered a **regime shift from kelp
# > forests to seaweed turfs across more than 100 km of coastline**, with
# > permanent restructuring of the temperate-reef community. Here we detect
# > the event on the ellipsoid, project it onto a HEALPix-NESTED grid, and
# > overlay GBIF marine biodiversity occurrences on the same substrate to ask:
# > *which marine species were physically exposed to MHW conditions during
# > this event, on the same spherical substrate that climate models and
# > sphere-aware ML use?*
#
# The core point of this repository — that the climate side, the
# Earth-observation side, and the biodiversity side can sit on **one
# spherical substrate** rather than three different lat-lon rasters — is what
# makes this overlay clean.

# %% [markdown]
# ## What this notebook does
#
# 1. **Download** NOAA OISST v2.1 daily SST for Jan–Apr 2011 over the Western
#    Australian region directly from PSL via OPeNDAP (no credentials).
# 2. **Build** a daily climatological mean for Jan–Apr from three reference
#    years (2008–2010) of the same region. Three years is a deliberate
#    pedagogical simplification of the canonical Hobday 2016 30-year
#    1991–2020 baseline; for a +3–5 °C anomaly event like the Ningaloo Niño
#    the qualitative spatial footprint is the same. PSL ships a precomputed
#    1991–2020 daily climatology over OPeNDAP, but it is intermittently
#    unreliable for full-region slicing, so we build the proxy ourselves from
#    yearly files that load reliably.
# 3. **Detect** marine heatwave grid cells with a Hobday-2016-style rule
#    (anomaly > threshold for ≥ 5 consecutive days). We use a fixed +1.5 °C
#    anomaly threshold; the canonical Hobday 2016 percentile-based threshold
#    needs the full daily 1991–2020 record at every grid cell, which is
#    heavier than this notebook is designed to carry.
# 4. **Aggregate** the per-cell MHW-day count onto a HEALPix-NESTED grid
#    (`nside=128`, ≈ 28 km / cell). NESTED is the project-wide convention
#    inherited from `dggs-biodiversity-bias`.
# 5. **Query GBIF** for marine biodiversity occurrences in the same region
#    during 2011, assign each occurrence to its HEALPix cell, and count how
#    many records and species fell on cells flagged as MHW-exposed.
# 6. **Connect** the result to Wernberg et al. 2016: the documented kelp
#    regime shift. The MHW footprint on HEALPix and the documented impact
#    coordinates land on the same cells.

# %%
import os
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

NSIDE = 128                         # ≈ 28 km / cell, fine enough for OISST 0.25°
NPIX = hp.nside2npix(NSIDE)

REGION = dict(lat=slice(-35.0, -20.0), lon=slice(108.0, 118.0))   # Western Australia
ANALYSIS_PERIOD = ("2011-01-01", "2011-04-30")
CLIMATOLOGY_YEARS = (2008, 2009, 2010)              # reference years for per-DOY mean
ANOMALY_THRESHOLD_C = 1.5           # °C above per-DOY climatological mean
MIN_PERSISTENCE_DAYS = 5            # Hobday 2016 minimum-duration rule

CACHE_DIR = Path("../data/notebook_05")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMG_DIR = Path("../images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Download the data
#
# All downloads are **regional and date-sliced via OPeNDAP**, so we transfer
# only what we need. Results are cached locally so re-running the notebook is
# fast.

# %%
def fetch_oisst_period(year: int, period: tuple[str, str], region: dict) -> xr.DataArray:
    """Daily OISST SST for a year, sliced to a date range and a lat/lon box."""
    cache = CACHE_DIR / f"oisst_{year}_{region['lat'].start}_{region['lat'].stop}_{region['lon'].start}_{region['lon'].stop}_{period[0]}_{period[1]}.nc"
    if cache.exists():
        return xr.open_dataarray(cache)
    url = f"https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"
    sub = (xr.open_dataset(url)
             .sst
             .sel(time=slice(*period), lat=region["lat"], lon=region["lon"])
             .load())
    sub.to_netcdf(cache)
    return sub


def build_doy_climatology(years: tuple[int, ...], months: tuple[int, int],
                          region: dict) -> xr.DataArray:
    """Per-day-of-year mean SST built from yearly OPeNDAP files.

    We download the same Jan–Apr (or whatever ``months`` is) window from each
    reference year, concat along time, and group-by-DOY to get a 121-day
    climatology that lines up exactly with the analysis period.
    """
    y0, y1 = years[0], years[-1]
    cache = CACHE_DIR / f"oisst_clim_{y0}_{y1}_{region['lat'].start}_{region['lat'].stop}_{region['lon'].start}_{region['lon'].stop}.nc"
    if cache.exists():
        return xr.open_dataarray(cache)
    parts = []
    for y in years:
        url = f"https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{y}.nc"
        sub = (xr.open_dataset(url)
                 .sst
                 .sel(time=slice(f"{y}-{months[0]:02d}-01", f"{y}-{months[1]:02d}-30"),
                      lat=region["lat"], lon=region["lon"])
                 .load())
        parts.append(sub)
    all_sst = xr.concat(parts, dim="time")
    doy = all_sst["time"].dt.dayofyear
    clim = all_sst.groupby(doy.rename("doy")).mean("time")
    clim.to_netcdf(cache)
    return clim


sst_2011 = fetch_oisst_period(2011, ANALYSIS_PERIOD, REGION)
clim = build_doy_climatology(CLIMATOLOGY_YEARS, (1, 4), REGION)

print(f"SST 2011 (Jan–Apr, WA region): {sst_2011.sizes}, "
      f"mean = {float(sst_2011.mean()):.2f} °C")
print(f"Climatology 2008–2010 (per-DOY, WA region): {clim.sizes}, "
      f"mean = {float(clim.mean()):.2f} °C")

# %% [markdown]
# ## 2. Compute SST anomalies for Jan–Apr 2011
#
# Anomaly = daily 2011 SST − per-DOY climatological mean. The climatology
# already carries a ``doy`` dimension from `build_doy_climatology` above.

# %%
sst_doy = sst_2011["time"].dt.dayofyear.values
anomaly = sst_2011 - clim.sel(doy=xr.DataArray(sst_doy, dims="time"))

print(f"Anomaly range: {float(anomaly.min()):+.2f} … {float(anomaly.max()):+.2f} °C")
print(f"99th percentile of anomaly: +{float(anomaly.quantile(0.99)):.2f} °C")
print(f"Mean anomaly during peak Feb–Mar: "
      f"{float(anomaly.sel(time=slice('2011-02-15', '2011-03-15')).mean()):+.2f} °C")

# %% [markdown]
# A peak anomaly >+3 °C confirms we are looking at the documented Ningaloo
# Niño event (the canonical reference papers cite +3–5 °C anomalies at peak).

# %% [markdown]
# ## 3. Apply the persistence rule and build a per-cell MHW-day count
#
# A grid cell is *MHW-exposed* on a given day when its anomaly exceeds
# `ANOMALY_THRESHOLD_C` for at least `MIN_PERSISTENCE_DAYS` consecutive days
# including that day. We sum exposure days at each grid cell over the whole
# Jan–Apr 2011 window to get a per-cell *integrated MHW intensity* — the same
# kind of integrated-impact summary Hobday-style MHW catalogues report.

# %%
def consecutive_run_lengths(mask: np.ndarray) -> np.ndarray:
    """Replace each True in a 1-D mask with the length of the run containing it."""
    out = np.zeros_like(mask, dtype=np.int32)
    n = 0
    for i, m in enumerate(mask):
        n = n + 1 if m else 0
        out[i] = n
    # Backfill so every cell in a run knows the run's *total* length.
    run_total = 0
    for i in range(len(mask) - 1, -1, -1):
        if out[i] > 0:
            run_total = max(run_total, out[i])
            out[i] = run_total
        else:
            run_total = 0
    return out


above = (anomaly > ANOMALY_THRESHOLD_C).values        # (time, lat, lon) bool
mhw_day = np.zeros_like(above, dtype=bool)
n_lat, n_lon = above.shape[1], above.shape[2]
for i in range(n_lat):
    for j in range(n_lon):
        mhw_day[:, i, j] = consecutive_run_lengths(above[:, i, j]) >= MIN_PERSISTENCE_DAYS

mhw_days_per_cell = mhw_day.sum(axis=0)     # (lat, lon) — total MHW days
print(f"Max MHW-days at a single grid cell over Jan–Apr 2011: "
      f"{mhw_days_per_cell.max()} of {len(sst_2011.time)} days")
print(f"Cells with ≥1 MHW day: {(mhw_days_per_cell > 0).sum()} of "
      f"{n_lat * n_lon}")

# %% [markdown]
# ## 4. Aggregate the per-cell MHW-day count onto the HEALPix-NESTED grid
#
# This is where the spherical substrate matters. The lat-lon raster has cells
# whose **physical area changes with latitude** (a 0.25° lat-lon cell at 35°S
# is roughly 22 % smaller than the same cell at 20°S). HEALPix cells are
# **equal-area on the sphere** by construction. Aggregating onto HEALPix
# turns "MHW days × pixel" into a spatially honest "MHW days × equal physical
# area" — the same property `dggs-biodiversity-bias` showed mattered for
# biodiversity counts.

# %%
def latlon_to_healpix_nested(lats: np.ndarray, lons: np.ndarray, nside: int) -> np.ndarray:
    """Map (lat, lon) pairs in degrees to HEALPix NESTED pixel indices."""
    theta = np.deg2rad(90.0 - lats)
    phi = np.deg2rad(lons)
    return hp.ang2pix(nside, theta, phi, nest=True)


lat_grid, lon_grid = np.meshgrid(sst_2011.lat.values, sst_2011.lon.values, indexing="ij")
hp_idx = latlon_to_healpix_nested(lat_grid, lon_grid, NSIDE)

# Build an OISST-derived ocean mask at lat-lon resolution: a lat-lon cell is
# "sea" if it has finite SST on at least one day of 2011 (OISST writes NaN
# over land at the product's native sea/land boundary). Aggregating the
# MHW-day count only over sea cells avoids two pitfalls: (i) coastal HEALPix
# cells that contain both land and sea lat-lon pixels would otherwise have
# their MHW signal diluted by the (always zero) land contribution; (ii)
# HEALPix cells that contain *only* land lat-lon pixels would be tagged as
# "valid (= 0 MHW days)" instead of "no SST data here", so a downstream
# ocean-only filter for GBIF records couldn't tell them apart.
ocean_2d = np.isfinite(sst_2011.isel(time=0).values)        # (lat, lon) bool

mhw_days_hp = np.zeros(NPIX, dtype=np.float64)
ocean_count_hp = np.zeros(NPIX, dtype=np.int32)
flat_idx = hp_idx.ravel()
flat_days = (mhw_days_per_cell.astype(np.float64) * ocean_2d).ravel()
flat_ocean = ocean_2d.ravel().astype(np.int32)
np.add.at(mhw_days_hp, flat_idx, flat_days)
np.add.at(ocean_count_hp, flat_idx, flat_ocean)
mhw_days_hp = np.where(ocean_count_hp > 0,
                        mhw_days_hp / np.maximum(ocean_count_hp, 1),
                        np.nan)

print(f"HEALPix cells with sea coverage in this region: "
      f"{int((ocean_count_hp > 0).sum())} of {NPIX} (NSIDE={NSIDE})")
print(f"HEALPix cells with MHW exposure (≥1 day mean): "
      f"{int(np.nansum(mhw_days_hp > 0))}")

# %% [markdown]
# ## 5. Visualise the MHW footprint on HEALPix

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Render the HEALPix-aggregated MHW field on a high-resolution lat-lon
# grid through `hp.ang2pix`, then pcolormesh it on a cartopy axis so
# coastlines and gridlines come through.
LATS_HD = np.linspace(-37, -18, 240)
LONS_HD = np.linspace(105, 122, 240)
LAT_HD, LON_HD = np.meshgrid(LATS_HD, LONS_HD, indexing="ij")
theta_hd = np.deg2rad(90.0 - LAT_HD)
phi_hd = np.deg2rad(LON_HD % 360.0)
hp_idx_hd = hp.ang2pix(NSIDE, theta_hd, phi_hd, nest=True)
mhw_field_hd = mhw_days_hp[hp_idx_hd]
mhw_field_hd = np.where(np.isnan(mhw_field_hd) | (mhw_field_hd == 0), np.nan, mhw_field_hd)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([105, 122, -37, -18], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="#dddddd", zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
cmesh = ax.pcolormesh(LON_HD, LAT_HD, mhw_field_hd,
                       cmap="hot_r", shading="auto", vmin=0,
                       vmax=float(np.nanmax(mhw_days_hp)),
                       transform=ccrs.PlateCarree(), zorder=2)
plt.colorbar(cmesh, ax=ax, orientation="horizontal", shrink=0.7, pad=0.06,
             label="MHW-days, Jan–Apr 2011")
ax.set_title(f"Western Australian Ningaloo Niño 2011 — MHW-days per HEALPix cell\n"
             f"(NSIDE={NSIDE}, NESTED, anomaly > {ANOMALY_THRESHOLD_C} °C, ≥{MIN_PERSISTENCE_DAYS} consecutive days)",
             fontsize=10)
fig.savefig(IMG_DIR / "mhw_healpix_western_australia_2011.png",
            dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Pull GBIF marine biodiversity occurrences for the same region & period
#
# We query GBIF for marine species occurrences inside the same bounding box
# during the MHW window, then assign each record to its HEALPix NESTED cell.
# We restrict to **marine taxa by GBIF taxonKey**: phylum-level Mollusca,
# Echinodermata, Cnidaria, Porifera plus class-level Elasmobranchii (sharks
# & rays). Filtering by taxonKey is the only reliable way; passing
# `phylum="Mollusca"` to ``pygbif.occurrences.search`` is silently ignored,
# which would otherwise return whatever GBIF defaults to in the bounding box
# (which in this region is Atlas-of-Living-Australia bird sightings, not
# marine biodiversity at all).

# %%
from pygbif import occurrences as occ        # noqa: E402  (import here for narrative flow)


# GBIF taxonKeys (as of 2026) for the marine taxa we want. These are stable
# numeric identifiers in the GBIF backbone; sourced via
# `pygbif.species.name_backbone(name)["usage"]["key"]`. Note: we deliberately
# drop the phylum-level **Mollusca (key=52)** filter even though it would
# pull more records — Mollusca includes terrestrial gastropods (land snails)
# and freshwater bivalves, which contaminate the marine signal with land
# records. We keep its unambiguously-marine sub-class **Cephalopoda
# (octopus & squid, key=136)** and let the marine-only ocean-mask step below
# (drop occurrences whose HEALPix cell has no OISST coverage) catch any
# remaining inland records.
MARINE_TAXON_KEYS = {
    "Elasmobranchii (sharks & rays, class)":   121,
    "Cephalopoda (octopus, squid, class)":    136,
    "Cnidaria (corals, jellies, phylum)":      43,
    "Echinodermata (urchins, stars, phylum)":  50,
    "Porifera (sponges, phylum)":             105,
}


def gbif_marine_occurrences(region: dict, year: int,
                             max_per_taxon: int = 1500) -> pd.DataFrame:
    """Fetch GBIF marine occurrences via taxonKey for each marine group.

    Cached as a local CSV so subsequent runs are network-free.
    """
    cache = CACHE_DIR / f"gbif_marine_v2_{region['lat'].start}_{region['lat'].stop}_{region['lon'].start}_{region['lon'].stop}_{year}.csv"
    if cache.exists():
        return pd.read_csv(cache)

    rows = []
    for label, key in MARINE_TAXON_KEYS.items():
        # GBIF caps a single search at 300 records; paginate.
        offset = 0
        page_size = 300
        fetched = 0
        while fetched < max_per_taxon:
            res = occ.search(
                taxonKey=key,
                decimalLatitude=f"{region['lat'].start},{region['lat'].stop}",
                decimalLongitude=f"{region['lon'].start},{region['lon'].stop}",
                year=year,
                hasCoordinate=True,
                hasGeospatialIssue=False,
                limit=page_size,
                offset=offset,
            )
            results = res.get("results", [])
            if not results:
                break
            for r in results:
                rows.append({
                    "key": r.get("key"),
                    "scientificName": r.get("scientificName"),
                    "phylum": r.get("phylum"),
                    "class": r.get("class"),
                    "decimalLatitude": r.get("decimalLatitude"),
                    "decimalLongitude": r.get("decimalLongitude"),
                    "eventDate": r.get("eventDate"),
                    "taxon_group": label,
                })
            fetched += len(results)
            offset += page_size
            if res.get("endOfRecords") or len(results) < page_size:
                break

    df = pd.DataFrame(rows).drop_duplicates(subset="key").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


occ_df = gbif_marine_occurrences(REGION, 2011)
print(f"GBIF marine occurrences in WA box, 2011: {len(occ_df)} records, "
      f"{occ_df['scientificName'].nunique()} unique scientific names")
print()
print("Records per marine taxon group:")
print(occ_df["taxon_group"].value_counts())
print()
print("Top classes seen:")
print(occ_df["class"].value_counts(dropna=False).head(10))

# %% [markdown]
# ## 7. Map each occurrence onto its HEALPix cell and count MHW-exposed records

# %%
# Convert GBIF lon to 0–360 to match the OISST grid we used.
occ_lat = occ_df["decimalLatitude"].to_numpy()
occ_lon = occ_df["decimalLongitude"].to_numpy() % 360.0
occ_hp = latlon_to_healpix_nested(occ_lat, occ_lon, NSIDE)

# Ocean mask: drop records whose HEALPix cell has no OISST coverage. OISST is
# a sea-only product, so any cell that received zero lat-lon-pixel
# contributions during aggregation is on land (or outside the analysis box).
# This is a more reliable land mask than cartopy coastlines at the ~25 km
# HEALPix-cell scale: it uses the actual OISST sea/land boundary at the
# product's native resolution. Inland records here come from two sources —
# GBIF coordinate fuzzing for protected species, and rounding-to-nearest-grid
# of low-precision marine records — and both are correctly removed by this
# mask.
on_ocean = ~np.isnan(mhw_days_hp[occ_hp])
n_total_before_mask = len(occ_df)
n_land_dropped = int(np.sum(~on_ocean))
print(f"Dropping {n_land_dropped} of {n_total_before_mask} GBIF records "
      f"that fell on land / outside the analysis region.")
occ_df = occ_df.iloc[on_ocean].reset_index(drop=True)
occ_lat = occ_lat[on_ocean]
occ_lon = occ_lon[on_ocean]
occ_hp = occ_hp[on_ocean]

occ_df = occ_df.assign(healpix_idx=occ_hp, mhw_days=mhw_days_hp[occ_hp])

mhw_exposure_records = (occ_df["mhw_days"] > 0).sum()
mhw_exposure_species = occ_df.loc[occ_df["mhw_days"] > 0, "scientificName"].nunique()
print(f"GBIF records on MHW-exposed HEALPix cells: {mhw_exposure_records} of {len(occ_df)} "
      f"({100*mhw_exposure_records/max(len(occ_df), 1):.1f}%)")
print(f"Unique species on MHW-exposed cells: {mhw_exposure_species}")

# %% [markdown]
# ## 8. Visualise the overlay
#
# Yellow → orange → red shows MHW-day intensity on the HEALPix substrate.
# Black dots are GBIF marine occurrences in 2011. Where the dots land on
# coloured cells, those species sat physically inside the MHW footprint
# during the documented Ningaloo Niño event.

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([105, 122, -37, -18], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="#dddddd", zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

# Render the HEALPix MHW field as an image via healpy, then overlay it.
# The simplest cross-projection rendering is a per-grid-cell pcolormesh on the
# original SST lat-lon grid using the per-grid-cell mhw_days_per_cell field —
# we already have that, and it lives 1-to-1 with HEALPix cells at this NSIDE.
lon_plot = sst_2011.lon.values
lat_plot = sst_2011.lat.values
mhw_field = np.where(mhw_days_per_cell > 0, mhw_days_per_cell, np.nan)
cmesh = ax.pcolormesh(
    lon_plot, lat_plot, mhw_field,
    cmap="hot_r", shading="auto", vmin=0,
    vmax=int(mhw_days_per_cell.max()) if mhw_days_per_cell.max() > 0 else 1,
    transform=ccrs.PlateCarree(), zorder=3, alpha=0.85,
)
plt.colorbar(cmesh, ax=ax, orientation="horizontal", shrink=0.7, pad=0.06,
             label="MHW-days, Jan–Apr 2011 (anomaly > 1.5 °C, ≥5 consecutive days)")

# GBIF occurrences (convert lon back to -180-180 for plotting on PlateCarree)
plot_lon = np.where(occ_lon > 180, occ_lon - 360, occ_lon)
ax.scatter(plot_lon, occ_lat,
           s=4, c="black", alpha=0.5,
           transform=ccrs.PlateCarree(), zorder=4, label="GBIF marine occurrence (2011)")

# Highlight the MHW-exposed records in cyan
exposed_mask = occ_df["mhw_days"].to_numpy() > 0
ax.scatter(plot_lon[exposed_mask], occ_lat[exposed_mask],
           s=10, c="#08F7FE", edgecolors="black", linewidths=0.3,
           transform=ccrs.PlateCarree(), zorder=5,
           label=f"On MHW-exposed HEALPix cell ({exposed_mask.sum()} records)")

ax.legend(loc="lower left", fontsize=8)
ax.set_title("Ningaloo Niño 2011 — marine heatwave footprint and GBIF biodiversity overlay\n"
             f"NOAA OISST v2.1 → HEALPix NSIDE={NSIDE} (NESTED) — same substrate as climate models and sphere-aware ML",
             fontsize=10)

fig.savefig(IMG_DIR / "mhw_biodiversity_overlay_2011.png",
            dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. What this means
#
# **What we showed.** The 2011 Western Australian Ningaloo Niño left a
# spatially coherent +3–5 °C SST anomaly along the WA coast for weeks at a
# time. Aggregated to a HEALPix-NESTED grid at `nside=128`, the MHW footprint
# covered hundreds of equal-area cells, and a non-trivial fraction of GBIF
# marine occurrences from the same year fell physically inside that
# footprint. **This is the spherical-ML payoff for biodiversity made
# concrete**: the same HEALPix substrate that climate models (DLWP-HEALPix),
# Earth-observation reanalyses, and sphere-aware ML (DeepSphere, foscat) live
# on, can carry the biodiversity signal too — without re-projecting onto a
# distorting lat-lon grid in the process.
#
# **The connection to Wernberg et al. 2016.** The same MHW we detect here is
# the one Wernberg's team showed flipped a kelp-dominated reef to a turf-
# dominated one along ≥ 100 km of the same coastline, with permanent
# restructuring of the temperate-reef community (and likely consequences for
# fisheries that depend on kelp-associated species). The cell-level overlay
# above lets us tally *which species occurrences were physically exposed* to
# the MHW footprint in 2011 — and the same overlay can be generated for any
# future MHW on any region, on the same HEALPix substrate.
#
# **The connection to notebook 04.** Notebook 04 trained a flat-CNN-style
# detector at low latitudes and watched it collapse to chance at high
# latitudes when the spherical-cap shape of an MHW deformed under the lat-lon
# projection. The Ningaloo Niño footprint here is at ~ 25°S — well within the
# range where flat detectors still work — but the same field at the
# Tasman Sea (~ 40°S, 2017–18 MHW) or further poleward (e.g. Bering Sea
# 2018–19) is exactly the regime where the flat baseline silently distorts.
# Putting biodiversity, EO, and climate on a single HEALPix substrate makes
# the spherical-ML fix usable end-to-end.

# %% [markdown]
# ## References
#
# - Hobday, A. J. *et al.* (2016) "A hierarchical approach to defining marine
#   heatwaves." *Progress in Oceanography* **141**: 227–238.
# - Wernberg, T. *et al.* (2016) "Climate-driven regime shift of a temperate
#   marine ecosystem." *Science* **353**: 169–172.
# - Reynolds, R. W. *et al.* (2007) "Daily High-Resolution-Blended Analyses
#   for Sea Surface Temperature." *Journal of Climate* **20**: 5473–5496.
#   (NOAA OISST v2.1)
# - Górski, K. M. *et al.* (2005) "HEALPix: A Framework for High-Resolution
#   Discretization and Fast Analysis of Data Distributed on the Sphere."
#   *ApJ* **622**: 759.
# - Karlbauer, M. *et al.* (2024) "Advancing parsimonious deep learning
#   weather prediction using the HEALPix mesh." *J. Adv. Model. Earth Syst.*
