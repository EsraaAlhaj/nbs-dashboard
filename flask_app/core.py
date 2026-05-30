"""
flask_app/core.py
Pure-Python science layer for NBS4MED — no Streamlit imports.
All GEE / NASA POWER fetching, indices, charts (as JSON), reports.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import A4 as RL_A4
    from reportlab.lib.colors import HexColor as RLHex, white as rl_white
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

try:
    import ee
    EE_AVAILABLE = True
except Exception:
    EE_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
PALETTE = {
    "navy":   "#005088", "navy_d": "#003a66",
    "green":  "#11caa0", "green_d": "#0fb38e",
    "slate":  "#f8fafc", "orange": "#f59e0b",
    "red":    "#ef4444", "ink":    "#0f172a",
    "muted":  "#64748b", "blue_l": "#3b82f6",
    "purple": "#8b5cf6", "cyan":   "#06b6d4",
}

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

PILOT_SITES: Dict[str, Dict[str, Any]] = {
    "🇪🇸  Spain — Cerdanyola del Vallès": {
        "coords": (41.485473, 2.146883),
        "wp4_action": "Tiny urban forest (Miyawaki method)",
        "area_m2": 1850,
        "primary_risk": "Heat stress",
        "partner": "AMB (PP3)",
        "color": PALETTE["red"],
    },
    "🇹🇷  Türkiye — İzmir (Göztepe)": {
        "coords": (38.397758, 27.112083),
        "wp4_action": "Green roof on parking terrace",
        "area_m2": 820,
        "primary_risk": "Heat stress",
        "partner": "IMM (PP2)",
        "color": PALETTE["orange"],
    },
    "🇬🇷  Greece — Sikionies": {
        "coords": (38.023695, 22.734917),
        "wp4_action": "Smart eco-park (flood & heat)",
        "area_m2": 2500,
        "primary_risk": "Flooding & heat",
        "partner": "REGPEL (LP)",
        "color": PALETTE["blue_l"],
    },
    "🇹🇳  Tunisia — Greater Tunis": {
        "coords": (36.838967, 10.211539),
        "wp4_action": "Water retention basin",
        "area_m2": 2500,
        "primary_risk": "Flooding",
        "partner": "CITET (PP5)",
        "color": PALETTE["purple"],
    },
    "🇪🇬  Egypt — Heliopolis University": {
        "coords": (30.152959, 31.429408),
        "wp4_action": "Green corridor (Miyawaki method)",
        "area_m2": 2200,
        "primary_risk": "Heat stress & drought",
        "partner": "SEKEM (PP6)",
        "color": PALETTE["cyan"],
    },
}

SITE_SHORT = {
    "🇪🇸  Spain — Cerdanyola del Vallès": "Cerdanyola",
    "🇹🇷  Türkiye — İzmir (Göztepe)": "İzmir",
    "🇬🇷  Greece — Sikionies": "Sikionies",
    "🇹🇳  Tunisia — Greater Tunis": "Tunis",
    "🇪🇬  Egypt — Heliopolis University": "Heliopolis",
}

# ════════════════════════════════════════════════════════════════════════════
#  PLOTLY TEMPLATE
# ════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="DM Sans, sans-serif", size=12, color=PALETTE["ink"]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=16, r=16, t=40, b=16),
    legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="DM Sans",
                    bordercolor=PALETTE["navy"]),
)


def styled_fig(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(showgrid=False, linecolor="rgba(0,80,136,0.12)", tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="rgba(0,80,136,0.06)", gridwidth=1,
                     linecolor="rgba(0,80,136,0.12)", tickfont=dict(size=11), zeroline=False)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  EARTH ENGINE
# ════════════════════════════════════════════════════════════════════════════
def init_ee() -> Tuple[bool, str]:
    import sys
    if not EE_AVAILABLE:
        print("GEE-DEBUG: earthengine-api not installed", flush=True, file=sys.stderr)
        return False, "earthengine-api not installed."
    sa_json = os.environ.get("GEE_SERVICE_ACCOUNT_JSON")
    print(f"GEE-DEBUG: SA_JSON present={bool(sa_json)} len={len(sa_json) if sa_json else 0}", flush=True, file=sys.stderr)
    if sa_json:
        try:
            import json as _json
            sa_info = _json.loads(sa_json)
            print(f"GEE-DEBUG: client_email={sa_info.get('client_email')}", flush=True, file=sys.stderr)
            credentials = ee.ServiceAccountCredentials(
                email=sa_info["client_email"],
                key_data=sa_json,
            )
            ee.Initialize(credentials, project="ee-esraahalhaj")
            print("GEE-DEBUG: Authenticated via service account OK", flush=True, file=sys.stderr)
            return True, "Authenticated via service account."
        except Exception as err:
            print(f"GEE-DEBUG: Service account auth FAILED: {err}", flush=True, file=sys.stderr)
            return False, f"Service account auth failed: {err}"
    try:
        ee.Initialize(project="ee-esraahalhaj")
        print("GEE-DEBUG: Authenticated via project credentials", flush=True, file=sys.stderr)
        return True, "Authenticated via project credentials."
    except Exception:
        try:
            ee.Initialize()
            return True, "Authenticated via default credentials."
        except Exception as err:
            print(f"GEE-DEBUG: All auth methods failed: {err}", flush=True, file=sys.stderr)
            return False, f"Earth Engine not authenticated: {err}"


def site_buffer(area_m2: int, sensor: str = "landsat") -> int:
    if sensor == "chirps":  return 5000
    return 200


def aoi_geom(lat: float, lon: float, buffer_m: int = 500):
    return ee.Geometry.Point([lon, lat]).buffer(buffer_m)


# ════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING — EARTH ENGINE
# ════════════════════════════════════════════════════════════════════════════
def fetch_landsat_indices(lat: float, lon: float,
                          start: str, end: str,
                          area_m2: int = 2000) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"LST": None, "NDVI": None, "SAVI": None, "MSI": None}
    try:
        buf = site_buffer(area_m2, "landsat")
        out["buffer_m"] = buf
        aoi = aoi_geom(lat, lon, buffer_m=buf)

        def _mask(img):
            qa = img.select("QA_PIXEL")
            return img.updateMask(qa.bitwiseAnd(1 << 3).eq(0)).updateMask(qa.bitwiseAnd(1 << 4).eq(0))

        def _scale(img):
            optical = img.select(["SR_B4", "SR_B5", "SR_B6"]).multiply(0.0000275).add(-0.2)
            thermal = img.select("ST_B10").multiply(0.00341802).add(149.0)
            return img.addBands(optical, None, True).addBands(thermal, None, True)

        col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
               .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
               .filterBounds(aoi).filterDate(start, end)
               .filter(ee.Filter.lt("CLOUD_COVER", 40))
               .map(_mask).map(_scale))
        img  = col.median()
        red  = img.select("SR_B4")
        nir  = img.select("SR_B5")
        swir = img.select("SR_B6")
        lst_c = img.select("ST_B10").subtract(273.15).rename("LST")
        ndvi  = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        savi  = nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename("SAVI")
        msi   = swir.divide(nir).rename("MSI")
        stats = (lst_c.addBands([ndvi, savi, msi])
                 .reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi,
                               scale=30, maxPixels=1e9, bestEffort=True)
                 .getInfo()) or {}
        out.update({k: stats.get(k) for k in ("LST", "NDVI", "SAVI", "MSI")})
    except Exception as e:
        out["error"] = str(e)
    return out


def fetch_terrain(lat: float, lon: float, area_m2: int = 2000) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"elevation": None, "slope": None}
    try:
        aoi   = aoi_geom(lat, lon, buffer_m=site_buffer(area_m2, "terrain"))
        srtm  = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(srtm).rename("slope")
        stats = (srtm.rename("elevation").addBands(slope)
                 .reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi,
                               scale=30, maxPixels=1e9, bestEffort=True)
                 .getInfo()) or {}
        out["elevation"] = stats.get("elevation")
        out["slope"]     = stats.get("slope")
    except Exception as e:
        out["error"] = str(e)
    return out


def fetch_chirps_monthly(lat: float, lon: float,
                         months: int = 12, area_m2: int = 2000) -> pd.DataFrame:
    try:
        aoi   = aoi_geom(lat, lon, buffer_m=site_buffer(area_m2, "chirps"))
        end   = ee.Date(datetime.utcnow().strftime("%Y-%m-%d"))
        start = end.advance(-months, "month")
        col   = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").filterDate(start, end).filterBounds(aoi)

        def _month_sum(i):
            i = ee.Number(i)
            s = start.advance(i, "month")
            e = s.advance(1, "month")
            mp = (col.filterDate(s, e).sum()
                  .reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi,
                                scale=5000, maxPixels=1e9, bestEffort=True)
                  .get("precipitation"))
            return ee.Feature(None, {"date": s.format("YYYY-MM"), "precip": mp})

        feats = ee.FeatureCollection(ee.List.sequence(0, months - 1).map(_month_sum)).getInfo()
        rows  = [{"date": f["properties"]["date"],
                  "precip": f["properties"].get("precip") or 0.0}
                 for f in feats["features"]]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date", "precip"])


def fetch_ndvi_timeseries(lat: float, lon: float,
                          months: int = 12, area_m2: int = 2000) -> pd.DataFrame:
    try:
        aoi   = aoi_geom(lat, lon, buffer_m=site_buffer(area_m2, "modis"))
        end   = ee.Date(datetime.utcnow().strftime("%Y-%m-%d"))
        start = end.advance(-months, "month")
        col   = (ee.ImageCollection("MODIS/061/MOD13Q1")
                 .filterDate(start, end).filterBounds(aoi).select("NDVI"))

        def _m(i):
            i = ee.Number(i)
            s = start.advance(i, "month")
            e = s.advance(1, "month")
            v = (col.filterDate(s, e).mean().multiply(0.0001)
                 .reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi,
                               scale=250, maxPixels=1e9, bestEffort=True)
                 .get("NDVI"))
            return ee.Feature(None, {"date": s.format("YYYY-MM"), "ndvi": v})

        feats = ee.FeatureCollection(ee.List.sequence(0, months - 1).map(_m)).getInfo()
        rows  = [{"date": f["properties"]["date"], "ndvi": f["properties"].get("ndvi")}
                 for f in feats["features"]]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date", "ndvi"])


# ════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING — NASA POWER
# ════════════════════════════════════════════════════════════════════════════
def fetch_nasa_power(lat: float, lon: float,
                     start_year: int, end_year: int) -> pd.DataFrame:
    params = "T2M,T2M_MAX,T2M_MIN,RH2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR"
    url = (f"https://power.larc.nasa.gov/api/temporal/monthly/point"
           f"?parameters={params}&community=AG"
           f"&longitude={lon}&latitude={lat}"
           f"&start={start_year}&end={end_year}&format=JSON")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json().get("properties", {}).get("parameter", {})
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df.index.name = "yyyymm"
        df = df.reset_index()
        df["yyyymm"] = df["yyyymm"].astype(str)
        df = df[df["yyyymm"].str.len() == 6]
        df["mm"] = df["yyyymm"].str[-2:]
        df = df[df["mm"].isin([f"{m:02d}" for m in range(1, 13)])]
        df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m", errors="coerce")
        df = df.dropna(subset=["date"]).drop(columns=["mm", "yyyymm"])
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df.loc[df[c] <= -999, c] = np.nan
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
#  DERIVED INDICES
# ════════════════════════════════════════════════════════════════════════════
def compute_thermal_zscore(
    observed_val: Optional[float],
    power_df: pd.DataFrame,
    column: str = "T2M_MAX",
    summer_only: bool = True,
) -> Optional[float]:
    """Z-score of observed_val vs. site's own 30-year NASA POWER baseline.
    summer_only restricts the baseline to Jun-Aug (peak heat months).
    Returns None when < 24 valid monthly values are available."""
    if observed_val is None or power_df.empty or column not in power_df.columns:
        return None
    src = power_df[power_df["date"].dt.month.isin([6, 7, 8])] if summer_only else power_df
    vals = src[column].dropna()
    if len(vals) < 24:
        return None
    sig = float(vals.std())
    if sig < 0.01:
        return None
    return round((observed_val - float(vals.mean())) / sig, 2)


def heat_index_noaa(t_c: Optional[float], rh: Optional[float]) -> Optional[float]:
    """NOAA Heat Index (Rothfusz 1990 + NWS adjustments). Input: air temp °C,
    RH %. Output: °C. Includes the NWS low-RH and high-RH corrections, which
    materially lower HI for dry/hot sites (Egypt, Tunisia)."""
    if t_c is None or rh is None or pd.isna(t_c) or pd.isna(rh):
        return None
    t_f = t_c * 9 / 5 + 32
    if t_f < 80:
        return t_c
    hi = (-42.379 + 2.04901523 * t_f + 10.14333127 * rh
          - 0.22475541 * t_f * rh - 6.83783e-3 * t_f ** 2
          - 5.481717e-2 * rh ** 2 + 1.22874e-3 * t_f ** 2 * rh
          + 8.5282e-4 * t_f * rh ** 2 - 1.99e-6 * t_f ** 2 * rh ** 2)
    # NWS low-humidity adjustment (subtracts from HI in dry heat)
    if rh < 13 and 80 <= t_f <= 112:
        hi -= ((13 - rh) / 4) * math.sqrt((17 - abs(t_f - 95)) / 17)
    # NWS high-humidity adjustment
    elif rh > 85 and 80 <= t_f <= 87:
        hi += ((rh - 85) / 10) * ((87 - t_f) / 5)
    return (hi - 32) * 5 / 9


def compute_spi(precip: pd.Series, window: int = 3,
                dates: Optional[pd.Series] = None) -> pd.Series:
    """SPI-{window} via Gamma fitting (McKee et al. 1993).

    When ``dates`` are supplied the gamma distribution is fit *separately for
    each calendar month*, which removes the seasonal cycle. Without per-month
    fitting a normal-but-dry Mediterranean summer always reads as drought.
    Falls back to a pooled standardized anomaly when dates/scipy are absent.
    """
    if precip is None or len(precip) < window + 2:
        return pd.Series(dtype=float)
    precip  = pd.Series(np.asarray(precip, dtype=float)).reset_index(drop=True)
    rolling = precip.rolling(window=window, min_periods=window).sum()

    def _pooled() -> pd.Series:
        mu, sd = rolling.mean(), rolling.std()
        if sd is None or pd.isna(sd) or sd == 0:
            return pd.Series([0.0] * len(precip))
        return (rolling - mu) / sd

    if dates is None or not SCIPY_AVAILABLE:
        return _pooled()

    months = pd.to_datetime(pd.Series(np.asarray(dates))).dt.month.to_numpy()
    spi    = pd.Series(np.nan, index=precip.index)
    for m in range(1, 13):
        idx   = np.where(months == m)[0]
        if len(idx) == 0:
            continue
        vals  = rolling.iloc[idx]
        valid = vals.dropna()
        pos   = valid[valid > 0]
        if len(pos) < 8:
            sd = valid.std()
            if sd and sd > 0:
                spi.iloc[idx] = (vals - valid.mean()) / sd
            continue
        try:
            alpha, _loc, beta = scipy_stats.gamma.fit(pos, floc=0)
        except Exception:
            sd = valid.std()
            if sd and sd > 0:
                spi.iloc[idx] = (vals - valid.mean()) / sd
            continue
        q_zero = (valid == 0).sum() / len(valid) if len(valid) else 0.0
        for j in idx:
            v = rolling.iloc[j]
            if pd.isna(v):
                continue
            prob = (q_zero * 0.5 if v <= 0
                    else q_zero + (1 - q_zero) * scipy_stats.gamma.cdf(v, alpha, loc=0, scale=beta))
            prob = max(0.0015, min(prob, 0.9985))
            spi.iloc[j] = float(scipy_stats.norm.ppf(prob))
    return spi


def _day_length(lat_deg: float, month: int) -> float:
    """Approximate mean daylight hours (Forsythe et al. 1995)."""
    doy_mid = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    j = doy_mid[month - 1]
    decl    = 23.45 * np.sin(np.radians(360 / 365 * (j - 81)))
    cos_ha  = np.clip(-np.tan(np.radians(lat_deg)) * np.tan(np.radians(decl)), -1, 1)
    return 2 * np.degrees(np.arccos(cos_ha)) / 15


def thornthwaite_pet(power_df: pd.DataFrame, lat: float) -> pd.Series:
    """Thornthwaite (1948) monthly PET (mm/month)."""
    if power_df.empty or "T2M" not in power_df.columns:
        return pd.Series(dtype=float)
    df = power_df[["date", "T2M"]].dropna().copy()
    if len(df) < 12:
        return pd.Series(dtype=float)
    df["T_clip"] = df["T2M"].clip(lower=0)
    df["month"]  = df["date"].dt.month
    monthly_mean = df.groupby("month")["T_clip"].mean()
    I_annual     = ((monthly_mean / 5) ** 1.514).sum()
    if I_annual <= 0:
        return pd.Series([0.0] * len(df), index=df.index)
    a = 6.75e-7 * I_annual**3 - 7.71e-5 * I_annual**2 + 1.792e-2 * I_annual + 0.49239
    pet_values = []
    for _, row in df.iterrows():
        t, m = row["T_clip"], row["month"]
        if t <= 0:
            pet_values.append(0.0)
            continue
        N = _day_length(lat, m)
        d = DAYS_IN_MONTH[m - 1]
        pet_values.append(max(0.0, 16 * (10 * t / I_annual) ** a * (N / 12) * (d / 30)))
    return pd.Series(pet_values, index=df.index)


def _extraterrestrial_radiation(lat_deg: float, month: int) -> float:
    """FAO-56 extraterrestrial radiation Ra (MJ m⁻² day⁻¹) for the mid-month day."""
    doy_mid = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    j   = doy_mid[month - 1]
    phi = math.radians(lat_deg)
    dr  = 1 + 0.033 * math.cos(2 * math.pi / 365 * j)
    dec = 0.409 * math.sin(2 * math.pi / 365 * j - 1.39)
    x   = max(-1.0, min(1.0, -math.tan(phi) * math.tan(dec)))
    ws  = math.acos(x)
    gsc = 0.0820  # MJ m⁻² min⁻¹
    ra  = (24 * 60 / math.pi) * gsc * dr * (
        ws * math.sin(phi) * math.sin(dec)
        + math.cos(phi) * math.cos(dec) * math.sin(ws))
    return max(0.0, ra)


def hargreaves_pet(power_df: pd.DataFrame, lat: float) -> pd.Series:
    """Hargreaves & Samani (1985) / FAO-56 monthly PET (mm/month).

    Radiation-aware (via extraterrestrial radiation Ra and the diurnal
    temperature range), so it does not under-predict PET in arid/windy
    climates the way temperature-only Thornthwaite does.
    """
    need = {"T2M", "T2M_MAX", "T2M_MIN"}
    if power_df.empty or not need.issubset(power_df.columns):
        return pd.Series(dtype=float)
    df = power_df[["date", "T2M", "T2M_MAX", "T2M_MIN"]].dropna().copy()
    if len(df) < 12:
        return pd.Series(dtype=float)
    df["month"] = df["date"].dt.month
    pet_values = []
    for _, r in df.iterrows():
        m       = int(r["month"])
        ra_mm   = _extraterrestrial_radiation(lat, m) * 0.408  # MJ→mm equivalent
        d_temp  = max(0.0, float(r["T2M_MAX"]) - float(r["T2M_MIN"]))
        pet_day = max(0.0, 0.0023 * ra_mm * (float(r["T2M"]) + 17.8) * math.sqrt(d_temp))
        pet_values.append(pet_day * DAYS_IN_MONTH[m - 1])
    return pd.Series(pet_values, index=df.index)


def compute_aridity(power_df: pd.DataFrame, lat: float,
                    method: str = "hargreaves") -> Optional[float]:
    """UNEP (1992) Aridity Index = P_annual / PET_annual.

    PET via Hargreaves/FAO-56 by default (radiation-aware); falls back to
    Thornthwaite (1948) when Tmax/Tmin are unavailable.
    """
    if power_df.empty or "PRECTOTCORR" not in power_df.columns:
        return None
    df = power_df.dropna(subset=["T2M", "PRECTOTCORR"]).copy()
    if len(df) < 12:
        return None
    df["month"]     = df["date"].dt.month
    df["year"]      = df["date"].dt.year
    df["p_monthly"] = df["PRECTOTCORR"] * df["month"].map(lambda m: DAYS_IN_MONTH[m - 1])
    pet_s = hargreaves_pet(df, lat) if method == "hargreaves" else pd.Series(dtype=float)
    if pet_s.empty:
        pet_s = thornthwaite_pet(df, lat)
    if pet_s.empty:
        return None
    df = df.loc[pet_s.index].copy()
    df["pet_monthly"] = pet_s.values
    annual = df.groupby("year").agg(
        p_total=("p_monthly", "sum"),
        pet_total=("pet_monthly", "sum"),
        n_months=("month", "count"),
    )
    annual = annual[annual["n_months"] >= 11]
    if annual.empty:
        return None
    p_mean, pet_mean = annual["p_total"].mean(), annual["pet_total"].mean()
    return None if pet_mean <= 0 else p_mean / pet_mean


def compute_vhi(ndvi_series: pd.Series, lst_now: Optional[float],
                lst_min_est: float = 5.0, lst_max_est: float = 55.0,
                alpha: float = 0.5) -> Optional[float]:
    """VHI = α·VCI + (1-α)·TCI — Kogan (1995).
    TCI now uses dynamic regional thermal envelopes."""
    if ndvi_series is None or ndvi_series.dropna().empty or lst_now is None:
        return None
    n = ndvi_series.dropna()
    n_min, n_max = n.min(), n.max()
    if n_max == n_min:
        return None
    vci = (n.iloc[-1] - n_min) / (n_max - n_min) * 100

    if lst_max_est == lst_min_est:
        tci = 50.0
    else:
        tci = max(0.0, min(100.0, (lst_max_est - lst_now) / (lst_max_est - lst_min_est) * 100))
    return float(alpha * vci + (1 - alpha) * tci)


def compute_vci(ndvi_series: pd.Series) -> Optional[float]:
    """Vegetation Condition Index (Kogan 1995), 0–100. Unlike VHI this uses
    only the observed NDVI record — no synthetic LST climatology — so it is
    the defensible vegetation KPI for a screening tool."""
    if ndvi_series is None:
        return None
    n = pd.Series(ndvi_series).dropna()
    if n.empty:
        return None
    n_min, n_max = n.min(), n.max()
    if n_max == n_min:
        return None
    return float((n.iloc[-1] - n_min) / (n_max - n_min) * 100)


# ════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION HELPERS
# ════════════════════════════════════════════════════════════════════════════
def aridity_class(ai: Optional[float]) -> Tuple[str, str]:
    if ai is None: return "Unknown",      PALETTE["muted"]
    if ai < 0.05:  return "Hyper-arid",   PALETTE["red"]
    if ai < 0.20:  return "Arid",         PALETTE["red"]
    if ai < 0.50:  return "Semi-arid",    PALETTE["orange"]
    if ai < 0.65:  return "Dry sub-humid",PALETTE["orange"]
    return "Humid", PALETTE["green"]


def vhi_class(vhi: Optional[float]) -> Tuple[str, str]:
    if vhi is None: return "Unknown",          PALETTE["muted"]
    if vhi < 10:    return "Extreme drought",  PALETTE["red"]
    if vhi < 20:    return "Severe drought",   PALETTE["red"]
    if vhi < 30:    return "Moderate drought", PALETTE["orange"]
    if vhi < 40:    return "Mild drought",     PALETTE["orange"]
    return "Healthy", PALETTE["green"]


def kpi_classifications(metrics: Dict[str, Any]) -> Dict[str, Dict]:
    """Return CSS class + accent hex for each KPI card (Flask template use)."""
    _CSS = {
        "green": ("status-green", "#15803d"),
        "amber": ("status-amber", "#b45309"),
        "red":   ("status-red",   "#b91c1c"),
        "none":  ("",             "#003D7A"),
    }

    def _r(key, tier, lbl): return {"value": key, "label": lbl, "cls": _CSS[tier][0], "accent": _CSS[tier][1]}

    result = {}
    lst = metrics.get("LST")
    if   lst is not None and lst > 40: t, lbl = "red",   "Critical"
    elif lst is not None and lst > 32: t, lbl = "amber", "High"
    elif lst is not None:              t, lbl = "green", "Normal"
    else:                              t, lbl = "none",  ""
    result["lst"] = {"value": f"{lst:.1f}" if lst is not None else "—",
                     "label": lbl, "cls": _CSS[t][0], "accent": _CSS[t][1]}

    hi = metrics.get("HeatIndex")
    if   hi is not None and hi > 40: t, lbl = "red",   "Danger"
    elif hi is not None and hi > 32: t, lbl = "amber", "Caution"
    elif hi is not None:             t, lbl = "green", "Comfort"
    else:                            t, lbl = "none",  ""
    result["heat_index"] = {"value": f"{hi:.1f}" if hi is not None else "—",
                             "label": lbl, "cls": _CSS[t][0], "accent": _CSS[t][1]}

    vhi = metrics.get("VHI")
    vhi_lbl, _ = vhi_class(vhi)
    t = {"Healthy": "green", "Mild drought": "amber", "Moderate drought": "amber",
         "Severe drought": "red", "Extreme drought": "red"}.get(vhi_lbl, "none")
    result["vegetation"] = {"value": f"{vhi:.0f}" if vhi is not None else "—",
                             "label": vhi_lbl, "cls": _CSS[t][0], "accent": _CSS[t][1]}

    ai = metrics.get("Aridity")
    ai_lbl, _ = aridity_class(ai)
    t = {"Humid": "green", "Dry sub-humid": "amber", "Semi-arid": "amber",
         "Arid": "red", "Hyper-arid": "red"}.get(ai_lbl, "none")
    result["aridity"] = {"value": f"{ai:.2f}" if ai is not None else "—",
                          "label": ai_lbl, "cls": _CSS[t][0], "accent": _CSS[t][1]}
    return result


# ════════════════════════════════════════════════════════════════════════════
#  CHARTS — return Plotly JSON strings for Jinja2 / JavaScript
# ════════════════════════════════════════════════════════════════════════════
def chart_ndvi_json(ndvi_df: pd.DataFrame) -> Optional[str]:
    if ndvi_df is None or ndvi_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ndvi_df["date"].dt.strftime("%Y-%m").tolist(),
        y=ndvi_df["ndvi"].tolist(),
        name="NDVI (MODIS)",
        line=dict(color=PALETTE["green"], width=2.8, shape="spline"),
        mode="lines+markers",
        marker=dict(size=5, color=PALETTE["green"], line=dict(width=1, color="white")),
        fill="tozeroy", fillcolor="rgba(17,202,160,0.08)",
        hovertemplate="NDVI: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.2, line_dash="dash", line_color=PALETTE["muted"], line_width=1,
                  annotation_text="Sparse vegetation", annotation_font_size=9,
                  annotation_font_color=PALETTE["muted"], annotation_position="top left")
    fig.update_yaxes(title_text="NDVI", range=[-0.05, 1])
    styled_fig(fig, 380)
    return fig.to_json()


def chart_spi_json(chirps_df: pd.DataFrame,
                   power_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    spi_df, spi_label = None, "SPI-3"
    if not chirps_df.empty:
        spi_df = chirps_df.copy()
        spi_df["spi"] = compute_spi(spi_df["precip"], window=3, dates=spi_df["date"])
        spi_label = "SPI-3 (CHIRPS)"
    elif power_df is not None and not power_df.empty and "PRECTOTCORR" in power_df.columns:
        tmp = power_df[["date", "PRECTOTCORR"]].dropna().copy()
        tmp["month"]      = tmp["date"].dt.month
        tmp["precip_mm"]  = tmp["PRECTOTCORR"] * tmp["month"].map(lambda m: DAYS_IN_MONTH[m - 1])
        tmp = tmp.reset_index(drop=True)
        if len(tmp) >= 6:
            spi_df = tmp[["date", "precip_mm"]].rename(columns={"precip_mm": "precip"})
            spi_df["spi"] = compute_spi(spi_df["precip"], window=3, dates=spi_df["date"])
            spi_label = "SPI-3 (NASA POWER)"
    if spi_df is None or spi_df.empty or "spi" not in spi_df.columns:
        return None
    colors = [PALETTE["red"] if v < -1.5 else PALETTE["orange"] if v < -1
              else PALETTE["navy"] if v < 0 else PALETTE["green"] if v < 1
              else PALETTE["blue_l"] for v in spi_df["spi"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=spi_df["date"].dt.strftime("%Y-%m").tolist(),
        y=spi_df["spi"].tolist(),
        name=spi_label, marker_color=colors, opacity=0.72,
        hovertemplate="SPI-3: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=-1.5, line_dash="dot", line_color=PALETTE["red"], line_width=1,
                  annotation_text="Severe drought", annotation_font_size=9,
                  annotation_font_color=PALETTE["red"], annotation_position="bottom left")
    fig.add_hline(y=-1.0, line_dash="dot", line_color=PALETTE["orange"], line_width=1,
                  annotation_text="Moderate drought", annotation_font_size=9,
                  annotation_font_color=PALETTE["orange"], annotation_position="bottom left")
    fig.update_yaxes(title_text="SPI-3", range=[-3, 3])
    styled_fig(fig, 380)
    return fig.to_json()


def chart_climate_profile_json(power_df: pd.DataFrame) -> Optional[str]:
    if power_df.empty:
        return None
    df  = power_df.copy()
    df["month"] = df["date"].dt.month
    agg = (df.groupby("month")
             .agg(srad=("ALLSKY_SFC_SW_DWN", "mean"), precip=("PRECTOTCORR", "mean"),
                  tmean=("T2M", "mean"), tmax=("T2M_MAX", "mean"), tmin=("T2M_MIN", "mean"))
             .reset_index())
    agg["m_name"] = pd.to_datetime(agg["month"], format="%m").dt.strftime("%b")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=agg["m_name"].tolist(), y=agg["tmax"].tolist(),
                             mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg["m_name"].tolist(), y=agg["tmin"].tolist(),
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(239,68,68,0.10)", name="T range (min–max)",
                             hoverinfo="skip"), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg["m_name"].tolist(), y=agg["tmean"].tolist(),
                             name="Mean Air T",
                             line=dict(color=PALETTE["red"], width=2.5, shape="spline"),
                             mode="lines+markers",
                             marker=dict(size=6, color=PALETTE["red"],
                                         line=dict(width=1.5, color="white")),
                             hovertemplate="%{y:.1f} °C<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Bar(x=agg["m_name"].tolist(), y=agg["precip"].tolist(),
                         name="Precipitation", marker_color=PALETTE["navy"], opacity=0.7,
                         hovertemplate="%{y:.1f} mm/day<extra></extra>"), secondary_y=True)
    fig.add_trace(go.Scatter(x=agg["m_name"].tolist(), y=agg["srad"].tolist(),
                             name="Solar Radiation",
                             line=dict(color=PALETTE["orange"], width=2, dash="dot", shape="spline"),
                             mode="lines+markers",
                             marker=dict(size=5, symbol="diamond", color=PALETTE["orange"]),
                             hovertemplate="%{y:.1f} kWh/m²/day<extra></extra>"), secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C) · Solar Rad (kWh/m²/d)", secondary_y=False)
    fig.update_yaxes(title_text="Precipitation (mm/day)", secondary_y=True)
    styled_fig(fig, 420)
    return fig.to_json()


def chart_temperature_profile_json(power_df: pd.DataFrame) -> Optional[str]:
    """Monthly temperature profile: mean, max, min air temperature (NASA POWER)."""
    if power_df.empty or "T2M" not in power_df.columns:
        return None
    df = power_df.copy()
    df["month"] = df["date"].dt.month
    need = [c for c in ("T2M", "T2M_MAX", "T2M_MIN") if c in df.columns]
    agg = df.groupby("month")[need].mean().reset_index()
    agg["m_name"] = pd.to_datetime(agg["month"], format="%m").dt.strftime("%b")
    fig = go.Figure()
    if "T2M_MAX" in agg.columns and "T2M_MIN" in agg.columns:
        fig.add_trace(go.Scatter(
            x=agg["m_name"].tolist(), y=agg["T2M_MAX"].tolist(),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=agg["m_name"].tolist(), y=agg["T2M_MIN"].tolist(),
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(239,68,68,0.10)", name="T range (min–max)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=agg["m_name"].tolist(), y=agg["T2M"].tolist(),
        name="Mean Air Temp",
        line=dict(color=PALETTE["red"], width=2.5, shape="spline"),
        mode="lines+markers",
        marker=dict(size=6, color=PALETTE["red"], line=dict(width=1.5, color="white")),
        hovertemplate="%{y:.1f} °C<extra></extra>"))
    fig.update_yaxes(title_text="Temperature (°C)")
    styled_fig(fig, 360)
    return fig.to_json()


def chart_hydro_solar_profile_json(power_df: pd.DataFrame) -> Optional[str]:
    """Monthly hydro-solar profile: precipitation + solar radiation (NASA POWER)."""
    if power_df.empty:
        return None
    df = power_df.copy()
    df["month"] = df["date"].dt.month
    need = [c for c in ("PRECTOTCORR", "ALLSKY_SFC_SW_DWN") if c in df.columns]
    if not need:
        return None
    agg = df.groupby("month")[need].mean().reset_index()
    agg["m_name"] = pd.to_datetime(agg["month"], format="%m").dt.strftime("%b")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "PRECTOTCORR" in agg.columns:
        fig.add_trace(go.Bar(
            x=agg["m_name"].tolist(), y=agg["PRECTOTCORR"].tolist(),
            name="Precipitation", marker_color=PALETTE["navy"], opacity=0.7,
            hovertemplate="%{y:.1f} mm/day<extra></extra>"), secondary_y=False)
    if "ALLSKY_SFC_SW_DWN" in agg.columns:
        fig.add_trace(go.Scatter(
            x=agg["m_name"].tolist(), y=agg["ALLSKY_SFC_SW_DWN"].tolist(),
            name="Solar Radiation",
            line=dict(color=PALETTE["orange"], width=2, dash="dot", shape="spline"),
            mode="lines+markers",
            marker=dict(size=5, symbol="diamond", color=PALETTE["orange"]),
            hovertemplate="%{y:.1f} kWh/m²/day<extra></extra>"), secondary_y=True)
    fig.update_yaxes(
        title_text="Precipitation (mm/day)",
        title_font=dict(color=PALETTE["navy"]),
        tickfont=dict(color=PALETTE["navy"]),
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Solar Radiation (kWh/m²/d)",
        title_font=dict(color=PALETTE["orange"]),
        tickfont=dict(color=PALETTE["orange"]),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Monthly Average (30-Year Baseline)")
    styled_fig(fig, 360)
    return fig.to_json()


def chart_stress_radar_json(metrics: Dict[str, Any], site_name: str) -> Optional[str]:
    lst  = metrics.get("LST")
    ndvi = metrics.get("NDVI")
    msi  = metrics.get("MSI")
    vhi  = metrics.get("VHI")
    ai   = metrics.get("Aridity")
    hi   = metrics.get("HeatIndex")
    categories = ["Heat Stress", "Vegetation\nDeficit", "Moisture\nStress",
                  "Drought\nVulnerability", "Aridity", "Bioclimatic\nRisk"]
    scores = [
        min(1.0, max(0, (lst - 20) / 30)) if lst is not None else 0,
        min(1.0, max(0, 1 - (ndvi or 0) / 0.6)),
        min(1.0, max(0, ((msi or 1) - 0.5) / 2.0)),
        min(1.0, max(0, 1 - (vhi or 50) / 100)),
        min(1.0, max(0, 1 - (ai or 0.5) / 0.65)),
        min(1.0, max(0, ((hi if hi is not None else 25) - 25) / 25)) if hi is not None else 0,
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(0,80,136,0.12)",
        line=dict(color=PALETTE["navy"], width=2.5),
        marker=dict(size=7, color=PALETTE["navy"], line=dict(width=2, color="white")),
        name=SITE_SHORT.get(site_name, "Site"),
        hovertemplate="Stress: %{r:.2f}<extra></extra>"))
    fig.add_trace(go.Scatterpolar(
        r=[0.6] * 7, theta=categories + [categories[0]],
        line=dict(color=PALETTE["orange"], width=1.5, dash="dot"),
        name="Alert threshold", hoverinfo="skip"))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.2, 0.4, 0.6, 0.8],
                            ticktext=["Low", "", "High", "Critical"],
                            tickfont=dict(size=9, color=PALETTE["muted"]),
                            gridcolor="rgba(0,80,136,0.08)"),
            angularaxis=dict(tickfont=dict(size=11, color=PALETTE["ink"]),
                             gridcolor="rgba(0,80,136,0.08)"),
            bgcolor="rgba(0,0,0,0)"),
        font=dict(family="DM Sans", size=12), paper_bgcolor="rgba(0,0,0,0)",
        height=400, margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(orientation="h", y=-0.05, x=0.3, font=dict(size=11)), showlegend=True)
    return fig.to_json()


def chart_annual_precip_json(power_df: pd.DataFrame) -> Optional[str]:
    if power_df.empty:
        return None
    df = power_df.copy()
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["p_monthly"] = df["PRECTOTCORR"] * df["month"].map(lambda m: DAYS_IN_MONTH[m - 1])
    annual = (df.groupby("year")
               .agg(total_precip=("p_monthly", "sum"), n_months=("month", "count"))
               .reset_index())
    annual = annual[annual["n_months"] >= 11]
    if len(annual) < 3:
        return None
    mean_p = annual["total_precip"].mean()
    colors = [PALETTE["red"] if v < mean_p * 0.7 else PALETTE["orange"] if v < mean_p * 0.9
              else PALETTE["green"] if v < mean_p * 1.1 else PALETTE["blue_l"]
              for v in annual["total_precip"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=annual["year"].tolist(), y=annual["total_precip"].tolist(),
                         marker_color=colors, opacity=0.85, name="Annual Precipitation",
                         hovertemplate="<b>%{x}</b><br>%{y:.0f} mm/year<extra></extra>"))
    fig.add_hline(y=mean_p, line_dash="dash", line_color=PALETTE["muted"], line_width=1.5,
                  annotation_text=f"Mean: {mean_p:.0f} mm/yr", annotation_font_size=11,
                  annotation_font_color=PALETTE["ink"])
    fig.update_yaxes(title_text="Precipitation (mm/year)")
    fig.update_xaxes(dtick=1)
    styled_fig(fig, 360)
    return fig.to_json()


# ════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION (exact logic from app1.py — no st.* calls)
# ════════════════════════════════════════════════════════════════════════════
def generate_report_context(site: str, lat: float, lon: float,
                             site_data: Dict, metrics: Dict[str, Any],
                             power_df: pd.DataFrame) -> dict:
    """Return structured context dict consumed by report.html template."""
    lst  = metrics.get("LST");  ndvi = metrics.get("NDVI");  savi = metrics.get("SAVI")
    msi  = metrics.get("MSI");  vhi  = metrics.get("VCI", metrics.get("VHI"))
    ai   = metrics.get("Aridity");  hi = metrics.get("HeatIndex")
    elev = metrics.get("elevation"); slope = metrics.get("slope")

    t_mean = rh_mean = srad_mean = p_yr = pet_yr = None
    if not power_df.empty:
        t_mean    = power_df["T2M"].mean()
        rh_mean   = power_df["RH2M"].mean()
        srad_mean = power_df["ALLSKY_SFC_SW_DWN"].mean()
        _tmp = power_df.copy()
        _tmp["month"] = _tmp["date"].dt.month
        _tmp["p_mm"]  = _tmp["PRECTOTCORR"] * _tmp["month"].map(lambda m: DAYS_IN_MONTH[m - 1])
        p_yr = _tmp.groupby(_tmp["date"].dt.year)["p_mm"].sum().mean()
        pet_s = hargreaves_pet(power_df, lat)
        if pet_s.empty: pet_s = thornthwaite_pet(power_df, lat)
        if not pet_s.empty:
            _pet = power_df.loc[pet_s.index, ["date"]].copy()
            _pet["pet"] = pet_s.values
            pet_yr = _pet.groupby(_pet["date"].dt.year)["pet"].sum().mean()

    klass_a, _ = aridity_class(ai)
    today = datetime.utcnow().strftime("%d %B %Y")
    short = SITE_SHORT.get(site, site)

    def _fv(v, p=2, u=""):
        if v is None: return "n/a"
        try:
            f = float(v)
            if math.isnan(f): return "n/a"
            return f"{f:.{p}f}{u}"
        except (TypeError, ValueError): return "n/a"

    def _cls_lst(v):
        if v is None: return "Unknown",     "neutral"
        if v < 28:    return "Normal",      "green"
        if v < 32:    return "Elevated",    "amber"
        if v < 40:    return "High",        "orange"
        return             "Critical",     "red"

    def _cls_hi(v):
        if v is None: return "Unknown",     "neutral"
        if v < 27:    return "Comfortable", "green"
        if v < 32:    return "Caution",     "amber"
        if v < 41:    return "High Risk",   "orange"
        return             "Danger",       "red"

    def _cls_vci(v):
        if v is None: return "Unknown",         "neutral"
        if v < 25:    return "Severe stress",   "red"
        if v < 40:    return "Moderate stress", "orange"
        if v < 60:    return "Fair",            "amber"
        return             "Good",             "green"

    def _cls_ai(v):
        if v is None: return "Unknown",       "neutral"
        if v >= 0.65: return "Humid",         "green"
        if v >= 0.50: return "Dry sub-humid", "amber"
        if v >= 0.20: return "Semi-arid",     "orange"
        if v >= 0.05: return "Arid",          "red"
        return             "Hyper-arid",     "red"

    def _cls_ndvi(v):
        if v is None: return "Unknown",  "neutral"
        if v < 0.2:   return "Sparse",   "red"
        if v < 0.35:  return "Low",      "orange"
        if v < 0.5:   return "Moderate", "amber"
        return             "Dense",     "green"

    def _cls_msi(v):
        if v is None: return "Unknown",      "neutral"
        if v < 0.6:   return "Well-watered", "green"
        if v < 1.0:   return "Low stress",   "amber"
        if v < 1.5:   return "Moderate",     "orange"
        return             "High stress",   "red"

    lst_lbl,  lst_col  = _cls_lst(lst)
    hi_lbl,   hi_col   = _cls_hi(hi)
    vci_lbl,  vci_col  = _cls_vci(vhi)
    ai_lbl,   ai_col   = _cls_ai(ai)
    ndvi_lbl, ndvi_col = _cls_ndvi(ndvi)
    msi_lbl,  msi_col  = _cls_msi(msi)

    recs = []
    if lst  is not None and lst  > 32:  recs.append({"text": "Urban green corridors and cool pavements to mitigate surface heat island effect.",             "cat": "Urban Cooling NBS"})
    if hi   is not None and hi   > 35:  recs.append({"text": "Green roofs and urban forest canopy to reduce ambient air temperature and heat stress.",        "cat": "Green Infrastructure"})
    if vhi  is not None and vhi  < 40:  recs.append({"text": "Drought-tolerant native afforestation to rebuild vegetation resilience.",                       "cat": "Ecological Restoration"})
    if ai   is not None and ai   < 0.5: recs.append({"text": "Constructed wetlands and water-sensitive urban design for precipitation capture and groundwater recharge.", "cat": "Water-Sensitive Design"})
    if msi  is not None and msi  > 1.5: recs.append({"text": "Mulching, biochar amendment, and soil-moisture conservation to address vegetation water stress.", "cat": "Soil Management NBS"})
    if slope is not None and slope > 8: recs.append({"text": "Bioengineered terraces and riparian buffer strips for slope stabilisation and erosion control.",  "cat": "Slope Stabilisation NBS"})
    if not recs: recs.append({"text": "Preventive NBS (green roofs, permeable surfaces, pocket parks) — compound stress is currently low.", "cat": "Preventive NBS"})

    scores = []
    if lst is not None: scores.append(min(100, max(0, (lst - 20) / 30 * 100)))
    if vhi is not None: scores.append(min(100, max(0, (40 - vhi) / 40 * 100)))
    if ai  is not None: scores.append(min(100, max(0, (0.65 - ai) / 0.65 * 100)))
    comp = sum(scores) / len(scores) if scores else None
    if   comp is None: comp_label, comp_col = "Unknown",  "neutral"
    elif comp < 25:    comp_label, comp_col = "LOW",      "green"
    elif comp < 50:    comp_label, comp_col = "MODERATE", "amber"
    elif comp < 75:    comp_label, comp_col = "HIGH",     "orange"
    else:              comp_label, comp_col = "CRITICAL", "red"

    thermal_level = (
        "high"     if (lst is not None and lst > 36) or (hi is not None and hi > 38) else
        "moderate" if (lst is not None and lst > 32) or (hi is not None and hi > 32) else "low"
    )
    veg_level = (
        "sparse and stressed" if ndvi is not None and ndvi < 0.20 else
        "below moderate"      if ndvi is not None and ndvi < 0.35 else
        "moderate"            if ndvi is not None and ndvi < 0.50 else "healthy"
    )

    _CROSS = [
        {"short": "Cerdanyola", "country": "Spain",   "flag": "ES", "aridity": "Semi-arid",    "ai_v": 0.45, "lst_v": 36.2, "hi_v": 34.8, "vci_v": 35, "ndvi_v": 0.22},
        {"short": "Izmir",      "country": "Turkey",  "flag": "TR", "aridity": "Dry sub-humid", "ai_v": 0.56, "lst_v": 37.8, "hi_v": 37.5, "vci_v": 31, "ndvi_v": 0.19},
        {"short": "Sikionies",  "country": "Greece",  "flag": "GR", "aridity": "Dry sub-humid", "ai_v": 0.53, "lst_v": 34.9, "hi_v": 35.2, "vci_v": 38, "ndvi_v": 0.31},
        {"short": "Tunis",      "country": "Tunisia", "flag": "TN", "aridity": "Semi-arid",    "ai_v": 0.32, "lst_v": 40.1, "hi_v": 37.0, "vci_v": 26, "ndvi_v": 0.15},
        {"short": "Heliopolis", "country": "Egypt",   "flag": "EG", "aridity": "Arid",         "ai_v": 0.09, "lst_v": 44.6, "hi_v": 42.8, "vci_v": 12, "ndvi_v": 0.08},
    ]
    _sn = short.lower().replace("İ", "i").replace("ı", "i")
    for row in _CROSS:
        row["is_current"] = (row["short"].lower() in _sn or _sn in row["short"].lower())
        if row["is_current"]:
            if lst  is not None and not (isinstance(lst,  float) and pd.isna(lst)):  row["lst_v"]  = round(float(lst), 1)
            if hi   is not None and not (isinstance(hi,   float) and pd.isna(hi)):   row["hi_v"]   = round(float(hi), 1)
            if vhi  is not None and not (isinstance(vhi,  float) and pd.isna(vhi)):  row["vci_v"]  = int(round(float(vhi)))
            if ndvi is not None and not (isinstance(ndvi, float) and pd.isna(ndvi)): row["ndvi_v"] = round(float(ndvi), 3)
            if ai   is not None and not (isinstance(ai,   float) and pd.isna(ai)):   row["ai_v"]   = round(float(ai), 3)

    return {
        "site":      short,
        "today":     today,
        "lat":       lat,
        "lon":       lon,
        "site_data": site_data,
        "elev":      _fv(elev, 0, " m"),
        "slope":     _fv(slope, 1, "°"),
        "kpis": [
            {"name": "Land Surface Temp", "abbr": "LST", "value": _fv(lst, 1), "unit": "°C",   "label": lst_lbl, "color": lst_col, "note": "Landsat C2L2 · peak summer"},
            {"name": "Heat Index",        "abbr": "HI",  "value": _fv(hi,  1), "unit": "°C",   "label": hi_lbl,  "color": hi_col,  "note": "NOAA Rothfusz · hot-month est."},
            {"name": "Vegetation Cond.",  "abbr": "VCI", "value": _fv(vhi, 0), "unit": "/100",      "label": vci_lbl, "color": vci_col, "note": "Kogan · NDVI-only record"},
            {"name": "Aridity Index",     "abbr": "AI",  "value": _fv(ai,  3), "unit": "P/PET",     "label": ai_lbl,  "color": ai_col,  "note": "UNEP 1992 · Hargreaves PET"},
        ],
        "supporting": [
            {"name": "NDVI", "value": _fv(ndvi, 3), "label": ndvi_lbl, "color": ndvi_col},
            {"name": "SAVI", "value": _fv(savi, 3), "label": None,     "color": "neutral"},
            {"name": "MSI",  "value": _fv(msi,  3), "label": msi_lbl,  "color": msi_col},
        ],
        "climate": [
            {"param": "Mean annual air temperature",      "value": _fv(t_mean,    1, " °C")},
            {"param": "Mean relative humidity",           "value": _fv(rh_mean,   0, " %")},
            {"param": "Mean solar irradiance",            "value": _fv(srad_mean, 2, " kWh/m²/day")},
            {"param": "Mean annual precipitation",        "value": _fv(p_yr,      0, " mm/yr")},
            {"param": "Reference ET (Hargreaves/FAO-56)", "value": _fv(pet_yr,    0, " mm/yr")},
        ],
        "diag_summary": (
            f"<b>{short}</b> exhibits <b>{thermal_level} thermal stress</b>, "
            f"with a Land Surface Temperature of {_fv(lst, 1)}°C ({lst_lbl}) "
            f"and a Heat Index of {_fv(hi, 1)}°C ({hi_lbl}). "
            f"Vegetation cover is <b>{veg_level}</b> "
            f"(NDVI = {_fv(ndvi, 3)}; VCI = {_fv(vhi, 0)}/100 — {vci_lbl}). "
            f"Water availability is <b>{klass_a.lower()}</b> (AI = {_fv(ai, 3)}), "
            f"{'indicating the site structurally lacks sufficient precipitation — NBS design must account for irrigation or water-harvesting' if ai is not None and ai < 0.50 else 'indicating an adequate long-term water balance for NBS establishment'}. "
            f"The planned intervention (<b>{site_data['wp4_action']}</b>) targets the site's primary risk of <b>{site_data['primary_risk'].lower()}</b>."
        ),
        "recommendations": recs,
        "comp_label":  comp_label,
        "comp_col":    comp_col,
        "comp_score":  round(comp, 1) if comp is not None else None,
        "cross_sites": _CROSS,
    }


def generate_report(site: str, lat: float, lon: float,
                    site_data: Dict, metrics: Dict[str, Any],
                    power_df: pd.DataFrame) -> str:
    lst   = metrics.get("LST");  ndvi  = metrics.get("NDVI");  savi  = metrics.get("SAVI")
    msi   = metrics.get("MSI");  vhi   = metrics.get("VCI", metrics.get("VHI"));  ai = metrics.get("Aridity")
    hi    = metrics.get("HeatIndex"); elev = metrics.get("elevation"); slope = metrics.get("slope")

    t_mean = rh_mean = srad_mean = p_yr = pet_yr = None
    if not power_df.empty:
        t_mean    = power_df["T2M"].mean()
        rh_mean   = power_df["RH2M"].mean()
        srad_mean = power_df["ALLSKY_SFC_SW_DWN"].mean()
        _tmp = power_df.copy()
        _tmp["month"] = _tmp["date"].dt.month
        _tmp["p_mm"]  = _tmp["PRECTOTCORR"] * _tmp["month"].map(lambda m: DAYS_IN_MONTH[m - 1])
        p_yr = _tmp.groupby(_tmp["date"].dt.year)["p_mm"].sum().mean()
        pet_s = hargreaves_pet(power_df, lat)
        if pet_s.empty:
            pet_s = thornthwaite_pet(power_df, lat)
        if not pet_s.empty:
            _pet = power_df.loc[pet_s.index, ["date"]].copy()
            _pet["pet"] = pet_s.values
            pet_yr = _pet.groupby(_pet["date"].dt.year)["pet"].sum().mean()

    klass_a, _ = aridity_class(ai)
    fmt = lambda v, p=2, u="": (
        f"{v:.{p}f}{u}" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "n/a"
    )
    today = datetime.utcnow().strftime("%d %B %Y")
    short = SITE_SHORT.get(site, site)

    def _lst_class(v):
        if v is None: return "Unknown",      "⬜"
        if v < 28:    return "Normal",        "🟢"
        if v < 32:    return "Elevated",      "🟡"
        if v < 40:    return "High",          "🟠"
        return             "Critical",       "🔴"

    def _hi_class(v):
        if v is None: return "Unknown",      "⬜"
        if v < 27:    return "Comfortable",  "🟢"
        if v < 32:    return "Caution",      "🟡"
        if v < 41:    return "High Risk",    "🟠"
        return             "Danger",        "🔴"

    def _vhi_class(v):
        if v is None: return "Unknown",          "⬜"
        if v < 25:    return "Severe stress",    "🔴"
        if v < 40:    return "Moderate stress",  "🟠"
        if v < 60:    return "Fair",             "🟡"
        return             "Good",               "🟢"

    def _ai_class(v):
        if v is None: return "Unknown",       "⬜"
        if v >= 0.65: return "Humid",         "🟢"
        if v >= 0.50: return "Dry sub-humid", "🟡"
        if v >= 0.20: return "Semi-arid",     "🟠"
        if v >= 0.05: return "Arid",          "🔴"
        return             "Hyper-arid",     "🔴"

    def _ndvi_class(v):
        if v is None: return "Unknown",  "⬜"
        if v < 0.2:   return "Sparse",   "🔴"
        if v < 0.35:  return "Low",      "🟠"
        if v < 0.5:   return "Moderate", "🟡"
        return             "Dense",     "🟢"

    def _msi_class(v):
        if v is None: return "Unknown",      "⬜"
        if v < 0.6:   return "Well-watered", "🟢"
        if v < 1.0:   return "Low stress",   "🟡"
        if v < 1.5:   return "Moderate",     "🟠"
        return             "High stress",   "🔴"

    lst_lbl,  lst_ico  = _lst_class(lst)
    hi_lbl,   hi_ico   = _hi_class(hi)
    vhi_lbl,  vhi_ico  = _vhi_class(vhi)
    ai_lbl,   ai_ico   = _ai_class(ai)
    ndvi_lbl, ndvi_ico = _ndvi_class(ndvi)
    msi_lbl,  msi_ico  = _msi_class(msi)

    recs, cats = [], []
    if lst is not None and lst > 32:
        recs.append("**Urban green corridors and cool pavements** to mitigate surface heat island effect.")
        cats.append("Urban Cooling NBS")
    if hi is not None and hi > 35:
        recs.append("**Green roofs and urban forest canopy** to reduce ambient air temperature and heat stress.")
        cats.append("Green Infrastructure")
    if vhi is not None and vhi < 40:
        recs.append("**Drought-tolerant native afforestation** to rebuild vegetation resilience.")
        cats.append("Ecological Restoration")
    if ai is not None and ai < 0.50:
        recs.append("**Constructed wetlands and water-sensitive urban design** for precipitation capture and groundwater recharge.")
        cats.append("Water-Sensitive Design")
    if msi is not None and msi > 1.5:
        recs.append("**Mulching, biochar amendment, and soil-moisture conservation** to address vegetation water stress.")
        cats.append("Soil Management NBS")
    if slope is not None and slope > 8:
        recs.append("**Bioengineered terraces and riparian buffer strips** for slope stabilisation and erosion control.")
        cats.append("Slope Stabilisation NBS")
    if not recs:
        recs.append("**Preventive NBS** (green roofs, permeable surfaces, pocket parks) — compound stress is currently low.")
        cats.append("Preventive NBS")

    scores = []
    if lst is not None: scores.append(min(100, max(0, (lst - 20) / 30 * 100)))
    if vhi is not None: scores.append(min(100, max(0, (40 - vhi) / 40 * 100)))
    if ai  is not None: scores.append(min(100, max(0, (0.65 - ai) / 0.65 * 100)))
    comp = sum(scores) / len(scores) if scores else None
    if   comp is None: comp_label = "Unknown"
    elif comp < 25:    comp_label = "LOW"
    elif comp < 50:    comp_label = "MODERATE"
    elif comp < 75:    comp_label = "HIGH"
    else:              comp_label = "CRITICAL"

    thermal_level = (
        "high"     if (lst is not None and lst > 36) or (hi is not None and hi > 38) else
        "moderate" if (lst is not None and lst > 32) or (hi is not None and hi > 32) else "low"
    )
    veg_level = (
        "sparse and stressed" if ndvi is not None and ndvi < 0.20 else
        "below moderate"      if ndvi is not None and ndvi < 0.35 else
        "moderate"            if ndvi is not None and ndvi < 0.50 else "healthy"
    )
    diag_summary = (
        f"This site — **{short}** — exhibits **{thermal_level}** thermal stress, "
        f"with a Land Surface Temperature of {fmt(lst, 1, 'C')} ({lst_lbl}) "
        f"and a Heat Index of {fmt(hi, 1, 'C')} ({hi_lbl}). "
        f"Vegetation cover is **{veg_level}** "
        f"(NDVI = {fmt(ndvi, 3)}; VCI = {fmt(vhi, 0, '/100')} — {vhi_lbl}). "
        f"Water availability is **{klass_a.lower()}** (Aridity Index = {fmt(ai, 3)}), "
        f"{'meaning the site structurally lacks sufficient precipitation — NBS design must account for irrigation or water-harvesting' if ai is not None and ai < 0.50 else 'indicating an adequate long-term water balance for NBS establishment'}. "
        f"The planned intervention (**{site_data['wp4_action']}**) targets the site's primary "
        f"risk of **{site_data['primary_risk'].lower()}**."
    )

    _CROSS = [
        {"short": "Cerdanyola", "flag": "🇪🇸", "country": "Spain",    "aridity": "Semi-arid",    "ai_v": 0.45, "lst_v": 36.2, "hi_v": 34.8, "vhi_v": 35, "ndvi_v": 0.22, "primary_risk": "Heat stress",    "nbs": "Urban forest (Miyawaki)"},
        {"short": "Izmir",      "flag": "🇹🇷", "country": "Turkiye",  "aridity": "Dry sub-humid", "ai_v": 0.56, "lst_v": 37.8, "hi_v": 37.5, "vhi_v": 31, "ndvi_v": 0.19, "primary_risk": "Heat stress",    "nbs": "Green roof"},
        {"short": "Sikionies",  "flag": "🇬🇷", "country": "Greece",   "aridity": "Dry sub-humid", "ai_v": 0.53, "lst_v": 34.9, "hi_v": 35.2, "vhi_v": 38, "ndvi_v": 0.31, "primary_risk": "Flood & heat",  "nbs": "Smart eco-park"},
        {"short": "Tunis",      "flag": "🇹🇳", "country": "Tunisia",  "aridity": "Semi-arid",    "ai_v": 0.32, "lst_v": 40.1, "hi_v": 37.0, "vhi_v": 26, "ndvi_v": 0.15, "primary_risk": "Flooding",       "nbs": "Water retention basin"},
        {"short": "Heliopolis", "flag": "🇪🇬", "country": "Egypt",    "aridity": "Arid",         "ai_v": 0.09, "lst_v": 44.6, "hi_v": 42.8, "vhi_v": 12, "ndvi_v": 0.08, "primary_risk": "Heat + drought", "nbs": "Green corridor (Miyawaki)"},
    ]
    _sn = short.lower().replace("İ", "i").replace("ı", "i")
    for row in _CROSS:
        if row["short"].lower() in _sn or _sn in row["short"].lower():
            if lst  is not None and not (isinstance(lst,  float) and pd.isna(lst)):  row["lst_v"]  = lst
            if hi   is not None and not (isinstance(hi,   float) and pd.isna(hi)):   row["hi_v"]   = hi
            if vhi  is not None and not (isinstance(vhi,  float) and pd.isna(vhi)):  row["vhi_v"]  = vhi
            if ndvi is not None and not (isinstance(ndvi, float) and pd.isna(ndvi)): row["ndvi_v"] = ndvi
            if ai   is not None and not (isinstance(ai,   float) and pd.isna(ai)):   row["ai_v"]   = ai
            break

    cross_rows = ""
    for row in _CROSS:
        marker = " **(This site)**" if (row["short"].lower() in _sn or _sn in row["short"].lower()) else ""
        cross_rows += (f"| {row['flag']} {row['country']} — {row['short']}{marker} "
                       f"| {row['aridity']} | {row['ai_v']:.2f} "
                       f"| {row['lst_v']:.1f} | {row['hi_v']:.1f} "
                       f"| {row['vhi_v']:.0f}/100 | {row['ndvi_v']:.2f} |\n")

    sep = "---"
    return (
        f"# NBS4MED Diagnostic Report\n{sep}\n\n"
        f"## Section 1: Introduction & Project Context\n\n"
        f"This dashboard provides the baseline environmental analysis of the five NBS4MED pilot sites "
        f"in Spain, Turkey, Greece, Tunisia, and Egypt, before any intervention takes place. "
        f"It measures the main climate risks at each site, including heat stress, drought, vegetation loss, "
        f"and flood risk, and turns them into the data inputs needed for the Self-Evaluation Tool "
        f"(Activity 3.3.2).\n\n{sep}\n\n"
        f"## Section 2: {short} — Baseline Environmental Diagnostic\n\n"
        f"### Site Information\n"
        f"| Parameter | Value |\n|---|---|\n"
        f"| **WP4 Action** | {site_data['wp4_action']} |\n"
        f"| **Intervention area** | {site_data['area_m2']:,} m² |\n"
        f"| **Project partner** | {site_data['partner']} |\n"
        f"| **Coordinates** | {lat:.4f}N, {lon:.4f}E |\n"
        f"| **Elevation** | {fmt(elev, 0, ' m a.s.l.')} |\n"
        f"| **Terrain slope** | {fmt(slope, 1, ' deg')} |\n"
        f"| **Primary climate risk** | {site_data['primary_risk']} |\n\n"
        f"### Key Performance Indicators\n\n"
        f"| Indicator | Value | Status | What it measures |\n|---|---|---|---|\n"
        f"| **Land Surface Temperature (LST)** | {fmt(lst, 1, ' °C')} | {lst_ico} {lst_lbl} | Measures how hot the ground surface is |\n"
        f"| **Heat Index** | {fmt(hi, 1, ' °C')} | {hi_ico} {hi_lbl} | Combines air temperature and humidity to show how hot it actually feels |\n"
        f"| **Vegetation Condition (VCI)** | {fmt(vhi, 0, '/100')} | {vhi_ico} {vhi_lbl} | NDVI-based vegetation condition, 0–100 (observed record only) |\n"
        f"| **Aridity Index (P/PET)** | {fmt(ai, 3)} | {ai_ico} {ai_lbl} | Ratio of precipitation to evaporation demand |\n\n"
        f"*Supporting indices:* NDVI = {fmt(ndvi, 3)} ({ndvi_ico} {ndvi_lbl}) "
        f"| SAVI = {fmt(savi, 3)} | MSI = {fmt(msi, 3)} ({msi_ico} {msi_lbl})\n\n"
        f"### Diagnostic Summary\n\n{diag_summary}\n\n"
        f"### Climatological Profile (NASA POWER MERRA-2, 30-year baseline)\n"
        f"| Parameter | Value |\n|---|---|\n"
        f"| Mean annual air temperature | {fmt(t_mean, 1, ' °C')} |\n"
        f"| Mean relative humidity | {fmt(rh_mean, 0, ' %')} |\n"
        f"| Mean solar irradiance | {fmt(srad_mean, 2, ' kWh/m²/day')} |\n"
        f"| Mean annual precipitation | {fmt(p_yr, 0, ' mm/yr')} |\n"
        f"| Reference ET — Hargreaves/FAO-56 | {fmt(pet_yr, 0, ' mm/yr')} |\n\n"
        f"### NBS Recommendations (Compound Risk: {comp_label})\n\n" + "\n".join(f"- {r}" for r in recs) + "\n\n"
        f"{sep}\n\n"
        f"## Section 3: Comparative Baseline Analysis — All Pilot Sites\n\n"
        f"*Values for sites other than {short} are representative baseline estimates. "
        f"Issued: {today}.*\n\n"
        f"| Site | Aridity Zone | AI | LST (°C) | Heat Index (°C) | VCI (/100) | NDVI |\n"
        f"|---|---|---|---|---|---|---|\n{cross_rows}\n"
        f"{sep}\n\n"
        f"*NBS4MED | Interreg NEXT MED Programme | University of Jordan (PP07) | Activity 3.3.2*  \n"
        f"*Pre-intervention baseline | Issued: {today}*\n"
    )


# ════════════════════════════════════════════════════════════════════════════
#  PDF REPORT (exact logic from app1.py — image path updated for Flask)
# ════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(site: str, lat: float, lon: float,
                        site_data: Dict, metrics: Dict[str, Any],
                        power_df: pd.DataFrame,
                        ndvi_df: Optional[pd.DataFrame] = None) -> Optional[bytes]:
    if not REPORTLAB_OK:
        return None
    from io import BytesIO

    lst   = metrics.get("LST");  ndvi  = metrics.get("NDVI");  savi  = metrics.get("SAVI")
    msi   = metrics.get("MSI");  vhi   = metrics.get("VCI", metrics.get("VHI"));  ai = metrics.get("Aridity")
    hi    = metrics.get("HeatIndex"); elev = metrics.get("elevation"); slope = metrics.get("slope")

    t_mean = rh_mean = srad_mean = p_yr = pet_yr = None
    if not power_df.empty:
        t_mean    = power_df["T2M"].mean()
        rh_mean   = power_df["RH2M"].mean()
        srad_mean = power_df["ALLSKY_SFC_SW_DWN"].mean()
        _tmp = power_df.copy()
        _tmp["month"] = _tmp["date"].dt.month
        _tmp["p_mm"]  = _tmp["PRECTOTCORR"] * _tmp["month"].map(lambda m: DAYS_IN_MONTH[m - 1])
        p_yr = _tmp.groupby(_tmp["date"].dt.year)["p_mm"].sum().mean()
        pet_s = hargreaves_pet(power_df, lat)
        if pet_s.empty:
            pet_s = thornthwaite_pet(power_df, lat)
        if not pet_s.empty:
            _pet = power_df.loc[pet_s.index, ["date"]].copy()
            _pet["pet"] = pet_s.values
            pet_yr = _pet.groupby(_pet["date"].dt.year)["pet"].sum().mean()

    def _trend_precip():
        if power_df.empty: return "n/a", "#64748b"
        _t = power_df.copy()
        _t["month"] = _t["date"].dt.month
        _t["p_mm"]  = _t["PRECTOTCORR"] * _t["month"].map(lambda m: DAYS_IN_MONTH[m - 1])
        ann = _t.groupby(_t["date"].dt.year)["p_mm"].sum()
        if len(ann) < 4: return "n/a", "#64748b"
        sl = float(np.polyfit(range(len(ann)), ann.values, 1)[0])
        if sl > 5:  return f"Increasing (+{sl:.0f} mm/yr)", "#11caa0"
        if sl < -5: return f"Declining ({sl:.0f} mm/yr)", "#ef4444"
        return "Stable", "#64748b"

    def _trend_ndvi():
        if ndvi_df is None or ndvi_df.empty or len(ndvi_df) < 6: return "n/a", "#64748b"
        sl = float(np.polyfit(range(len(ndvi_df)), ndvi_df["ndvi"].ffill().values, 1)[0])
        if sl > 0.002:  return f"Improving (+{sl:.4f}/yr)", "#11caa0"
        if sl < -0.002: return f"Declining ({sl:.4f}/yr)", "#ef4444"
        return "Stable", "#64748b"

    p_trend_txt,  p_trend_col  = _trend_precip()
    nd_trend_txt, nd_trend_col = _trend_ndvi()

    def _heat_risk(v):
        if v is None: return "Unknown",    0.0,  "#64748b"
        if v < 28:    return "Low",        0.18, "#11caa0"
        if v < 32:    return "Moderate",   0.44, "#f59e0b"
        if v < 40:    return "High",       0.72, "#ef4444"
        return             "Critical",    1.00, "#ef4444"

    def _hi_risk(v):
        if v is None: return "Unknown",     0.0,  "#64748b"
        if v < 27:    return "Comfortable", 0.15, "#11caa0"
        if v < 32:    return "Caution",     0.42, "#f59e0b"
        if v < 41:    return "High Risk",   0.72, "#ef4444"
        return             "Danger",       1.00, "#ef4444"

    def _veg_risk(v):
        if v is None: return "Unknown",     0.0,  "#64748b"
        if v >= 40:   return "Healthy",     0.12, "#11caa0"
        if v >= 30:   return "Mild stress", 0.35, "#f59e0b"
        if v >= 20:   return "Moderate",    0.58, "#f59e0b"
        if v >= 10:   return "Severe",      0.80, "#ef4444"
        return             "Extreme",      1.00, "#ef4444"

    def _arid_risk(v):
        if v is None: return "Unknown",       0.0,  "#64748b"
        if v >= 0.65: return "Humid",         0.08, "#11caa0"
        if v >= 0.50: return "Dry sub-humid", 0.28, "#11caa0"
        if v >= 0.20: return "Semi-arid",     0.55, "#f59e0b"
        if v >= 0.05: return "Arid",          0.80, "#ef4444"
        return             "Hyper-arid",     1.00, "#ef4444"

    hl, hf, hc  = _heat_risk(lst)
    il, inf, ic = _hi_risk(hi)
    vl, vf, vc  = _veg_risk(vhi)
    al, af, ac  = _arid_risk(ai)

    scores = []
    if lst is not None: scores.append(min(100, max(0, (lst - 20) / 30 * 100)))
    if vhi is not None: scores.append(min(100, max(0, (40 - vhi) / 40 * 100)))
    if ai  is not None: scores.append(min(100, max(0, (0.65 - ai) / 0.65 * 100)))
    comp = sum(scores) / len(scores) if scores else None
    if   comp is None: comp_label, comp_col = "UNKNOWN",  "#64748b"
    elif comp < 25:    comp_label, comp_col = "LOW",      "#11caa0"
    elif comp < 50:    comp_label, comp_col = "MODERATE", "#f59e0b"
    elif comp < 75:    comp_label, comp_col = "HIGH",     "#ef4444"
    else:              comp_label, comp_col = "CRITICAL", "#ef4444"

    recs, cats = [], []
    if lst is not None and lst > 32:
        recs.append("Urban green corridors and cool pavements to mitigate surface heat island effect.")
        cats.append("Urban Cooling NBS")
    if hi is not None and hi > 35:
        recs.append("Green roofs and urban forest canopy to reduce ambient air temperature.")
        cats.append("Green Infrastructure")
    if vhi is not None and vhi < 40:
        recs.append("Drought-tolerant native afforestation to restore vegetation resilience.")
        cats.append("Ecological Restoration")
    if ai is not None and ai < 0.50:
        recs.append("Constructed wetlands and water-sensitive urban design for precipitation retention.")
        cats.append("Water-Sensitive Design")
    if msi is not None and msi > 1.5:
        recs.append("Mulching and biochar amendment to address soil moisture stress.")
        cats.append("Soil Management NBS")
    if slope is not None and slope > 8:
        recs.append("Bioengineered terraces and buffer strips for slope stabilisation.")
        cats.append("Slope Stabilisation NBS")
    if not recs:
        recs.append("Low compound stress — preventive NBS (green roofs, permeable surfaces) recommended.")
        cats.append("Preventive NBS")

    fmt   = lambda v, p=1, u="": (f"{v:.{p}f}{u}" if v is not None
                                   and not (isinstance(v, float) and pd.isna(v)) else "n/a")
    short = SITE_SHORT.get(site, site)
    # Images live in flask_app/static/images/
    base  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "images")
    today = datetime.utcnow().strftime("%d %B %Y")

    buf  = BytesIO()
    cv   = rl_canvas.Canvas(buf, pagesize=RL_A4)
    W, H = RL_A4
    L, R = 30.0, W - 30.0
    CW   = R - L

    def hx(s): return RLHex(s)

    HDR_H = 58; HDR_Y = H - HDR_H
    cv.setFillColor(hx("#005088")); cv.rect(0, HDR_Y, W, HDR_H, fill=1, stroke=0)
    cv.setFillColor(hx("#11caa0")); cv.rect(0, H - 4, W, 4, fill=1, stroke=0)
    cv.setFillColor(rl_white); cv.setFont("Helvetica-Bold", 20); cv.drawString(L, HDR_Y + 30, "NBS4MED")
    cv.setFont("Helvetica", 9); cv.drawString(L, HDR_Y + 16, "Interreg NEXT MED  ·  Site Diagnostic Report  ·  PROMEA Framework")
    cv.setFillColor(hx("#a8c4d8")); cv.setFont("Helvetica", 8); cv.drawRightString(R, HDR_Y + 10, f"Generated: {today}")

    for fname, x0, w0 in [("interreg.png", R - 124, 52), ("uj_logo.png", R - 64, 52)]:
        try:
            p = os.path.join(base, fname)
            if os.path.exists(p):
                cv.drawImage(p, x0, HDR_Y + 6, width=w0, height=44, mask="auto")
        except Exception:
            pass

    SITE_Y = HDR_Y - 34
    cv.setFillColor(hx("#f8fafc")); cv.rect(0, SITE_Y, W, 34, fill=1, stroke=0)
    cv.setStrokeColor(hx("#e2e8f0")); cv.setLineWidth(0.5)
    cv.line(0, SITE_Y, W, SITE_Y); cv.line(0, SITE_Y + 34, W, SITE_Y + 34)
    cv.setFillColor(hx("#0f172a")); cv.setFont("Helvetica-Bold", 12); cv.drawString(L, SITE_Y + 18, site)
    cv.setFillColor(hx("#64748b")); cv.setFont("Helvetica", 8)
    cv.drawString(L, SITE_Y + 6, f"{lat:.4f}°N, {lon:.4f}°E  ·  Area: {site_data['area_m2']:,} m²  ·  Partner: {site_data['partner']}")
    bw = 108; cv.setFillColor(hx(comp_col)); cv.roundRect(R - bw, SITE_Y + 7, bw, 20, 4, fill=1, stroke=0)
    cv.setFillColor(rl_white); cv.setFont("Helvetica-Bold", 8)
    cv.drawCentredString(R - bw / 2, SITE_Y + 13, f"COMPOUND RISK: {comp_label}")

    y = SITE_Y - 14

    def sec(title, yp):
        cv.setFillColor(hx("#005088")); cv.setFont("Helvetica-Bold", 8); cv.drawString(L, yp, title)
        cv.setStrokeColor(hx("#11caa0")); cv.setLineWidth(1.2); cv.line(L, yp - 3, R, yp - 3)
        cv.setLineWidth(0.5); return yp - 14

    def risk_row(label, bar_fill, bar_col, value_txt, yp):
        BAR_X, BAR_W, BAR_H = L + 108, 155, 7
        cv.setFillColor(hx("#0f172a")); cv.setFont("Helvetica", 8); cv.drawString(L + 2, yp + 1, label)
        cv.setFillColor(hx("#e2e8f0")); cv.roundRect(BAR_X, yp - 1, BAR_W, BAR_H, 2, fill=1, stroke=0)
        if bar_fill > 0:
            cv.setFillColor(hx(bar_col)); cv.roundRect(BAR_X, yp - 1, max(6, int(BAR_W * bar_fill)), BAR_H, 2, fill=1, stroke=0)
        cv.setFillColor(hx("#0f172a")); cv.setFont("Helvetica-Bold", 8)
        cv.drawString(BAR_X + BAR_W + 6, yp + 1, value_txt); return yp - 16

    y = sec("RISK CLASSIFICATION", y)
    y = risk_row("Heat Stress  (LST)",           hf, hc,  f"{fmt(lst,1)}°C → {hl}", y)
    y = risk_row("Thermal Comfort  (Heat Index)", inf, ic, f"{fmt(hi,1)}°C → {il}", y)
    y = risk_row("Vegetation Condition  (VCI)",   vf, vc,  f"{fmt(vhi,0)}/100 → {vl}", y)
    y = risk_row("Water Stress  (Aridity Index)", af, ac,  f"AI={fmt(ai,3)} → {al}", y)
    y -= 5

    y = sec("KEY INDICATORS", y)
    _cw2 = (CW - 6) / 2
    _kpi_cards = [
        (fmt(lst, 1, "°C"), hl, hc, "LAND SURFACE TEMPERATURE (LST)",
         "How hot the ground surface gets.", "Indicates urban heat island intensity."),
        (fmt(hi, 1, "°C"), il, ic, "HEAT INDEX",
         "How hot it FEELS to people (air temp + humidity).", "Values above 41°C are dangerous."),
        (fmt(vhi, 0, "/100"), vl, vc, "VEGETATION CONDITION INDEX (VCI)",
         "NDVI-based vegetation condition 0–100.", "Below 40 = moderate/severe stress."),
        (fmt(ai, 3), al, ac, "ARIDITY INDEX  (P/PET)",
         "Water balance: precipitation ÷ evaporation demand.", "Below 0.5 = arid/semi-arid."),
    ]
    for _i, (_val, _lbl, _col, _title, _ex1, _ex2) in enumerate(_kpi_cards):
        _kx = L + (_i % 2) * (_cw2 + 6)
        _ky = y - (_i // 2) * 41
        cv.setFillColor(hx("#f0f7ff")); cv.roundRect(_kx, _ky - 38, _cw2, 39, 4, fill=1, stroke=0)
        cv.setFillColor(hx(_col));      cv.roundRect(_kx, _ky - 38, 3, 39, 2, fill=1, stroke=0)
        cv.setFillColor(hx("#005088")); cv.setFont("Helvetica-Bold", 6.5); cv.drawString(_kx + 9, _ky - 8, _title)
        cv.setFillColor(hx(_col));      cv.setFont("Helvetica-Bold", 11); cv.drawString(_kx + 9, _ky - 19, _val)
        cv.setFont("Helvetica-Bold", 7.5); cv.drawRightString(_kx + _cw2 - 6, _ky - 19, _lbl)
        cv.setFillColor(hx("#475569")); cv.setFont("Helvetica-Oblique", 6.5)
        cv.drawString(_kx + 9, _ky - 28, _ex1); cv.drawString(_kx + 9, _ky - 36, _ex2)
    y -= 2 * 41 + 8

    _sec_kpis = [("NDVI", fmt(ndvi, 3)), ("SAVI", fmt(savi, 3)),
                 ("MSI",  fmt(msi,  3)), ("Elev", fmt(elev, 0, " m")), ("Slope", fmt(slope, 1, "°"))]
    _sw = CW / 5
    for _si, (_sn, _sv) in enumerate(_sec_kpis):
        _sx = L + _si * _sw
        cv.setFillColor(hx("#f8fafc")); cv.roundRect(_sx + 2, y - 19, _sw - 4, 20, 3, fill=1, stroke=0)
        cv.setFillColor(hx("#64748b")); cv.setFont("Helvetica", 6.5); cv.drawCentredString(_sx + _sw / 2, y - 7, _sn)
        cv.setFillColor(hx("#0f172a")); cv.setFont("Helvetica-Bold", 8.5); cv.drawCentredString(_sx + _sw / 2, y - 17, _sv)
    y -= 28

    y = sec("CLIMATE PROFILE  (NASA POWER MERRA-2, 30-year baseline)", y)
    climate = [("Mean Temp", fmt(t_mean, 1, "°C")), ("Humidity", fmt(rh_mean, 0, "%")),
               ("Solar Rad", fmt(srad_mean, 2, " kWh/m²/d")), ("Annual P", fmt(p_yr, 0, " mm")),
               ("PET (Hargreaves)", fmt(pet_yr, 0, " mm/yr"))]
    clw = CW / len(climate)
    for i, (nm, vl_) in enumerate(climate):
        cx = L + i * clw
        cv.setFillColor(hx("#eef2ff")); cv.roundRect(cx + 2, y - 24, clw - 4, 26, 3, fill=1, stroke=0)
        cv.setFillColor(hx("#64748b")); cv.setFont("Helvetica", 7); cv.drawCentredString(cx + clw / 2, y - 9, nm)
        cv.setFillColor(hx("#005088")); cv.setFont("Helvetica-Bold", 9); cv.drawCentredString(cx + clw / 2, y - 20, vl_)
    y -= 34

    y = sec("TREND DIRECTION", y)
    for label, txt, col in [("Precipitation:", p_trend_txt, p_trend_col),
                              ("Vegetation (NDVI):", nd_trend_txt, nd_trend_col)]:
        cv.setFillColor(hx("#0f172a")); cv.setFont("Helvetica", 8); cv.drawString(L + 2, y + 1, label)
        cv.setFillColor(hx(col));       cv.setFont("Helvetica-Bold", 8); cv.drawString(L + 102, y + 1, txt)
        y -= 13
    y -= 4

    y = sec("WP4 PILOT ACTION", y)
    cv.setFillColor(hx("#f0fdf4")); cv.roundRect(L, y - 22, CW, 24, 3, fill=1, stroke=0)
    cv.setFillColor(hx("#11caa0")); cv.setLineWidth(2.5); cv.line(L + 1, y - 22, L + 1, y + 2)
    cv.setLineWidth(0.5); cv.setFillColor(hx("#0f172a")); cv.setFont("Helvetica-Bold", 9)
    cv.drawString(L + 8, y - 7, site_data["wp4_action"])
    cv.setFillColor(hx("#64748b")); cv.setFont("Helvetica", 7)
    cv.drawString(L + 8, y - 17, f"Primary risk: {site_data['primary_risk']}  ·  Partner: {site_data['partner']}")
    y -= 32

    cv.setFillColor(hx("#005088")); cv.rect(0, 0, W, 20, fill=1, stroke=0)
    cv.setFillColor(rl_white); cv.setFont("Helvetica", 7.5)
    cv.drawCentredString(W / 2, 6, "NBS4MED  ·  Interreg NEXT MED Programme  ·  University of Jordan (PP7)")
    cv.setFillColor(hx("#11caa0")); cv.setLineWidth(3); cv.line(0, 20, 0, H)

    # ── PAGE 2: Cross-site comparison table ──────────────────────────────
    cv.showPage()
    cv.setFillColor(hx("#005088")); cv.rect(0, H - 58, W, 58, fill=1, stroke=0)
    cv.setFillColor(hx("#11caa0")); cv.rect(0, H - 4, W, 4, fill=1, stroke=0)
    cv.setFillColor(rl_white); cv.setFont("Helvetica-Bold", 14)
    cv.drawString(L, H - 34, "Comparative Baseline Analysis — All Pilot Sites")
    cv.setFont("Helvetica", 8); cv.drawString(L, H - 48, f"NBS4MED  ·  Pre-intervention baseline  ·  {today}")

    _sn2 = short.lower().replace("İ", "i").replace("ı", "i")
    _TY = H - 80
    cv.setFont("Helvetica-Bold", 7.5); cv.setFillColor(hx("#005088"))
    cv.drawString(L, _TY + 4, "CROSS-SITE BASELINE DATA TABLE")
    cv.setStrokeColor(hx("#11caa0")); cv.setLineWidth(1.2); cv.line(L, _TY, R, _TY); _TY -= 16

    _TCOLS = ["Site", "Country", "AI", "LST (C)", "HI (C)", "VCI /100", "NDVI", "Primary Risk", "NBS Type"]
    _TCWS  = [105, 46, 28, 38, 36, 42, 30, 76, 94]
    cv.setFillColor(hx("#005088")); cv.rect(L, _TY - 12, sum(_TCWS), 13, fill=1, stroke=0)
    cv.setFillColor(rl_white); cv.setFont("Helvetica-Bold", 6.5)
    _tx = L
    for _h, _cw in zip(_TCOLS, _TCWS):
        cv.drawString(_tx + 2, _TY - 8, _h); _tx += _cw
    _TY -= 12

    _TROWS = [
        ("Spain — Cerdanyola",  "Spain",   0.45, 36.2, 34.8, 35,  0.22, "Heat stress",    "Urban forest"),
        ("Turkiye — Izmir",     "Turkiye", 0.56, 37.8, 37.5, 31,  0.19, "Heat stress",    "Green roof"),
        ("Greece — Sikionies",  "Greece",  0.53, 34.9, 35.2, 38,  0.31, "Flood & heat",   "Smart eco-park"),
        ("Tunisia — Tunis",     "Tunisia", 0.32, 40.1, 37.0, 26,  0.15, "Flooding",       "Water retention"),
        ("Egypt — Heliopolis",  "Egypt",   0.09, 44.6, 42.8, 12,  0.08, "Heat + drought", "Green corridor"),
    ]
    _TROWS_L = list(_TROWS)
    for _ri, _row in enumerate(_TROWS_L):
        _rs = _row[0].split("—")[1].strip().lower()
        if _rs in _sn2 or _sn2 in _rs:
            _TROWS_L[_ri] = (
                _row[0] + " *", _row[1],
                ai   if ai   is not None and not (isinstance(ai,   float) and pd.isna(ai))   else _row[2],
                lst  if lst  is not None and not (isinstance(lst,  float) and pd.isna(lst))  else _row[3],
                hi   if hi   is not None and not (isinstance(hi,   float) and pd.isna(hi))   else _row[4],
                vhi  if vhi  is not None and not (isinstance(vhi,  float) and pd.isna(vhi))  else _row[5],
                ndvi if ndvi is not None and not (isinstance(ndvi, float) and pd.isna(ndvi)) else _row[6],
                _row[7], _row[8]
            ); break

    _last_ry = _TY
    for _ri, _row in enumerate(_TROWS_L):
        _ry = _TY - _ri * 13; _last_ry = _ry
        cv.setFillColor(hx("#f8fafc") if _ri % 2 == 0 else rl_white)
        cv.rect(L, _ry - 12, sum(_TCWS), 13, fill=1, stroke=0)
        cv.setFillColor(hx("#0f172a")); cv.setFont("Helvetica", 6.5)
        _tx = L
        _cells = [str(_row[0])[:22], str(_row[1]), f"{float(_row[2]):.2f}", f"{float(_row[3]):.1f}",
                  f"{float(_row[4]):.1f}", f"{int(float(_row[5]))}", f"{float(_row[6]):.2f}",
                  str(_row[7])[:14], str(_row[8])[:16]]
        for _val, _cw in zip(_cells, _TCWS):
            cv.drawString(_tx + 2, _ry - 8, _val); _tx += _cw

    cv.setStrokeColor(hx("#e2e8f0")); cv.setLineWidth(0.5)
    cv.rect(L, _last_ry - 12, sum(_TCWS), _TY - _last_ry + 12, stroke=1, fill=0)
    cv.setFillColor(hx("#64748b")); cv.setFont("Helvetica", 6)
    cv.drawString(L, _last_ry - 22,
                  "* Actual computed values for currently selected site. All other values: representative baseline estimates.")

    cv.setFillColor(hx("#005088")); cv.rect(0, 0, W, 20, fill=1, stroke=0)
    cv.setFillColor(rl_white); cv.setFont("Helvetica", 7.5)
    cv.drawCentredString(W / 2, 6, "NBS4MED  ·  Interreg NEXT MED Programme  ·  University of Jordan (PP7)")
    cv.setFillColor(hx("#11caa0")); cv.setLineWidth(3); cv.line(0, 20, 0, H)

    cv.save(); buf.seek(0); return buf.read()


# ════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION — replaces main() logic for Flask routes
# ════════════════════════════════════════════════════════════════════════════
def compute_site_metrics(site_name: str,
                         trend_years: int = 10,
                         baseline_years: int = 30) -> Dict[str, Any]:
    """
    Fetch all data and compute all derived indices for one site.
    Returns a dict with keys: metrics, power_df, chirps_df, ndvi_df,
    ee_ok, ee_msg, site_data, lat, lon.
    """
    site_data = PILOT_SITES[site_name]
    lat, lon  = site_data["coords"]

    ee_ok, ee_msg = init_ee()

    end_year    = datetime.utcnow().year - 1
    start_year  = end_year - baseline_years
    trend_months = trend_years * 12
    end_str     = datetime.utcnow().strftime("%Y-%m-%d")
    start_str   = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

    metrics: Dict[str, Any] = {}

    power_df = fetch_nasa_power(lat, lon, start_year, end_year)

    if ee_ok:
        area = site_data["area_m2"]
        ls = fetch_landsat_indices(lat, lon, start_str, end_str, area_m2=area)
        metrics.update({k: v for k, v in ls.items() if k not in ("error", "buffer_m")})
        metrics["landsat_buffer_m"] = ls.get("buffer_m")
        terrain = fetch_terrain(lat, lon, area_m2=area)
        metrics["elevation"] = terrain.get("elevation")
        metrics["slope"]     = terrain.get("slope")
        chirps_df = fetch_chirps_monthly(lat, lon, months=trend_months, area_m2=area)
        ndvi_df   = fetch_ndvi_timeseries(lat, lon, months=trend_months, area_m2=area)
    else:
        chirps_df = pd.DataFrame(columns=["date", "precip"])
        ndvi_df   = pd.DataFrame(columns=["date", "ndvi"])

    if not power_df.empty:
        if "T2M_MAX" in power_df.columns and "RH2M" in power_df.columns:
            summer = power_df[power_df["date"].dt.month.isin([6, 7, 8])]
            src = summer if not summer.empty else power_df
            src = src.copy()
            src["_hi"] = src.apply(
                lambda r: heat_index_noaa(r["T2M_MAX"], r["RH2M"])
                if pd.notna(r["T2M_MAX"]) and pd.notna(r["RH2M"]) else None,
                axis=1,
            )
            hi_series = src["_hi"].dropna()
            metrics["HeatIndex"] = float(hi_series.mean())     if not hi_series.empty else None
            metrics["HI_p95"]    = float(hi_series.quantile(0.95)) if not hi_series.empty else None
        else:
            metrics["HeatIndex"] = None
            metrics["HI_p95"]    = None
        metrics["Aridity"]   = compute_aridity(power_df, lat)
        metrics["LST_zscore"] = compute_thermal_zscore(
            metrics.get("LST"), power_df, "T2M_MAX", summer_only=True)
        metrics["HI_zscore"]  = compute_thermal_zscore(
            metrics.get("HeatIndex"), power_df, "T2M_MAX", summer_only=True)

    if not ndvi_df.empty:
        tmax_max = power_df["T2M_MAX"].max() if (not power_df.empty and "T2M_MAX" in power_df.columns) else 40.0
        tmax_min = power_df["T2M_MIN"].min() if (not power_df.empty and "T2M_MIN" in power_df.columns) else 5.0
        
        metrics["VHI"] = compute_vhi(ndvi_df["ndvi"], metrics.get("LST"), tmax_min, tmax_max + 15.0)
        metrics["VCI"] = compute_vci(ndvi_df["ndvi"])

    return {
        "metrics":   metrics,
        "power_df":  power_df,
        "chirps_df": chirps_df,
        "ndvi_df":   ndvi_df,
        "ee_ok":     ee_ok,
        "ee_msg":    ee_msg,
        "site_data": site_data,
        "lat":       lat,
        "lon":       lon,
    }
