from flask import Flask, render_template, Response, request, jsonify
from datetime import datetime
import io, sys, os
import pandas as pd
import requests as _requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

try:
    from core import (
        generate_report, generate_report_context, generate_pdf_report,
        fetch_nasa_power, heat_index_noaa, compute_aridity, hargreaves_pet,
        compute_site_metrics,
        chart_ndvi_json, chart_spi_json,
        chart_temperature_profile_json, chart_hydro_solar_profile_json,
        chart_stress_radar_json, chart_annual_precip_json,
        DAYS_IN_MONTH,
    )
    _CORE_OK = True
except Exception as _core_err:
    _CORE_OK = False
    _core_err_msg = str(_core_err)

# ── In-memory caches (1 h TTL) ───────────────────────────────────────────────
_power_cache: dict = {}
_metrics_cache: dict = {}
_CACHE_TTL = 3600


def _get_power_df(lat, lon, start_year, end_year):
    key = (round(lat, 4), round(lon, 4), start_year, end_year)
    now = datetime.utcnow().timestamp()
    if key in _power_cache:
        ts, df = _power_cache[key]
        if now - ts < _CACHE_TTL:
            return df
    try:
        df = fetch_nasa_power(lat, lon, start_year, end_year)
    except Exception:
        df = pd.DataFrame()
    _power_cache[key] = (now, df)
    return df


def _get_site_metrics(site_name: str):
    """Cached wrapper around compute_site_metrics (GEE + NASA POWER)."""
    now = datetime.utcnow().timestamp()
    if site_name in _metrics_cache:
        ts, result = _metrics_cache[site_name]
        if now - ts < _CACHE_TTL:
            return result
    try:
        result = compute_site_metrics(site_name)
    except Exception as e:
        result = None
    _metrics_cache[site_name] = (now, result)
    return result


# ── Site metadata (NO scientific values) ────────────────────────────────────
_SITES_META = {
    "spain": {
        "name": "Spain — Cerdanyola del Vallès", "flag": "🇪🇸",
        "pilot_action": "Tiny urban forest (Miyawaki method)",
        "area": "1,850 m²", "partner": "AMB (PP3)",
        "coords": (41.485473, 2.146883),
        "core_name": "🇪🇸  Spain — Cerdanyola del Vallès",
    },
    "turkey": {
        "name": "Türkiye — İzmir (Göztepe)", "flag": "🇹🇷",
        "pilot_action": "Green roof on parking terrace",
        "area": "820 m²", "partner": "IMM (PP2)",
        "coords": (38.397758, 27.112083),
        "core_name": "🇹🇷  Türkiye — İzmir (Göztepe)",
    },
    "greece": {
        "name": "Greece — Sikionies", "flag": "🇬🇷",
        "pilot_action": "Smart eco-park (flood & heat)",
        "area": "2,500 m²", "partner": "REGPEL (LP)",
        "coords": (38.023695, 22.734917),
        "core_name": "🇬🇷  Greece — Sikionies",
    },
    "tunisia": {
        "name": "Tunisia — Greater Tunis", "flag": "🇹🇳",
        "pilot_action": "Water retention basin",
        "area": "2,500 m²", "partner": "CITET (PP5)",
        "coords": (36.838967, 10.211539),
        "core_name": "🇹🇳  Tunisia — Greater Tunis",
    },
    "egypt": {
        "name": "Egypt — Heliopolis University", "flag": "🇪🇬",
        "pilot_action": "Green corridor (Miyawaki method)",
        "area": "2,200 m²", "partner": "SEKEM (PP6)",
        "coords": (30.152959, 31.429408),
        "core_name": "🇪🇬  Egypt — Heliopolis University",
    },
}

ALL_SITES = {k: {"name": v["name"], "flag": v["flag"]}
             for k, v in _SITES_META.items()}


# ── Classification helpers ───────────────────────────────────────────────────
def _kpi(value, fmt, unit, label, color):
    return {"value": value, "fmt": fmt, "unit": unit, "status": label, "color": color}

def _na():
    return {"value": None, "fmt": "—", "unit": "", "status": "No data", "color": "muted"}

def lst_kpi(v, z=None):
    if v is None: return _na()
    if z is not None:
        if z >  2.0: return _kpi(v, f"{v:.1f}", "°C", f"Critical (+{z:.1f}σ)", "red")
        if z >  1.0: return _kpi(v, f"{v:.1f}", "°C", f"High (+{z:.1f}σ)",     "amber")
        if z < -1.0: return _kpi(v, f"{v:.1f}", "°C", f"Below avg ({z:.1f}σ)", "green")
        return             _kpi(v, f"{v:.1f}", "°C", f"Normal ({z:.1f}σ)",    "green")
    if v > 40: return _kpi(v, f"{v:.1f}", "°C", "Critical", "red")
    if v > 32: return _kpi(v, f"{v:.1f}", "°C", "High",     "amber")
    return          _kpi(v, f"{v:.1f}", "°C", "Normal",   "green")

def hi_kpi(v, z=None):
    if v is None: return _na()
    if z is not None:
        if z >  2.0: return _kpi(v, f"{v:.1f}", "°C", f"Danger (+{z:.1f}σ)",   "red")
        if z >  1.0: return _kpi(v, f"{v:.1f}", "°C", f"Caution (+{z:.1f}σ)",  "amber")
        if z < -1.0: return _kpi(v, f"{v:.1f}", "°C", f"Mild ({z:.1f}σ)",      "green")
        return             _kpi(v, f"{v:.1f}", "°C", f"Comfort ({z:.1f}σ)",   "green")
    if v > 40: return _kpi(v, f"{v:.1f}", "°C", "Danger",  "red")
    if v > 32: return _kpi(v, f"{v:.1f}", "°C", "Caution", "amber")
    return          _kpi(v, f"{v:.1f}", "°C", "Comfort", "green")

def vci_kpi(v):
    if v is None: return _na()
    if v < 25:    return _kpi(v, f"{v:.0f}", "/100", "Severe stress",   "red")
    if v < 40:    return _kpi(v, f"{v:.0f}", "/100", "Moderate stress", "amber")
    if v < 60:    return _kpi(v, f"{v:.0f}", "/100", "Fair",            "amber")
    return             _kpi(v, f"{v:.0f}", "/100", "Good",             "green")

def ai_kpi(v):
    if v is None: return _na()
    if v < 0.05:  return _kpi(v, f"{v:.2f}", "P/PET", "Hyper-arid",    "red")
    if v < 0.20:  return _kpi(v, f"{v:.2f}", "P/PET", "Arid",          "red")
    if v < 0.50:  return _kpi(v, f"{v:.2f}", "P/PET", "Semi-arid",     "amber")
    if v < 0.65:  return _kpi(v, f"{v:.2f}", "P/PET", "Dry sub-humid", "amber")
    return             _kpi(v, f"{v:.2f}", "P/PET", "Humid",          "green")


# ── Compute live values from NASA POWER ─────────────────────────────────────
def _compute_from_power(power_df, lat):
    result = {
        "hi": None, "ai": None,
        "climate": {"t_mean": None, "rh": None, "precip": None, "pet": None},
    }
    if not _CORE_OK or power_df.empty:
        return result

    # Heat Index — per-row then average (avoids Jensen's inequality bias)
    if "T2M_MAX" in power_df.columns and "RH2M" in power_df.columns:
        summer = power_df[power_df["date"].dt.month.isin([6, 7, 8])]
        src = (summer if not summer.empty else power_df).copy()
        src["_hi"] = src.apply(
            lambda r: heat_index_noaa(r["T2M_MAX"], r["RH2M"])
            if pd.notna(r["T2M_MAX"]) and pd.notna(r["RH2M"]) else None,
            axis=1,
        )
        hi_series = src["_hi"].dropna()
        result["hi"]     = float(hi_series.mean())          if not hi_series.empty else None
        result["hi_p95"] = float(hi_series.quantile(0.95))  if not hi_series.empty else None

    # Aridity Index
    result["ai"] = compute_aridity(power_df, lat)

    # Climate parameters
    if "T2M" in power_df.columns:
        v = power_df["T2M"].mean()
        result["climate"]["t_mean"] = round(float(v), 1) if pd.notna(v) else None
    if "RH2M" in power_df.columns:
        v = power_df["RH2M"].mean()
        result["climate"]["rh"] = round(float(v), 0) if pd.notna(v) else None
    if "PRECTOTCORR" in power_df.columns:
        df2 = power_df.copy()
        df2["month"] = df2["date"].dt.month
        df2["p_mm"]  = df2["PRECTOTCORR"] * df2["month"].map(
            lambda m: DAYS_IN_MONTH[m - 1])
        ann = df2.groupby(df2["date"].dt.year)["p_mm"].sum()
        result["climate"]["precip"] = round(float(ann.mean()), 0) if not ann.empty else None

    # PET (Hargreaves 1985)
    pet_s = hargreaves_pet(power_df, lat)
    if not pet_s.empty:
        df3 = power_df[["date", "T2M"]].dropna().copy()
        df3["pet"] = pet_s.values
        ann_pet = df3.groupby(df3["date"].dt.year)["pet"].sum()
        result["climate"]["pet"] = round(float(ann_pet.mean()), 0) if not ann_pet.empty else None

    return result


# ── Build site dict for template ─────────────────────────────────────────────
def _build_site(meta, computed):
    lat, lon = meta["coords"]
    return {
        "name":         meta["name"],
        "flag":         meta["flag"],
        "pilot_action": meta["pilot_action"],
        "area":         meta["area"],
        "partner":      meta["partner"],
        "coordinates":  f"{lat:.3f}°N, {lon:.3f}°E",
        "lat":          lat,
        "lon":          lon,
        # KPIs — live computed
        "lst":        _na(),                        # needs GEE
        "heat_index": hi_kpi(computed["hi"]),
        "vegetation": _na(),                        # needs GEE
        "aridity":    ai_kpi(computed["ai"]),
        # Spectral — needs GEE
        "ndvi":       None,
        "savi":       None,
        "msi":        None,
        "elevation":  None,
        "slope":      None,
        # Climate
        "climate":    computed["climate"],
    }


# ── Build site dict from full GEE metrics ────────────────────────────────────
def _build_site_from_metrics(meta, metrics, computed):
    lat, lon = meta["coords"]
    return {
        "name":         meta["name"],
        "flag":         meta["flag"],
        "pilot_action": meta["pilot_action"],
        "area":         meta["area"],
        "partner":      meta["partner"],
        "coordinates":  f"{lat:.3f}°N, {lon:.3f}°E",
        "lat":          lat,
        "lon":          lon,
        # KPIs from GEE where available, NASA POWER as fallback
        "lst":        lst_kpi(metrics.get("LST"),       metrics.get("LST_zscore")),
        "heat_index": hi_kpi(metrics.get("HeatIndex") or computed["hi"],
                             metrics.get("HI_zscore")),
        "vegetation": vci_kpi(metrics.get("VCI")),
        "aridity":    ai_kpi(metrics.get("Aridity")   or computed["ai"]),
        # Spectral from GEE
        "ndvi":       metrics.get("NDVI"),
        "savi":       metrics.get("SAVI"),
        "msi":        metrics.get("MSI"),
        "elevation":  metrics.get("elevation"),
        "slope":      metrics.get("slope"),
        # Climate from NASA POWER
        "climate":    computed["climate"],
    }


# ── NDVI trend DataFrame (static fallback until GEE available) ───────────────
_NDVI_TREND = {
    "spain":   {"labels": ["2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"],
                "values": [0.198,0.204,0.211,0.208,0.215,0.219,0.221,0.218,0.224,0.221]},
    "turkey":  {"labels": ["2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"],
                "values": [0.208,0.201,0.197,0.203,0.196,0.192,0.195,0.190,0.193,0.195]},
    "greece":  {"labels": ["2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"],
                "values": [0.295,0.301,0.308,0.304,0.312,0.318,0.310,0.315,0.320,0.312]},
    "tunisia": {"labels": ["2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"],
                "values": [0.162,0.155,0.151,0.158,0.149,0.145,0.148,0.143,0.146,0.148]},
    "egypt":   {"labels": ["2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"],
                "values": [0.094,0.089,0.086,0.091,0.084,0.080,0.082,0.078,0.081,0.082]},
}


def _ndvi_df(site_id):
    trend = _NDVI_TREND[site_id]
    dates = pd.to_datetime([f"{y}-07-01" for y in trend["labels"]])
    return pd.DataFrame({"date": dates, "ndvi": trend["values"]})


# ── Helpers for report routes ─────────────────────────────────────────────────
def _metrics_for_report(computed, site_id):
    """Build metrics dict for generate_report / generate_pdf_report."""
    return {
        "LST":       None,
        "NDVI":      None,
        "SAVI":      None,
        "MSI":       None,
        "VHI":       None,
        "VCI":       None,
        "Aridity":   computed["ai"],
        "HeatIndex": computed["hi"],
        "elevation": None,
        "slope":     None,
    }


def _site_data_for_report(meta):
    return {
        "coords":       meta["coords"],
        "wp4_action":   meta["pilot_action"],
        "area_m2":      int(meta["area"].replace(",", "").split()[0]),
        "primary_risk": "Heat stress",
        "partner":      meta["partner"],
        "color":        "#005088",
    }


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
@app.route("/site/<site_id>")
def dashboard(site_id="spain"):
    if site_id not in _SITES_META:
        site_id = "spain"
    meta  = _SITES_META[site_id]
    today = datetime.utcnow().strftime("%d %B %Y")

    charts = {"ndvi": None, "spi": None, "temperature": None, "hydro_solar": None, "radar": None, "precip": None}
    site   = None

    if _CORE_OK:
        # Full GEE + NASA POWER computation (same as Streamlit)
        gee_result = _get_site_metrics(meta["core_name"])

        if gee_result:
            metrics   = gee_result["metrics"]
            power_df  = gee_result["power_df"]
            chirps_df = gee_result["chirps_df"]
            ndvi_df   = gee_result["ndvi_df"]
            computed  = _compute_from_power(power_df, meta["coords"][0])

            # Override computed with GEE values where available
            computed["hi"] = metrics.get("HeatIndex") or computed["hi"]
            computed["ai"] = metrics.get("Aridity")   or computed["ai"]

            site = _build_site_from_metrics(meta, metrics, computed)

            charts["ndvi"]        = chart_ndvi_json(ndvi_df if not ndvi_df.empty else _ndvi_df(site_id))
            charts["spi"]         = chart_spi_json(chirps_df, power_df)
            charts["temperature"] = chart_temperature_profile_json(power_df)
            charts["hydro_solar"] = chart_hydro_solar_profile_json(power_df)
            charts["radar"]       = chart_stress_radar_json(metrics, meta["core_name"])
            charts["precip"]      = chart_annual_precip_json(power_df)
        else:
            pass  # falls through to NASA POWER fallback below

    # Fallback: NASA POWER only (no GEE)
    if site is None:
        lat, lon = meta["coords"]
        end_year = datetime.utcnow().year - 1
        power_df = _get_power_df(lat, lon, end_year - 10, end_year)
        computed = _compute_from_power(power_df, lat)
        site     = _build_site(meta, computed)
        if _CORE_OK:
            empty_chirps = pd.DataFrame(columns=["date", "precip"])
            metrics_fb   = _metrics_for_report(computed, site_id)
            charts["ndvi"]        = chart_ndvi_json(_ndvi_df(site_id))
            charts["spi"]         = chart_spi_json(empty_chirps, power_df)
            charts["temperature"] = chart_temperature_profile_json(power_df)
            charts["hydro_solar"] = chart_hydro_solar_profile_json(power_df)
            charts["radar"]       = chart_stress_radar_json(metrics_fb, meta["core_name"])
            charts["precip"]      = chart_annual_precip_json(power_df)

    return render_template("dashboard.html",
                           site=site, site_id=site_id,
                           all_sites=ALL_SITES, today=today,
                           charts=charts)


def _get_report_data(site_id):
    """Return (metrics, power_df, site_data) using GEE if available."""
    meta      = _SITES_META[site_id]
    lat, lon  = meta["coords"]
    site_data = _site_data_for_report(meta)

    gee = _get_site_metrics(meta["core_name"]) if _CORE_OK else None
    if gee:
        return gee["metrics"], gee["power_df"], site_data

    power_df = _get_power_df(lat, lon, datetime.utcnow().year - 11,
                             datetime.utcnow().year - 1)
    computed = _compute_from_power(power_df, lat)
    return _metrics_for_report(computed, site_id), power_df, site_data


@app.route("/site/<site_id>/report")
def text_report(site_id="spain"):
    if site_id not in _SITES_META:
        site_id = "spain"
    if not _CORE_OK:
        return Response(f"Report unavailable: {_core_err_msg}",
                        mimetype="text/plain; charset=utf-8")
    meta = _SITES_META[site_id]
    lat, lon = meta["coords"]
    metrics, power_df, site_data = _get_report_data(site_id)
    try:
        ctx = generate_report_context(meta["core_name"], lat, lon, site_data, metrics, power_df)
    except Exception as exc:
        return Response(f"Report generation failed: {exc}", mimetype="text/plain; charset=utf-8")
    return render_template("report.html", ctx=ctx, site_id=site_id)


@app.route("/site/<site_id>/pdf")
def pdf_report(site_id="spain"):
    if site_id not in _SITES_META:
        site_id = "spain"
    if not _CORE_OK:
        return Response(f"PDF unavailable: {_core_err_msg}", status=500,
                        mimetype="text/plain")
    meta = _SITES_META[site_id]
    lat, lon = meta["coords"]
    metrics, power_df, site_data = _get_report_data(site_id)
    try:
        pdf_bytes = generate_pdf_report(meta["core_name"], lat, lon,
                                        site_data, metrics, power_df)
        if not pdf_bytes:
            return Response("PDF unavailable: reportlab not installed",
                            status=500, mimetype="text/plain")
        fname = f"NBS4MED_{site_id}_diagnostic.pdf"
        return Response(pdf_bytes, mimetype="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename={fname}"})
    except Exception as exc:
        return Response(f"PDF generation failed: {exc}", status=500,
                        mimetype="text/plain")


@app.route("/chat", methods=["POST"])
def chat_proxy():
    N8N_WEBHOOK = "https://esraaalhaj.app.n8n.cloud/webhook/nbs-chatbot"
    try:
        body = request.get_json(force=True) or {}
        resp = _requests.post(N8N_WEBHOOK, json=body, timeout=20)
        return jsonify(resp.json()), resp.status_code
    except _requests.exceptions.Timeout:
        return jsonify({"reply": "The assistant took too long to respond. Please try again."}), 504
    except Exception as e:
        return jsonify({"reply": f"Assistant error: {e}"}), 500



@app.route("/debug")
def debug_info():
    import json as _json, os as _os, traceback as _tb

    lines = []
    lines.append(f"<b>_CORE_OK:</b> {_CORE_OK}")
    if not _CORE_OK:
        lines.append(f"<b>core import error:</b> {_core_err_msg}")

    sa_json = _os.environ.get("GEE_SERVICE_ACCOUNT_JSON", "")
    lines.append(f"<b>GEE_SERVICE_ACCOUNT_JSON present:</b> {bool(sa_json)} (len={len(sa_json)})")
    if sa_json:
        try:
            info = _json.loads(sa_json)
            lines.append(f"<b>service_account email:</b> {info.get('client_email', 'N/A')}")
            lines.append(f"<b>project_id:</b> {info.get('project_id', 'N/A')}")
        except Exception as e:
            lines.append(f"<b>JSON parse error:</b> {e}")

    if _CORE_OK:
        try:
            from core import init_ee
            ok, msg = init_ee()
            lines.append(f"<b>init_ee():</b> ok={ok}, msg={msg}")
        except Exception as e:
            lines.append(f"<b>init_ee() exception:</b> {e}")

        lines.append("<hr><b>Testing compute_site_metrics (Spain)...</b>")
        try:
            from core import compute_site_metrics
            result = compute_site_metrics("🇪🇸  Spain — Cerdanyola del Vallès")
            m = result.get("metrics", {})
            lines.append(f"<b>ee_ok:</b> {result.get('ee_ok')} — {result.get('ee_msg')}")
            lines.append(f"<b>LST:</b> {m.get('LST')}")
            lines.append(f"<b>NDVI:</b> {m.get('NDVI')}")
            lines.append(f"<b>VCI:</b> {m.get('VCI')}")
            lines.append(f"<b>HeatIndex:</b> {m.get('HeatIndex')}")
            lines.append(f"<b>Aridity:</b> {m.get('Aridity')}")
            lines.append(f"<b>power_df rows:</b> {len(result.get('power_df', []))}")
            lines.append(f"<b>ndvi_df rows:</b> {len(result.get('ndvi_df', []))}")
        except Exception as e:
            lines.append(f"<b>compute_site_metrics FAILED:</b> {e}")
            lines.append(f"<pre>{_tb.format_exc()}</pre>")

    html = "<html><body style='font-family:monospace;padding:20px;line-height:1.8;'>"
    html += "<h2>NBS4MED Debug</h2>"
    html += "<br>".join(lines)
    html += "</body></html>"
    return html


if __name__ == "__main__":
    app.run(debug=True, port=5000)