"""Analysis tools exposed to Qwen via function calling.

Rules: the LLM never does math. Each tool takes simple arguments, computes with pandas/scipy
and returns a small JSON-serializable dict. The docstring's first line is what Qwen reads
to decide WHEN to call the tool, so keep it precise.
"""
import inspect
import math
import types
import typing
from typing import Literal

import numpy as np
import pandas as pd
import requests
from scipy import stats

import data
from provinces_meta import COORDS

METRICS = {"max_temp": ("Max_Sicaklik", "°C"), "min_temp": ("Min_Sicaklik", "°C"),
           "mean_temp": ("Ort_Sicaklik", "°C"), "precip": ("Yagis_Miktari", "mm")}
Metric = Literal["max_temp", "min_temp", "mean_temp", "precip"]
Season = Literal["spring", "summer", "autumn", "winter"]
Order = Literal["highest", "lowest"]
REGISTRY = {}


def tool(fn):
    REGISTRY[fn.__name__] = fn
    return fn


# ------------------------------------------------------------------ helpers
def _clean(o):
    """Make results JSON-safe (numpy -> python, NaN -> None, Timestamp -> str)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if math.isnan(o) else round(float(o), 3)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


def _pv(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _filter(df, year=None, month=None):
    if year:
        df = df[df.Yil == int(year)]
    if month:
        df = df[df.Ay == int(month)]
    return df


SEASON_MONTHS = {"spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11), "winter": (12, 1, 2)}


def _season(df, year, season):
    months = SEASON_MONTHS[season]
    if season == "winter" and year:
        return df[((df.Yil == int(year)) & df.Ay.isin((1, 2))) | ((df.Yil == int(year) - 1) & (df.Ay == 12))]
    if year:
        df = df[df.Yil == int(year)]
    return df[df.Ay.isin(months)]


def _range(df):
    return [df.Tarih.min(), df.Tarih.max()]


def _meta(df):
    return {"days_covered": len(df), "data_range": _range(df)}


def _complete_years(df):
    c = df.groupby("Yil").size()
    return list(c[c >= 360].index)       # partial years (2022, 2026 ...) must not bias yearly stats


def _yearly(df, metric):
    col = METRICS[metric][0]
    g = df[df.Yil.isin(_complete_years(df))].groupby("Yil")[col]
    return g.sum() if metric == "precip" else g.mean()


def _agg(df, metric):
    col = METRICS[metric][0]
    return df[col].sum() if metric == "precip" else df[col].mean()


def _chart(kind, title, x, series, unit=""):
    return {"type": kind, "title": title, "unit": unit, "x": [str(i) for i in x],
            "series": [{"name": n, "values": list(v)} for n, v in series]}


def _no_data(df):
    return {"error": "No data for that period", "available_range": _range(df)}


# ------------------------------------------------------------------ descriptive tools
@tool
def get_summary(province: str, year: int | None = None, month: int | None = None, season: Season | None = None) -> dict:
    """Summary statistics for ONE province (mean max/min/mean temperature, total precipitation, hottest and coldest day), optionally for one year, one month, or one season (spring/summer/autumn/winter). Use for 'how was 2025 in Ankara?' or 'how was summer 2024?'."""
    p = data.resolve_province(province)
    df = data.frame(p)
    sub = _season(df, year, season) if season else _filter(df, year, month)
    if sub.empty:
        return _no_data(df)
    per_year = bool(month and not year)       # month across years -> average yearly total
    hot, cold = sub.loc[sub.Max_Sicaklik.idxmax()], sub.loc[sub.Min_Sicaklik.idxmin()]
    return {"province": p, "source": data.source_name(p), "period": {"year": year, "month": month, "season": season}, **_meta(sub),
            "mean_max_temp": sub.Max_Sicaklik.mean(), "mean_min_temp": sub.Min_Sicaklik.mean(),
            "mean_temp": sub.Ort_Sicaklik.mean(),
            "precip_total_mm": sub.Yagis_Miktari.sum() / (sub.Yil.nunique() if per_year else 1),
            "precip_total_is_average_per_year": per_year,
            "rainy_days_pct": 100 * (sub.Yagis_Miktari >= 1).mean(),
            "hottest_day": [hot.Tarih, hot.Max_Sicaklik], "coldest_day": [cold.Tarih, cold.Min_Sicaklik]}


@tool
def find_extremes(province: str, metric: Metric, n: int = 5, order: Order = "highest") -> dict:
    """Record days for one province: hottest days, coldest days, wettest days (top n)."""
    p = data.resolve_province(province)
    df = data.frame(p)
    col, unit = METRICS[metric]
    s = df.nlargest(min(n, 15), col) if order == "highest" else df.nsmallest(min(n, 15), col)
    return {"province": p, "metric": metric, "unit": unit, "order": order, "source": data.source_name(p),
            "days": [[t, v] for t, v in zip(s.Tarih, s[col])], "data_range": _range(df)}


@tool
def count_threshold_days(province: str, metric: Metric, threshold: float, direction: Literal["above", "below"],
                         year: int | None = None) -> dict:
    """Count days where a metric is above/below a threshold (e.g. days >= 35 °C, frost days below 0 °C, days with >10 mm rain). Returns yearly counts (decade averages if the series is long)."""
    p = data.resolve_province(province)
    df = _filter(data.frame(p), year)
    if df.empty:
        return _no_data(data.frame(p))
    col, unit = METRICS[metric]
    hit = df[col] >= threshold if direction == "above" else df[col] < threshold
    g = pd.DataFrame({"days": df.groupby("Yil").size(), "count": df[hit].groupby("Yil").size()}).fillna(0)
    g["complete"] = g.days >= 360
    out = {"province": p, "metric": metric, "threshold": threshold, "unit": unit, "direction": direction,
           "total_days_matching": int(hit.sum()), "data_range": _range(df)}
    full = g[g.complete]
    if len(full) > 12:   # long series: keep the answer small for the LLM
        full = full.assign(decade=(full.index // 10) * 10)
        out["avg_days_per_year_by_decade"] = full.groupby("decade")["count"].mean().round(1).to_dict()
        lr = stats.linregress(full.index.values.astype(float), full["count"].values)
        out["trend_days_per_decade"] = lr.slope * 10
        out["trend_p_value"] = _pv(lr.pvalue)
        out["chart"] = _chart("bar", f"{p}: days {direction} {threshold} {unit} per year", full.index, [("days", full["count"])], "days")
    else:
        out["by_year"] = [{"year": int(y), "days_observed": int(r.days), "count": int(r["count"]), "complete_year": bool(r.complete)}
                          for y, r in g.iterrows()]
    return out


# ------------------------------------------------------------------ cross-province tools (81 provinces, 4 years)
@tool
def rank_provinces(metric: Metric, year: int | None = None, month: int | None = None, top_n: int = 5,
                   order: Order = "highest") -> dict:
    """Rank all 81 provinces by mean temperature (max/min/mean) or total precipitation over a period (year, month, or everything). Use for 'which provinces are hottest/wettest?'."""
    df = _filter(data.provinces(), year, month)
    if df.empty:
        return _no_data(data.provinces())
    col, unit = METRICS[metric]
    r = (df.groupby("Sehir")[col].sum() if metric == "precip" else df.groupby("Sehir")[col].mean())
    r = r.nlargest(min(top_n, 20)) if order == "highest" else r.nsmallest(min(top_n, 20))
    return {"metric": metric, "unit": unit, "aggregation": "total" if metric == "precip" else "mean",
            "period": {"year": year, "month": month}, **_meta(df),
            "ranking": [{"province": k, "value": v} for k, v in r.items()],
            "chart": _chart("bar", f"Provinces by {metric} ({order})", r.index, [(metric, r.values)], unit)}


@tool
def compare_provinces(province_a: str, province_b: str, year: int | None = None, month: int | None = None) -> dict:
    """Compare two provinces (mean temperatures, total precipitation, monthly profile). Both come from the same 4-year dataset so the comparison is fair."""
    a, b = data.resolve_province(province_a), data.resolve_province(province_b)
    d = _filter(data.provinces(), year, month)
    if d.empty:
        return _no_data(data.provinces())
    res = {}
    for name in (a, b):
        s = d[d.Sehir == name]
        res[name] = {"mean_max_temp": s.Max_Sicaklik.mean(), "mean_min_temp": s.Min_Sicaklik.mean(),
                     "mean_temp": s.Ort_Sicaklik.mean(), "precip_total_mm": s.Yagis_Miktari.sum(),
                     "rainy_days_pct": 100 * (s.Yagis_Miktari >= 1).mean()}
    prof = d[d.Sehir.isin([a, b])].groupby(["Ay", "Sehir"]).Ort_Sicaklik.mean().unstack()
    return {"period": {"year": year, "month": month}, **_meta(d[d.Sehir == a]), "results": res,
            "mean_temp_difference_a_minus_b": res[a]["mean_temp"] - res[b]["mean_temp"],
            "chart": _chart("line", f"Monthly mean temperature: {a} vs {b}", prof.index,
                            [(a, prof[a].values), (b, prof[b].values)], "°C")}


@tool
def similar_climates(province: str, top_n: int = 5) -> dict:
    """Find the provinces whose yearly climate profile (12 monthly mean temperatures + 12 monthly mean precipitation values) is closest to the given province."""
    p = data.resolve_province(province)
    prof = data.provinces().groupby(["Sehir", "Ay"]).agg(t=("Ort_Sicaklik", "mean"), r=("Yagis_Miktari", "mean")).unstack("Ay")
    z = (prof - prof.mean()) / prof.std()
    dist = np.sqrt(((z - z.loc[p]) ** 2).sum(axis=1)).drop(p).nsmallest(min(top_n, 15))
    return {"province": p, "most_similar": [{"province": k, "distance": v} for k, v in dist.items()],
            "note": "distance is on standardized monthly features; smaller = more similar"}


@tool
def anomaly_check(province: str, date: str) -> dict:
    """Is the temperature on a given date (YYYY-MM-DD) normal or unusual for that time of year? Uses the same calendar window (+-3 days) in all other years available."""
    p = data.resolve_province(province)
    df = data.frame(p)
    t = pd.Timestamp(date)
    row = df[df.Tarih == t]
    if row.empty:
        return _no_data(df)
    row = row.iloc[0]
    d = (df.Gun_Yili - t.dayofyear).abs()
    ref = df[(np.minimum(d, 366 - d) <= 3) & (df.Yil != t.year)]
    out = {"province": p, "date": t, "reference_years": int(ref.Yil.nunique()), "source": data.source_name(p)}
    if out["reference_years"] < 10:
        out["warning"] = "Few reference years; percentiles are rough."
    for col, key in [("Max_Sicaklik", "max_temp"), ("Min_Sicaklik", "min_temp")]:
        out[key] = {"value": row[col], "historical_mean": ref[col].mean(),
                    "percentile": 100 * (ref[col] < row[col]).mean()}
    return out


# ------------------------------------------------------------------ long-term tools (Kutahya, 1940-2026)
@tool
def long_term_trend(metric: Metric) -> dict:
    """Long-term climate trend for KUTAHYA only (1940-2025, yearly values): slope per decade, Mann-Kendall significance, first-20-years vs last-20-years. Use for 'is it getting warmer/drier?'."""
    y = _yearly(data.kutahya_long(), metric)
    x, v = y.index.values.astype(float), y.values
    lr, sen, (_, p) = stats.linregress(x, v), stats.theilslopes(v, x), stats.kendalltau(x, v)
    return {"province": "Kutahya", "metric": metric, "unit": METRICS[metric][1], "years": [int(x[0]), int(x[-1])],
            "aggregation": "yearly total" if metric == "precip" else "yearly mean",
            "slope_per_decade": lr.slope * 10, "sen_slope_per_decade": sen[0] * 10, "mann_kendall_p": _pv(p),
            "significant_at_5pct": bool(p < 0.05), "first_20_years_mean": v[:20].mean(), "last_20_years_mean": v[-20:].mean(),
            "chart": _chart("line", f"Kutahya yearly {metric}", y.index,
                            [("observed", v), ("trend", lr.intercept + lr.slope * x)], METRICS[metric][1])}


@tool
def frost_calendar() -> dict:
    """Last spring frost and first autumn frost (min temperature below 0 °C) for KUTAHYA: average dates, frost-free period, planting risk and trend. Use for planting / seedling / agriculture questions."""
    df = data.kutahya_long()
    years = _complete_years(df)
    spring, autumn = [], []
    for y in years:
        f = df[(df.Yil == y) & (df.Min_Sicaklik < 0)]
        s_, a_ = f[f.Ay <= 6], f[f.Ay >= 8]
        spring.append(s_.Gun_Yili.max() if len(s_) else np.nan)
        autumn.append(a_.Gun_Yili.min() if len(a_) else np.nan)
    spring, autumn = np.array(spring), np.array(autumn)
    months = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

    def to_ts(y, doy):
        return pd.Timestamp(year=int(y), month=1, day=1) + pd.Timedelta(days=int(round(doy)) - 1)

    def tr(doy):                                   # day-of-year -> "14 Nisan" (non-leap reference year)
        t = to_ts(2001, doy)
        return f"{t.day} {months[t.month - 1]}"

    def slope(v):
        ok = ~np.isnan(v)
        return stats.linregress(np.array(years)[ok], np.array(v)[ok]).slope * 10

    def when(x):
        return f"{abs(x):.1f} gün {'erken' if x < 0 else 'geç'}"
    md = [(to_ts(y, d).month * 100 + to_ts(y, d).day) for y, d in zip(years, spring) if not np.isnan(d)]
    late_apr, late_may = 100 * np.mean([m > 430 for m in md]), 100 * np.mean([m > 515 for m in md])
    worst_i = int(np.nanargmax(spring))
    free = int(round(np.nanmean(autumn) - np.nanmean(spring)))
    es, ea = slope(spring), slope(autumn)
    return {"province": "Kutahya", "years": [years[0], years[-1]], "years_used": len(years),
            "avg_last_spring_frost": to_ts(2001, np.nanmean(spring)).strftime("%m-%d"),
            "avg_first_autumn_frost": to_ts(2001, np.nanmean(autumn)).strftime("%m-%d"),
            "frost_free_days": free,
            "pct_years_with_frost_after_apr_30": late_apr, "pct_years_with_frost_after_may_15": late_may,
            "latest_spring_frost_on_record": {"year": years[worst_i], "date": to_ts(2001, spring[worst_i]).strftime("%m-%d")},
            "last_spring_frost_trend_days_per_decade": es, "first_autumn_frost_trend_days_per_decade": ea,
            "plain_summary_tr": (
                f"Kütahya'da ilkbaharın SON donu ortalama {tr(np.nanmean(spring))}, sonbaharın İLK donu ortalama {tr(np.nanmean(autumn))} "
                f"civarındadır; arada yaklaşık {free} gün donsuz dönem vardır. Dona duyarlı fideler ilkbaharın son donundan SONRA dikilir. "
                f"Yılların %{late_apr:.0f}'inde 30 Nisan'dan sonra hâlâ don görülmüştür; 15 Mayıs'tan sonra "
                + ("hiç don görülmemiştir" if late_may == 0 else f"yılların %{late_may:.0f}'inde don görülmüştür")
                + f"; kayıtlardaki en geç ilkbahar donu {years[worst_i]} yılında {tr(spring[worst_i])} tarihindedir. Eğilim: ilkbahar donu on yılda {when(es)}, "
                f"sonbahar ilk donu on yılda {when(ea)} gelmektedir ({years[0]}-{years[-1]}).")}


# ------------------------------------------------------------------ forecast tools
@tool
def seasonal_outlook(province: str, month: int) -> dict:
    """What to typically EXPECT in a given month (1-12): historical distribution of that month's mean temperature and total precipitation (mean, 10th/50th/90th percentile). A climatological expectation, not a weather forecast."""
    p = data.resolve_province(province)
    df = data.frame(p)
    m = df.groupby(["Yil", "Ay"]).agg(days=("Tarih", "size"), t=("Ort_Sicaklik", "mean"), r=("Yagis_Miktari", "sum")).reset_index()
    m = m[(m.Ay == month) & (m.days >= pd.Period(f"2001-{month:02d}").days_in_month)]
    if m.empty:
        return _no_data(df)
    q = lambda s: {"mean": s.mean(), "p10": s.quantile(.1), "p50": s.quantile(.5), "p90": s.quantile(.9)}
    out = {"province": p, "month": month, "years_used": int(len(m)), "mean_temp_c": q(m.t), "precip_total_mm": q(m.r),
           "note": "climatological expectation, not a forecast", "source": data.source_name(p)}
    if len(m) >= 30:
        out["last_10_years_mean_temp_c"] = m.t.tail(10).mean()
        out["warming_vs_all_years_c"] = m.t.tail(10).mean() - m.t.mean()
    if len(m) < 10:
        out["warning"] = f"Only {len(m)} years of data for this province; the distribution is very rough."
    return out


@tool
def project_trend(metric: Metric, target_year: int) -> dict:
    """Project KUTAHYA's yearly mean temperature or yearly precipitation to a future year with a linear trend and a 95% prediction interval. Statistical extrapolation, NOT a weather or climate-model forecast."""
    y = _yearly(data.kutahya_long(), metric)
    x, v = y.index.values.astype(float), y.values
    if target_year <= x[-1]:
        return {"error": f"target_year must be after {int(x[-1])}"}
    b, a = np.polyfit(x, v, 1)
    n = len(x)
    sd = np.sqrt(((v - (a + b * x)) ** 2).sum() / (n - 2))
    se = sd * np.sqrt(1 + 1 / n + (target_year - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    pred, pad = a + b * target_year, stats.t.ppf(.975, n - 2) * se
    future = list(range(int(x[-1]) + 1, int(target_year) + 1))
    xs = list(y.index) + future
    return {"province": "Kutahya", "metric": metric, "unit": METRICS[metric][1], "target_year": target_year,
            "prediction": pred, "interval_95": [pred - pad, pred + pad], "slope_per_decade": b * 10,
            "note": "linear extrapolation of the 1940-2025 trend; real climate may change non-linearly",
            "chart": _chart("line", f"Kutahya {metric}: observed and projection", xs,
                            [("observed", list(v) + [None] * len(future)), ("trend", [a + b * i for i in xs])], METRICS[metric][1])}


@tool
def live_forecast(province: str, days: int = 7, target_date: str | None = None) -> dict:
    """REAL weather forecast from the Open-Meteo forecast API (needs internet). For ONE specific day (e.g. tomorrow) pass target_date as YYYY-MM-DD; otherwise it returns the next `days` days starting TODAY (1-16)."""
    import datetime as dt
    p = data.resolve_province(province)
    lat, lon = COORDS[p]
    today = dt.date.today()
    n = 16 if target_date else max(1, min(int(days), 16))
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&forecast_days={n}"
           "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Europe%2FIstanbul")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        d = r.json()["daily"]
    except Exception as e:  # noqa: BLE001
        return {"error": f"Forecast service unreachable ({e.__class__.__name__}). Historical tools still work."}
    rows = [{"date": t, "weekday": dt.date.fromisoformat(t).strftime("%A"), "max_temp": a, "min_temp": b, "precip_mm": c}
            for t, a, b, c in zip(d["time"], d["temperature_2m_max"], d["temperature_2m_min"], d["precipitation_sum"])]
    if not target_date:
        rows = rows[:n]
    else:
        rows = [r for r in rows if r["date"] == target_date]
        if not rows:
            return {"error": f"{target_date} is outside the 16-day forecast window", "forecast_window": [d["time"][0], d["time"][-1]]}
    return {"province": p, "source": "open-meteo forecast", "today": today.isoformat(), "days": rows,
            "chart": _chart("line", f"{p}: forecast", [r["date"] for r in rows],
                            [("max", [r["max_temp"] for r in rows]), ("min", [r["min_temp"] for r in rows])], "°C")}


# ------------------------------------------------------------------ schema + dispatcher (used by llm.py)
def _json_type(t):
    origin = typing.get_origin(t)
    if origin in (typing.Union, types.UnionType):
        return _json_type([a for a in typing.get_args(t) if a is not type(None)][0])
    if origin is Literal:
        return {"type": "string", "enum": list(typing.get_args(t))}
    return {"type": {int: "integer", float: "number", str: "string"}[t]}


def build_schema():
    """Turn every @tool function into the JSON schema Qwen/Ollama expects (no hand-written schemas)."""
    out = []
    for name, fn in REGISTRY.items():
        hints, props, required = typing.get_type_hints(fn), {}, []
        for pn, p in inspect.signature(fn).parameters.items():
            props[pn] = _json_type(hints[pn])
            if p.default is inspect.Parameter.empty:
                required.append(pn)
        out.append({"type": "function", "function": {"name": name, "description": inspect.getdoc(fn),
                                                     "parameters": {"type": "object", "properties": props, "required": required}}})
    return out


def _coerce(value, hint):
    """Small models sometimes send '2025' as text or 35.0 for an int; fix that."""
    base = _json_type(hint).get("type")
    try:
        if base == "integer" and value is not None:
            return int(float(value))
        if base == "number" and value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass
    return value


def call_tool(name, args):
    if name not in REGISTRY:
        return {"error": f"Unknown tool: {name}", "available": list(REGISTRY)}
    fn = REGISTRY[name]
    hints = typing.get_type_hints(fn)
    try:
        kwargs = {k: _coerce(v, hints[k]) for k, v in (args or {}).items() if k in hints and v is not None}
        return _clean(fn(**kwargs))
    except data.ProvinceNotFound as e:
        return {"error": str(e), "did_you_mean": e.suggestions}
    except TypeError as e:
        return {"error": f"Bad arguments: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{e.__class__.__name__}: {e}"}