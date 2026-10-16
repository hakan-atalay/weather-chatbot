import difflib
import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
LONG_FILE = "kutahya_1940_2026.csv"
PROVINCES_FILE = "turkey_81_provinces_2023_2026.csv"
_cache = {}


class ProvinceNotFound(ValueError):
    def __init__(self, name, suggestions):
        super().__init__(f"Unknown province: {name}")
        self.suggestions = suggestions


def _prepare(df):
    df["Ort_Sicaklik"] = (df.Max_Sicaklik + df.Min_Sicaklik) / 2
    df["Yil"], df["Ay"], df["Gun_Yili"] = df.Tarih.dt.year, df.Tarih.dt.month, df.Tarih.dt.dayofyear
    return df


def kutahya_long():
    if "long" not in _cache:
        _cache["long"] = _prepare(pd.read_csv(DATA_DIR / LONG_FILE, parse_dates=["Tarih"]))
    return _cache["long"]


def provinces():
    if "prov" not in _cache:
        _cache["prov"] = _prepare(pd.read_csv(DATA_DIR / PROVINCES_FILE, parse_dates=["Tarih"]))
    return _cache["prov"]


def fold(text):
    t = unicodedata.normalize("NFKD", str(text).replace("İ", "I").replace("ı", "i"))
    return "".join(c for c in t if not unicodedata.combining(c)).lower().strip()


def province_names():
    return sorted(provinces().Sehir.unique())


def resolve_province(name):
    names = province_names()
    folded = {fold(n): n for n in names}
    f = fold(name)
    if f in folded:
        return folded[f]
    contains = [n for k, n in folded.items() if f and (f in k or k in f)]
    if len(contains) == 1:
        return contains[0]
    close = difflib.get_close_matches(f, list(folded), n=3, cutoff=0.6)
    raise ProvinceNotFound(name, contains or [folded[c] for c in close])


def frame(province, prefer_long=True):
    if province == "Kutahya" and prefer_long:
        df = kutahya_long().copy()
        df["Sehir"] = "Kutahya"
        return df
    d = provinces()
    return d[d.Sehir == province]


def source_name(province, prefer_long=True):
    return "kutahya_1940_2026" if province == "Kutahya" and prefer_long else "provinces_2022-10-08_to_2026-10-07"
