"""Fetch DEFRA AURN hourly particulate data and reduce it to a daily panel.

Raw per-site-year CSVs are large and are not committed; this script writes the
compact daily panel (data/aurn_daily_pm.csv) that the rest of the pipeline reads.
Re-running is cheap: a site-year already on disk is not downloaded again.
"""
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyreadr
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw" / "aurn"
OUT = ROOT / "data" / "aurn_daily_pm.csv"
YEARS = range(2021, 2026)
BASE = "https://uk-air.defra.gov.uk/datastore/data_files/site_data/{site}_{year}.csv"


def pm10_sites():
    """AURN sites measuring PM10 across the whole study window, with coordinates."""
    meta = pyreadr.read_r(str(RAW / "AURN_metadata.RData"))["AURN_metadata"]
    meta["start_date"] = pd.to_datetime(meta.start_date, errors="coerce")
    meta["end_date"] = pd.to_datetime(meta.end_date, errors="coerce")
    act = meta[
        (meta.parameter == "PM10")
        & (meta.start_date <= "2021-01-01")
        & ((meta.end_date >= "2025-12-01") | (meta.end_date.isna()))
    ]
    cols = ["site_id", "site_name", "latitude", "longitude", "location_type", "zone"]
    return act[cols].drop_duplicates("site_id").reset_index(drop=True)


def fetch(site, year):
    dest = RAW / f"{site}_{year}.csv"
    if dest.exists() and dest.stat().st_size > 2000:
        return dest
    url = BASE.format(site=site, year=year)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read()
            if len(body) < 2000:
                return None
            dest.write_bytes(body)
            return dest
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 * (attempt + 1))
    return None


def parse(path):
    """One site-year CSV -> daily mean PM10/PM2.5. Row 4 holds the real header."""
    txt = path.read_text(encoding="latin-1", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(txt), skiprows=4, low_memory=False)
    except Exception:
        return None
    if "Date" not in df.columns:
        return None
    cols = {}
    for c in df.columns:
        if c.startswith("PM<sub>10</sub>"):
            cols[c] = "pm10"
        elif c.startswith("PM<sub>2.5</sub>"):
            cols[c] = "pm25"
    if not cols:
        return None
    keep = df[["Date"] + list(cols)].rename(columns=cols)
    keep["date"] = pd.to_datetime(keep.Date, format="%d-%m-%Y", errors="coerce")
    for c in set(cols.values()):
        keep[c] = pd.to_numeric(keep[c], errors="coerce")
    keep = keep.dropna(subset=["date"])
    agg = {c: "mean" for c in set(cols.values())}
    out = keep.groupby("date").agg(agg).reset_index()
    out["pm10_hours"] = keep.groupby("date")["pm10"].count().values if "pm10" in keep else 0
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    sites = pm10_sites()
    sites.to_csv(ROOT / "data" / "aurn_sites.csv", index=False)
    jobs = [(s, y) for s in sites.site_id for y in YEARS]
    print(f"{len(sites)} PM10 sites x {len(list(YEARS))} years = {len(jobs)} site-years")

    with ThreadPoolExecutor(max_workers=8) as ex:
        paths = list(ex.map(lambda j: fetch(*j), jobs))
    got = sum(p is not None for p in paths)
    print(f"downloaded/present: {got}/{len(jobs)}")

    frames = []
    for (site, year), p in zip(jobs, paths):
        if p is None:
            continue
        d = parse(p)
        if d is None or d.empty:
            continue
        d["site_id"] = site
        frames.append(d)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["site_id", "date"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT, index=False)
    print(f"daily panel: {panel.shape[0]} site-days, {panel.site_id.nunique()} sites -> {OUT}")
    print(panel.head().to_string())


if __name__ == "__main__":
    sys.exit(main())
