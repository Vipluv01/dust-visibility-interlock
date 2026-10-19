"""Part 1 - relational ingestion, quality repair, and the particulate join.

Reads the three STATS19 tables (2021-2025) and the AURN daily particulate panel,
links them, repairs validity errors, and writes two analysis files:

  data/collision_panel.csv  one row per collision involving a goods vehicle
  data/daily_panel.csv      one row per calendar day (for the Part 2/3 models)

Sentinel codes are converted to NaN *before* the completeness table is computed,
and the missingness mask is recorded *before* median imputation, because the
Part 5 governor must know that a reading was absent rather than see the value it
was imputed to.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw" / "stats19"
DATA = ROOT / "data"

# Authoritative labels, from DfT's 2024 "Road Safety Open Dataset Data Guide".
WEATHER = {1: "Fine no high winds", 2: "Raining no high winds", 3: "Snowing no high winds",
           4: "Fine + high winds", 5: "Raining + high winds", 6: "Snowing + high winds",
           7: "Fog or mist", 8: "Other", 9: "Unknown"}
SEVERITY = {1: "Fatal", 2: "Serious", 3: "Slight"}
SURFACE = {1: "Dry", 2: "Wet or damp", 3: "Snow", 4: "Frost or ice",
           5: "Flood over 3cm deep", 6: "Oil or diesel", 7: "Mud"}
LIGHT = {1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit",
         6: "Darkness - no lighting", 7: "Darkness - lighting unknown"}
ROAD_CLASS = {1: "Motorway", 2: "A(M)", 3: "A", 4: "B", 5: "C", 6: "Unclassified"}
GOODS_VEHICLES = {19: "Van / Goods <=3.5t", 20: "Goods 3.5-7.5t",
                  21: "Goods >=7.5t", 98: "Goods - unknown weight"}
# vehicle_manoeuvre 4 is "Slowing or stopping"; skidding 1-5 are loss-of-control
# outcomes. Together these are this study's observable proxy for the brief's
# "sudden deceleration spike" - STATS19 carries no vehicle telemetry.
DECEL_MANOEUVRE = 4
SKID_CODES = {1, 2, 3, 4, 5}

# Fields where 9 / 99 means "unknown (self reported)" rather than a real level.
UNKNOWN_NINE = {"road_surface_conditions", "special_conditions_at_site",
                "carriageway_hazards", "skidding_and_overturning", "vehicle_manoeuvre"}
SPEED_LIMITS_VALID = {20, 30, 40, 50, 60, 70}

COLLISION_COLS = ["collision_index", "collision_year", "date", "time", "day_of_week",
                  "longitude", "latitude", "police_force", "collision_severity",
                  "number_of_vehicles", "number_of_casualties", "speed_limit",
                  "first_road_class", "road_type", "junction_detail", "light_conditions",
                  "weather_conditions", "road_surface_conditions",
                  "special_conditions_at_site", "carriageway_hazards",
                  "urban_or_rural_area", "trunk_road_flag", "local_authority_ons_district"]
VEHICLE_COLS = ["collision_index", "vehicle_reference", "vehicle_type",
                "vehicle_manoeuvre", "skidding_and_overturning", "towing_and_articulation",
                "age_of_driver", "age_of_vehicle", "journey_purpose_of_driver"]


def sentinel_to_nan(df):
    """-1 means 'data missing or out of range' in every STATS19 field."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind in "ifu":
            out.loc[out[c] == -1, c] = np.nan
        else:
            out.loc[out[c].astype(str).isin(["-1", "-1.0"]), c] = np.nan
    for c in UNKNOWN_NINE:
        if c in out.columns:
            out.loc[out[c].isin([9, 99]), c] = np.nan
    return out


def completeness(df, label):
    pct = (df.isna().mean() * 100).round(4).sort_values(ascending=False)
    tab = pd.DataFrame({"column": pct.index, "missing_pct": pct.values})
    tab["table"] = label
    return tab


def main():
    DATA.mkdir(exist_ok=True)
    print("=== PART 1: RELATIONAL INGESTION & INFORMATION QUALITY PIPELINE ===\n")

    coll = pd.read_csv(RAW / "collision-last5.csv", usecols=COLLISION_COLS, low_memory=False)
    veh = pd.read_csv(RAW / "vehicle-last5.csv", usecols=VEHICLE_COLS, low_memory=False)
    cas = pd.read_csv(RAW / "casualty-last5.csv",
                      usecols=["collision_index", "casualty_severity", "casualty_class"],
                      low_memory=False)
    print(f"collision table : {coll.shape[0]:,} rows x {coll.shape[1]} cols")
    print(f"vehicle table   : {veh.shape[0]:,} rows x {veh.shape[1]} cols")
    print(f"casualty table  : {cas.shape[0]:,} rows x {cas.shape[1]} cols")
    print(f"join key        : collision_index "
          f"(collision 1:N vehicle, collision 1:N casualty)\n")

    coll = sentinel_to_nan(coll)
    veh = sentinel_to_nan(veh)
    cas = sentinel_to_nan(cas)

    # --- completeness, computed after sentinel conversion and before repair ---
    comp = pd.concat([completeness(coll, "collision"),
                      completeness(veh, "vehicle"),
                      completeness(cas, "casualty")], ignore_index=True)
    comp.to_csv(DATA / "completeness.csv", index=False)
    print("--- column completeness, worst 12 (percentage of NaN entries) ---")
    print(comp.nlargest(12, "missing_pct").to_string(index=False), "\n")

    # --- validity errors and their repair ---
    repairs = {}
    bad_speed = (~coll.speed_limit.isin(SPEED_LIMITS_VALID)) & coll.speed_limit.notna()
    out_of_range = ((coll.speed_limit < 0) | (coll.speed_limit > 85)) & coll.speed_limit.notna()
    repairs["speed_limit_not_in_statutory_set"] = int(bad_speed.sum())
    repairs["speed_limit_outside_0_85"] = int(out_of_range.sum())
    coll["speed_limit_was_missing"] = coll.speed_limit.isna() | bad_speed
    coll.loc[bad_speed, "speed_limit"] = np.nan
    speed_median = float(coll.speed_limit.median())
    coll["speed_limit"] = coll.speed_limit.fillna(speed_median)
    repairs["speed_limit_median_used"] = speed_median

    veh["driver_age_was_missing"] = veh.age_of_driver.isna()
    bad_age = ((veh.age_of_driver < 16) | (veh.age_of_driver > 110)) & veh.age_of_driver.notna()
    repairs["age_of_driver_outside_16_110"] = int(bad_age.sum())
    veh.loc[bad_age, "age_of_driver"] = np.nan
    age_median = float(veh.age_of_driver.median())
    veh["age_of_driver"] = veh.age_of_driver.fillna(age_median)
    repairs["age_of_driver_median_used"] = age_median

    coll["datetime"] = pd.to_datetime(coll.date + " " + coll.time.fillna("00:00"),
                                      format="%d/%m/%Y %H:%M", errors="coerce")
    coll["date"] = pd.to_datetime(coll.date, format="%d/%m/%Y", errors="coerce")
    repairs["unparseable_datetime"] = int(coll.datetime.isna().sum())
    repairs["missing_coordinates"] = int(coll.longitude.isna().sum())

    print("--- validity rules applied, and the repair used ---")
    for k, v in repairs.items():
        print(f"  {k:42s} {v}")
    print(f"\n  rule: speed_limit must be in {sorted(SPEED_LIMITS_VALID)}; values <0 or >85 "
          f"are out of range.\n  Both are set to NaN and replaced with the column median "
          f"({speed_median:.4f} mph).\n  rule: age_of_driver must be in [16, 110]; replaced "
          f"with the column median ({age_median:.4f} yr).\n")

    # --- goods-vehicle flag: Project 10 is about commercial fleets ---
    veh["is_goods"] = veh.vehicle_type.isin(GOODS_VEHICLES)
    veh["decel_event"] = (veh.vehicle_manoeuvre == DECEL_MANOEUVRE)
    veh["skid_event"] = veh.skidding_and_overturning.isin(SKID_CODES)

    per_coll = veh.groupby("collision_index").agg(
        n_goods=("is_goods", "sum"),
        any_goods=("is_goods", "any"),
        any_decel=("decel_event", "any"),
        any_skid=("skid_event", "any"),
        n_articulated=("towing_and_articulation",
                       lambda s: int((s.fillna(0) > 0).sum())),
        mean_driver_age=("age_of_driver", "mean"),
    ).reset_index()

    worst_cas = cas.groupby("collision_index").casualty_severity.min().rename("worst_casualty")
    df = coll.merge(per_coll, on="collision_index", how="left").merge(
        worst_cas, on="collision_index", how="left")
    print(f"linked frame    : {df.shape[0]:,} collisions x {df.shape[1]} columns")
    print(f"involving goods vehicles: {int(df.any_goods.fillna(False).sum()):,} "
          f"({100 * df.any_goods.fillna(False).mean():.4f}% of all collisions)\n")

    # --- nearest-AURN-site join, then the daily particulate reading ---
    sites = pd.read_csv(DATA / "aurn_sites.csv")
    pm = pd.read_csv(DATA / "aurn_daily_pm.csv", parse_dates=["date"])
    tree = BallTree(np.radians(sites[["latitude", "longitude"]].values), metric="haversine")
    has_xy = df.latitude.notna() & df.longitude.notna()
    dist = np.full(len(df), np.nan)
    idx = np.full(len(df), -1)
    d, i = tree.query(np.radians(df.loc[has_xy, ["latitude", "longitude"]].values), k=1)
    dist[has_xy.values] = d[:, 0] * 6371.0088  # earth radius, km
    idx[has_xy.values] = i[:, 0]
    df["aurn_site"] = np.where(idx >= 0, sites.site_id.reindex(idx).values, None)
    df["aurn_dist_km"] = dist
    print("--- spatial join to the particulate network ---")
    print(f"  AURN PM10 sites in window      : {len(sites)}")
    print(f"  median collision-to-site dist  : {np.nanmedian(dist):.4f} km")
    print(f"  90th percentile distance       : {np.nanpercentile(dist, 90):.4f} km\n")

    df = df.merge(pm.rename(columns={"site_id": "aurn_site"}),
                  on=["aurn_site", "date"], how="left")
    df["pm10_was_missing"] = df.pm10.isna()
    print(f"  collisions with a same-day PM10 reading: "
          f"{100 * df.pm10.notna().mean():.4f}%\n")

    df["weather_label"] = df.weather_conditions.map(WEATHER)
    df["severity_label"] = df.collision_severity.map(SEVERITY)
    df["surface_label"] = df.road_surface_conditions.map(SURFACE)
    df["light_label"] = df.light_conditions.map(LIGHT)
    df["road_class_label"] = df.first_road_class.map(ROAD_CLASS)
    df["is_fog"] = df.weather_conditions.eq(7)

    df.to_csv(DATA / "collision_panel.csv", index=False)
    print(f"wrote {DATA/'collision_panel.csv'}  ({df.shape[0]:,} rows)")

    with open(DATA / "part1_repairs.json", "w") as fh:
        json.dump(repairs, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
