"""Capstone Project 10 - Sudden environmental visibility loss via particulate plumes.

Runs Parts 2-5 end to end and prints every number the report and slides quote.
Part 1 is scripts/build_dataset.py, which must run first and writes
data/collision_panel.csv.

    .venv/bin/python scripts/build_dataset.py
    .venv/bin/python run_capstone.py | tee run_output.txt
"""
import json
import os
import numpy as np
import pandas as pd

import analysis
import validation

SEED = 42

HEADER = """
=============================================================================
 Sudden Environmental Visibility Loss via Particulate Dust Plumes
 A hybrid statistical, machine-learning and STAMP software-interlock study
 of goods-vehicle risk on the GB road network, 2021-2025

 Vipluv  -  22MF3IM22
 Centre of Excellence on Safety Engineering & Analytics (CoE-SEA), IIT Kharagpur
=============================================================================
"""


def main():
    os.makedirs(analysis.FIG, exist_ok=True)
    np.random.seed(SEED)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(HEADER)

    df = pd.read_csv("data/collision_panel.csv", parse_dates=["date", "datetime"],
                     low_memory=False)
    print(f"analysis frame loaded: {df.shape[0]:,} collisions x {df.shape[1]} columns")

    panel = analysis.daily_panel(df)
    panel.to_csv("data/daily_panel.csv", index=False)

    assoc = analysis.part2(df, panel)
    glm, pca_info = analysis.part3(df, panel)
    clf, te, fn, metrics = analysis.part4(df, seed=SEED)

    validation.print_control_structure()
    aud = validation.offline_audit(fn, te)
    scenarios, results = validation.closed_loop()
    checks = validation.pass_fail(results, scenarios, aud)
    cross = validation.plot_physics()

    print("\n\n=== SUMMARY OF HEADLINE NUMBERS ===\n")
    print(f"  goods-vehicle collisions, 2021-2025      : "
          f"{int(panel.collisions.sum()):,}")
    print(f"  of which killed or seriously injured     : {int(panel.ksi.sum()):,}")
    print(f"  chi-square, fog x loss of control        : "
          f"{assoc['fog_loc']['chi2']:.4f} (critical {analysis.CRITICAL_CHI2_DF1:.4f}), "
          f"OR = {assoc['fog_loc']['odds_ratio']:.4f}")
    print(f"  chi-square, fog x fatal outcome          : "
          f"{assoc['fog_fatal']['chi2']:.4f}, OR = {assoc['fog_fatal']['odds_ratio']:.4f}")
    print(f"  chi-square, PM10 exceedance x loss of ctl: "
          f"{assoc['pm10_loc']['chi2']:.4f}, OR = {assoc['pm10_loc']['odds_ratio']:.4f} "
          f"(opposite sign)")
    print(f"  chi-square, pooled composite             : "
          f"{assoc['pooled']['chi2']:.4f} (the two effects cancel)")
    print(f"  daily PM10 vs wet-surface correlation    : {assoc['r_wet']:.4f}")
    print(f"  PCA components for 95% of variance       : {pca_info['k95']} of "
          f"{pca_info['ncols']}")
    print(f"  classifier macro F1                      : {metrics['macro_f1']:.4f}")
    print(f"  classifier accuracy / baseline           : {metrics['accuracy']:.4f} / "
          f"{metrics['baseline']:.4f}")
    print(f"  false negatives (Safe, actually Severe)  : {metrics['n_fn']:,}")
    print(f"  governor restricts on false negatives    : "
          f"{int((aud.gov_mode == 'RESTRICT').sum()):,} of {len(aud):,}")
    print(f"  safe speed at the brief's 15 m trigger   : "
          f"{validation.safe_speed_mph(15):.4f} mph")
    print(f"  20 mph only becomes safe beyond          : {cross:.4f} m")
    print(f"  constraint checks passing                : "
          f"{sum(1 for _, ok in checks if ok)} of {len(checks)}")
    summary = {
        "window": [str(panel.date.min().date()), str(panel.date.max().date())],
        "days": int(len(panel)),
        "collisions_all": int(len(df)),
        "collisions_goods": int(panel.collisions.sum()),
        "ksi": int(panel.ksi.sum()),
        "ksi_pct": float(100 * panel.ksi.sum() / panel.collisions.sum()),
        "assoc": {k: {"chi2": v["chi2"], "chi2_yates": v["chi2_yates"],
                      "odds_ratio": v["odds_ratio"], "phi": v["phi"],
                      "n": v["n"], "observed": v["observed"],
                      "expected": v["expected"], "rows": v["rows"],
                      "cols": v["cols"]}
                  for k, v in assoc.items()
                  if k not in ("confound", "r_wet", "r_fog", "march_pct")},
        "confound": assoc["confound"].to_dict("records"),
        "r_wet": assoc["r_wet"], "r_fog": assoc["r_fog"],
        "march_pct": assoc["march_pct"],
        "critical_chi2": analysis.CRITICAL_CHI2_DF1,
        "pca": pca_info,
        "glm": {"terms": list(glm.params.index),
                "coef": [float(x) for x in glm.params.values],
                "pvalues": [float(x) for x in glm.pvalues.values],
                "exp_coef": [float(np.exp(x)) for x in glm.params.values],
                "pseudo_r2": float(glm.pseudo_rsquared(kind="cs")),
                "n_obs": int(glm.nobs)},
        "clf": metrics,
        "governor": {
            "fn_total": int(len(aud)),
            "fn_degraded": int(aud.visibility_degraded.sum()),
            "fn_degraded_restricted": int(
                (aud[aud.visibility_degraded].gov_mode == "RESTRICT").sum()),
            "fn_clear": int((~aud.visibility_degraded).sum()),
            "brief_fires": int((aud.brief_mode == "RESTRICT").sum()),
            "safe_speed_15m": float(validation.safe_speed_mph(15)),
            "clamp_safe_beyond_m": float(cross),
            "checks": [[d, bool(ok)] for d, ok in checks],
        },
        "closed_loop": [
            {"scenario": s.name, "controller": c,
             "front": results[(s.name, c)]["front_impact_mph"],
             "rear": results[(s.name, c)]["rear_impact_mph"]}
            for s in scenarios
            for c in ["unguarded", "brief 20 mph clamp", "governor",
                      "governor, SC-2 off", "governor, SC-1 off"]
        ],
    }
    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nwrote results/summary.json")
    print("\ndone.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
