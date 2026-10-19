"""Parts 2-4: exploration and association, feature/count modelling, and the AI audit."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from governor import PM10_LIMIT

FIG = "results/figures"
SEV_CLASS = {3: "Safe", 2: "Minor", 1: "Severe"}   # guideline's three class levels
CRITICAL_CHI2_DF1 = 3.8415                          # alpha = 0.05


# =============================================================================
# PART 2
# =============================================================================
def daily_panel(df):
    """One row per calendar day, restricted to collisions involving a goods vehicle."""
    g = df[df.any_goods.fillna(False)].copy()
    g["is_ksi"] = g.collision_severity.isin([1, 2])
    panel = g.groupby("date").agg(
        collisions=("collision_index", "size"),
        ksi=("is_ksi", "sum"),
        fatal=("collision_severity", lambda s: int((s == 1).sum())),
        fog_share=("is_fog", "mean"),
        decel_share=("any_decel", "mean"),
        skid_share=("any_skid", "mean"),
        pm10_mean=("pm10", "mean"),
        pm10_exceed_share=("pm10", lambda s: float((s >= PM10_LIMIT).mean())),
        dark_share=("light_conditions", lambda s: float(s.isin([4, 5, 6, 7]).mean())),
        wet_share=("road_surface_conditions", lambda s: float((s == 2).mean())),
        speed_dispersion=("speed_limit", "std"),
        mean_speed_limit=("speed_limit", "mean"),
    ).reset_index()
    panel["risk_index"] = (panel.ksi / panel.collisions * 100).round(6)
    panel["month"] = panel.date.dt.month
    panel["year"] = panel.date.dt.year
    panel["dow"] = panel.date.dt.dayofweek
    return panel


def plot_daily(panel):
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(panel.date, panel.ksi, lw=.5, color="#b9c2cc", label="daily KSI count")
    ax[0].plot(panel.date, panel.ksi.rolling(28, center=True).mean(), lw=1.8,
               color="#1f4e79", label="28-day rolling mean")
    ax[0].set_ylabel("Goods-vehicle KSI collisions")
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    ax[0].set_title("Daily goods-vehicle killed-or-seriously-injured collisions, "
                    "Great Britain 2021–2025", fontsize=10.5, loc="left")
    ax[1].plot(panel.date, panel.pm10_mean.rolling(28, center=True).mean(), lw=1.4,
               color="#9c3d1e", label="28-day mean PM10 at nearest site")
    ax[1].axhline(PM10_LIMIT, ls="--", lw=.9, color="#555",
                  label=f"PM10 limit value {PM10_LIMIT:.0f} µg/m³")
    ax[1].set_ylabel("PM10 (µg/m³)")
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", lw=.3, alpha=.4)
    fig.tight_layout()
    fig.savefig(f"{FIG}/daily_risk_index.png", dpi=300)
    fig.savefig(f"{FIG}/daily_risk_index.pdf")
    plt.close(fig)


def manual_chi2(a, b, c, d, label_rows, label_cols):
    """2x2 chi-square computed from the counts, not from a library call."""
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1, c2 = a + c, b + d
    exp = [[r1 * c1 / n, r1 * c2 / n], [r2 * c1 / n, r2 * c2 / n]]
    obs = [[a, b], [c, d]]
    chi2 = sum((obs[i][j] - exp[i][j]) ** 2 / exp[i][j] for i in range(2) for j in range(2))
    yates = sum((abs(obs[i][j] - exp[i][j]) - 0.5) ** 2 / exp[i][j]
                for i in range(2) for j in range(2))
    orat = (a * d) / (b * c) if b and c else float("nan")
    phi = ((a * d - b * c) / ((r1 * r2 * c1 * c2) ** 0.5)) if min(r1, r2, c1, c2) else float("nan")
    return {"observed": obs, "expected": exp, "n": n, "chi2": chi2, "chi2_yates": yates,
            "odds_ratio": orat, "phi": phi, "rows": label_rows, "cols": label_cols}


def show_chi2(res, title):
    print(f"\n--- {title} ---")
    rl, cl = res["rows"], res["cols"]
    w = max(len(x) for x in rl) + 2
    print(f"{'':{w}}{cl[0]:>22}{cl[1]:>22}{'row total':>12}")
    for i in range(2):
        o = res["observed"][i]
        print(f"{rl[i]:{w}}{o[0]:>22,}{o[1]:>22,}{sum(o):>12,}")
    print(f"{'col total':{w}}"
          f"{res['observed'][0][0] + res['observed'][1][0]:>22,}"
          f"{res['observed'][0][1] + res['observed'][1][1]:>22,}{res['n']:>12,}")
    print("  expected counts under independence:")
    for i in range(2):
        e = res["expected"][i]
        print(f"    {rl[i]:{w}}{e[0]:>22.4f}{e[1]:>22.4f}")
    print(f"  chi-square (uncorrected) = {res['chi2']:.4f}")
    print(f"  chi-square (Yates)       = {res['chi2_yates']:.4f}")
    print(f"  critical value, df=1, alpha=0.05 = {CRITICAL_CHI2_DF1:.4f}")
    verdict = "REJECT independence" if res["chi2"] > CRITICAL_CHI2_DF1 else "retain independence"
    print(f"  verdict                  = {verdict}")
    print(f"  odds ratio               = {res['odds_ratio']:.4f}")
    print(f"  phi coefficient          = {res['phi']:.4f}")


def part2(df, panel):
    print("\n\n=== PART 2: PATTERNS, CHARTS AND ASSOCIATION PROOF ===\n")
    print(f"study window        : {panel.date.min().date()} to {panel.date.max().date()}")
    print(f"days in panel       : {len(panel):,}")
    print(f"goods collisions    : {int(panel.collisions.sum()):,}")
    print(f"of which KSI        : {int(panel.ksi.sum()):,} "
          f"({100 * panel.ksi.sum() / panel.collisions.sum():.4f}%)")
    print(f"mean daily KSI      : {panel.ksi.mean():.4f}")
    print(f"mean daily risk idx : {panel.risk_index.mean():.4f} KSI per 100 collisions")
    print("\n--- by year ---")
    yr = panel.groupby("year").agg(collisions=("collisions", "sum"), ksi=("ksi", "sum"),
                                   risk=("risk_index", "mean"),
                                   pm10=("pm10_mean", "mean")).round(4)
    print(yr.to_string())
    print("\n--- by month, all years pooled ---")
    mo = panel.groupby("month").agg(ksi=("ksi", "mean"), risk=("risk_index", "mean"),
                                    fog=("fog_share", "mean"),
                                    pm10=("pm10_mean", "mean")).round(4)
    print(mo.to_string())
    plot_daily(panel)
    print(f"\nwrote {FIG}/daily_risk_index.png and .pdf")

    g = df[df.any_goods.fillna(False)].copy()
    g["loss_of_control"] = g.any_skid.fillna(False)
    g["exceed"] = (g.pm10 >= PM10_LIMIT).where(g.pm10.notna())
    g["dark"] = g.light_conditions.isin([4, 5, 6, 7])
    g["fatal"] = g.collision_severity.eq(1)

    def table(expo, out, rows, cols, title):
        m = expo.notna() & out.notna()
        e, o = expo[m].astype(bool), out[m].astype(bool)
        r = manual_chi2(int((e & o).sum()), int((e & ~o).sum()),
                        int((~e & o).sum()), int((~e & ~o).sum()), rows, cols)
        show_chi2(r, title)
        return r

    print("\n" + "=" * 70)
    print(" The brief asserts one dependency. The data contains two, with opposite")
    print(" signs, so they are tested separately rather than pooled.")
    print("=" * 70)

    loc_cols = ["loss of control", "no such event"]
    t1 = table(g.is_fog, g.loss_of_control, ["fog or mist", "clear"], loc_cols,
               "Table 2.1  RECORDED VISIBILITY: fog or mist x loss of control")
    t2 = table(g.is_fog, g.fatal, ["fog or mist", "clear"], ["fatal", "not fatal"],
               "Table 2.2  RECORDED VISIBILITY: fog or mist x fatal outcome")
    t3 = table(g.exceed, g.loss_of_control,
               ["PM10 >= 50 ug/m3", "PM10 below limit"], loc_cols,
               "Table 2.3  PARTICULATE CHANNEL: PM10 exceedance x loss of control")
    t4 = table(g.is_fog | g.exceed.fillna(False), g.loss_of_control,
               ["fog OR PM10 exceedance", "neither"], loc_cols,
               "Table 2.4  THE POOLED COMPOSITE THE BRIEF IMPLIES - and why it fails")
    print("\n  Tables 2.1 and 2.3 are both significant and point in OPPOSITE")
    print("  directions, so pooling them into one 'low visibility' exposure cancels")
    print("  the effect and reports a false null. Table 2.4 is reported to show that.")
    t5 = table(g.dark, g.loss_of_control, ["darkness", "daylight"], loc_cols,
               "Table 2.5  POSITIVE CONTROL: darkness x loss of control")
    print("\n  Table 2.5 is a known road-safety risk factor, included to show the test")
    print("  detects an association when one is present.")

    # --- why the particulate channel inverts -------------------------------
    print("\n\n--- Why the particulate channel carries the opposite sign ---\n")
    gp = g[g.pm10.notna()]
    rows = []
    for lab, sel in [("PM10 exceedance day", gp.exceed.astype(bool)),
                     ("all other days", ~gp.exceed.astype(bool))]:
        s = gp[sel]
        rows.append({"day type": lab,
                     "dry road %": round(100 * (s.road_surface_conditions == 1).mean(), 4),
                     "wet road %": round(100 * (s.road_surface_conditions == 2).mean(), 4),
                     "raining %": round(100 * (s.weather_conditions == 2).mean(), 4),
                     "fog %": round(100 * s.is_fog.mean(), 4),
                     "loss of control %": round(100 * s.loss_of_control.mean(), 4)})
    conf = pd.DataFrame(rows)
    print(conf.to_string(index=False))
    r_wet = panel[["pm10_mean", "wet_share"]].dropna().corr().iloc[0, 1]
    r_fog = panel[["pm10_mean", "fog_share"]].dropna().corr().iloc[0, 1]
    mar = 100 * (gp[gp.exceed.astype(bool)].date.dt.month == 3).mean()
    print(f"\n  daily PM10 against wet-surface share : r = {r_wet:.4f}")
    print(f"  daily PM10 against fog share         : r = {r_fog:.4f}")
    print(f"  share of exceedance collisions in March: {mar:.4f}%")
    print("\n  A regional PM10 exceedance in Great Britain is predominantly a dry,")
    print("  settled, anticyclonic day, and the March concentration is consistent with")
    print("  spring Saharan dust intrusion. Loss of control needs a wet or icy surface.")
    print("  So PM10 exceedance is negatively associated with loss of control through")
    print("  the weather regime, not because airborne particulate is protective.")
    print("  PM10 does track fog (fog is "
          f"{conf.loc[0, 'fog %'] / conf.loc[1, 'fog %']:.4f}x more likely on an")
    print("  exceedance day), so the two channels do co-occur - but a station-scale")
    print("  air-quality feed cannot stand in for a micro-scale visibility reading.")

    # --- the guideline's literal day-level speed-variance test --------------
    med_disp = panel.speed_dispersion.median()
    p = panel.dropna(subset=["speed_dispersion", "pm10_mean"]).copy()
    p["low_vis_day"] = (p.fog_share > 0) | (p.pm10_exceed_share > 0)
    p["high_var"] = p.speed_dispersion > med_disp
    t6 = manual_chi2(int(((p.low_vis_day) & (p.high_var)).sum()),
                     int(((p.low_vis_day) & (~p.high_var)).sum()),
                     int(((~p.low_vis_day) & (p.high_var)).sum()),
                     int(((~p.low_vis_day) & (~p.high_var)).sum()),
                     ["low-visibility day", "clear day"],
                     ["high speed variance", "low speed variance"])
    show_chi2(t6, "Table 2.6  day-level: low-visibility day x high speed-limit "
                  f"dispersion (median split {med_disp:.4f} mph)")
    print("\n  Day-level aggregation dilutes a micro-scale exposure across every")
    print("  collision that day, which is why the collision-level tables above are")
    print("  the ones the conclusions rest on.")
    return {"fog_loc": t1, "fog_fatal": t2, "pm10_loc": t3, "pooled": t4,
            "dark": t5, "speed_var": t6, "confound": conf,
            "r_wet": float(r_wet), "r_fog": float(r_fog), "march_pct": float(mar)}


# =============================================================================
# PART 3
# =============================================================================
def part3(df, panel):
    print("\n\n=== PART 3: FEATURE TRANSFORMATION, PCA AND POISSON COUNT MODEL ===\n")
    g = df[df.any_goods.fillna(False)].copy()
    cat = ["weather_label", "surface_label", "light_label", "road_class_label"]
    con = ["speed_limit", "number_of_vehicles", "pm10", "mean_driver_age", "aurn_dist_km"]
    feat = g[cat + con].copy()
    for c in con:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
        feat[c] = feat[c].fillna(feat[c].median())
    dummies = pd.get_dummies(feat[cat], prefix=cat, dummy_na=False).astype(float)
    print(f"one-hot encoding    : {len(cat)} categorical fields -> "
          f"{dummies.shape[1]} indicator columns")
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(feat[con]), columns=con,
                          index=feat.index)
    print(f"min-max scaling     : {len(con)} continuous fields scaled to [0, 1]")
    X = pd.concat([scaled, dummies], axis=1)
    print(f"design matrix       : {X.shape[0]:,} rows x {X.shape[1]} columns")

    pca = PCA().fit(X.values)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    print(f"\n--- PCA ---")
    print(f"  components retaining 95% of variance : {k95} of {X.shape[1]}")
    print(f"  cumulative variance at {k95} components : {100 * cum[k95 - 1]:.4f}%")
    print(f"  components retaining 99% of variance : {k99}")
    print(f"  cumulative variance at {k99} components : {100 * cum[k99 - 1]:.4f}%")
    print("  first eight components, individual and cumulative share:")
    for i in range(8):
        print(f"    PC{i + 1:<2d} {100 * pca.explained_variance_ratio_[i]:8.4f}%"
              f"   cumulative {100 * cum[i]:8.4f}%")

    # --- Poisson GLM on the daily KSI count, exposure-adjusted ---
    p = panel.dropna(subset=["pm10_mean", "speed_dispersion"]).copy()
    p["pm10_exceed_day"] = (p.pm10_exceed_share > 0).astype(float)
    p["fog_day"] = (p.fog_share > 0).astype(float)
    terms = ["fog_day", "pm10_exceed_day", "pm10_mean", "dark_share", "wet_share",
             "speed_dispersion", "mean_speed_limit"]
    Xg = sm.add_constant(p[terms].astype(float))
    model = sm.GLM(p.ksi.astype(float), Xg, family=sm.families.Poisson(),
                   offset=np.log(p.collisions.astype(float))).fit()
    print("\n--- Poisson regression: daily goods-vehicle KSI count ---")
    print("    offset = log(daily goods-vehicle collisions), so coefficients are "
          "rate effects")
    print(model.summary())
    print("\n--- risk multipliers, exp(coefficient) ---")
    print(f"{'term':<22}{'coef':>12}{'exp(coef)':>12}{'p':>12}  significant")
    for t in ["const"] + terms:
        coef = model.params[t]
        pv = model.pvalues[t]
        star = "yes (p < 0.05)" if pv < 0.05 else "no"
        print(f"{t:<22}{coef:>12.4f}{np.exp(coef):>12.4f}{pv:>12.4f}  {star}")
    sig = [t for t in terms if model.pvalues[t] < 0.05]
    print(f"\n  significant terms: {', '.join(sig) if sig else 'none'}")
    for t in sig:
        mult = np.exp(model.params[t])
        direction = "raises" if mult > 1 else "lowers"
        print(f"    {t}: a one-unit rise {direction} the KSI rate by a factor of "
              f"{mult:.4f} ({100 * abs(mult - 1):.4f}%)")
    return model, {"k95": k95, "cum95": float(cum[k95 - 1]), "ncols": int(X.shape[1])}


# =============================================================================
# PART 4
# =============================================================================
def part4(df, seed=42):
    print("\n\n=== PART 4: THE AI FAILURE AUDIT ===\n")
    g = df[df.any_goods.fillna(False)].copy()
    g["target"] = g.collision_severity.map(SEV_CLASS)
    g = g.dropna(subset=["target"])

    # Features the vehicle could observe before the outcome is known. Casualty
    # counts and casualty severity are excluded: they are the outcome.
    num = ["speed_limit", "number_of_vehicles", "pm10", "mean_driver_age",
           "aurn_dist_km", "n_goods", "n_articulated"]
    cat = ["weather_label", "surface_label", "light_label", "road_class_label"]
    flag = ["any_skid", "any_decel", "pm10_was_missing", "speed_limit_was_missing"]
    g["hour"] = g.datetime.dt.hour.fillna(12)
    g["month_n"] = g.date.dt.month
    num += ["hour", "month_n"]

    F = g[num].apply(pd.to_numeric, errors="coerce")
    F = F.fillna(F.median())
    F = pd.concat([F,
                   g[flag].astype(float).fillna(0.0),
                   pd.get_dummies(g[cat], prefix=cat).astype(float)], axis=1)
    y = g.target.values
    print(f"design matrix       : {F.shape[0]:,} rows x {F.shape[1]} features")
    print(f"class balance       : "
          f"{ {k: int(v) for k, v in pd.Series(y).value_counts().items()} }")
    print("  excluded as outcome leakage: number_of_casualties, worst_casualty, "
          "casualty_severity")

    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        F.values, y, g, test_size=0.25, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                 class_weight="balanced", n_jobs=-1,
                                 random_state=seed)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(f"\nclassifier          : RandomForest, 400 trees, min_samples_leaf=3, "
          f"class_weight='balanced', seed={seed}")
    print(f"train / test        : {len(ytr):,} / {len(yte):,}")

    labels = ["Safe", "Minor", "Severe"]
    pr, rc, f1, sup = precision_recall_fscore_support(yte, pred, labels=labels,
                                                      zero_division=0)
    print("\n--- class-balanced evaluation ---")
    print(f"{'class':<10}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>10}")
    for i, lab in enumerate(labels):
        print(f"{lab:<10}{pr[i]:>12.4f}{rc[i]:>12.4f}{f1[i]:>12.4f}{sup[i]:>10,}")
    mpr, mrc, mf1, _ = precision_recall_fscore_support(yte, pred, labels=labels,
                                                       average="macro", zero_division=0)
    acc = float((pred == yte).mean())
    base = float(pd.Series(ytr).value_counts(normalize=True).max())
    print(f"\n  macro precision   = {mpr:.4f}")
    print(f"  macro recall      = {mrc:.4f}")
    print(f"  macro F1          = {mf1:.4f}")
    print(f"  accuracy          = {acc:.4f}")
    print(f"  majority baseline = {base:.4f}")
    print("\n--- confusion matrix, rows = actual, columns = predicted ---")
    cm = confusion_matrix(yte, pred, labels=labels)
    print(f"{'actual':<10}" + "".join(f"{l:>12}" for l in labels))
    for i, lab in enumerate(labels):
        print(f"{lab:<10}" + "".join(f"{cm[i][j]:>12,}" for j in range(3)))

    print("\n--- THE CRITICAL CORE: false negatives ---")
    print("    a false negative here is a collision the model called 'Safe' that was "
          "actually 'Severe' (fatal).")
    te = gte.copy()
    te["predicted"] = pred
    fn = te[(te.predicted == "Safe") & (te.target == "Severe")]
    print(f"    false negatives: {len(fn):,} of {int((te.target == 'Severe').sum()):,} "
          f"actual Severe cases "
          f"({100 * len(fn) / max(int((te.target == 'Severe').sum()), 1):.4f}%)")
    cols = ["collision_index", "date", "speed_limit", "road_class_label",
            "weather_label", "light_label", "surface_label", "pm10", "aurn_dist_km",
            "any_skid", "any_decel", "number_of_vehicles"]
    show = fn[cols].head(25).copy()
    show["pm10"] = show.pm10.round(4)
    show["aurn_dist_km"] = show.aurn_dist_km.round(4)
    print("\n    first 25 false-negative rows:")
    print(show.to_string(index=False))

    print("\n--- what the false negatives have in common ---")
    for col, name in [("weather_label", "weather"), ("light_label", "light"),
                      ("surface_label", "road surface"), ("road_class_label", "road class")]:
        fn_share = fn[col].value_counts(normalize=True)
        all_share = te[col].value_counts(normalize=True)
        comp = pd.DataFrame({"false_neg_pct": (100 * fn_share).round(4),
                             "test_set_pct": (100 * all_share).round(4)})
        comp["lift"] = (comp.false_neg_pct / comp.test_set_pct).round(4)
        print(f"\n  by {name}:")
        print(comp.sort_values("false_neg_pct", ascending=False).head(5).to_string())
    print(f"\n  PM10 reading missing in false negatives : "
          f"{100 * fn.pm10_was_missing.mean():.4f}%")
    print(f"  PM10 reading missing in whole test set  : "
          f"{100 * te.pm10_was_missing.mean():.4f}%")
    print(f"  mean PM10, false negatives              : {fn.pm10.mean():.4f}")
    print(f"  mean PM10, whole test set               : {te.pm10.mean():.4f}")
    print(f"  fog or mist share, false negatives      : {100 * fn.is_fog.mean():.4f}%")
    print(f"  fog or mist share, whole test set       : {100 * te.is_fog.mean():.4f}%")

    imp = pd.Series(clf.feature_importances_, index=F.columns).nlargest(12)
    print("\n--- top 12 feature importances ---")
    print(imp.round(4).to_string())
    lifts = {}
    for col in ["weather_label", "light_label", "surface_label", "road_class_label"]:
        fs = fn[col].value_counts(normalize=True) * 100
        als = te[col].value_counts(normalize=True) * 100
        cmp_ = pd.DataFrame({"level": fs.index, "fn_pct": fs.values})
        cmp_["test_pct"] = [float(als.get(l, float("nan"))) for l in cmp_.level]
        cmp_["lift"] = cmp_.fn_pct / cmp_.test_pct
        lifts[col] = cmp_.sort_values("fn_pct", ascending=False).head(5).round(4) \
            .to_dict("records")

    n_severe = int((te.target == "Severe").sum())
    return clf, te, fn, {
        "macro_f1": mf1, "macro_precision": mpr, "macro_recall": mrc,
        "accuracy": acc, "baseline": base, "n_fn": len(fn),
        "n_severe_test": n_severe,
        "fn_rate_severe": float(100 * len(fn) / max(n_severe, 1)),
        "per_class": [{"cls": lab, "precision": float(pr[i]), "recall": float(rc[i]),
                       "f1": float(f1[i]), "support": int(sup[i])}
                      for i, lab in enumerate(labels)],
        "confusion": [[int(x) for x in row] for row in cm],
        "labels": labels,
        "n_features": int(F.shape[1]),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "class_balance": {k: int(v) for k, v in pd.Series(y).value_counts().items()},
        "lifts": lifts,
        "fn_pm10_missing_pct": float(100 * fn.pm10_was_missing.mean()),
        "test_pm10_missing_pct": float(100 * te.pm10_was_missing.mean()),
        "fn_fog_pct": float(100 * fn.is_fog.mean()),
        "test_fog_pct": float(100 * te.is_fog.mean()),
        "top_features": pd.Series(clf.feature_importances_, index=F.columns)
                          .nlargest(12).round(6).to_dict(),
    }
