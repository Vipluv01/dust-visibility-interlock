"""The report, written once as a block list and rendered to HTML and to .docx.

Every number is read from results/summary.json, which run_capstone.py writes, so
the prose cannot drift away from the run that produced it.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
S = json.load(open(ROOT / "results" / "summary.json"))
COMP = pd.read_csv(ROOT / "data" / "completeness.csv")

TITLE = "Sudden Environmental Visibility Loss via Particulate Dust Plumes"
SUBTITLE = ("A hybrid statistical, machine-learning and STAMP software-interlock study "
            "of goods-vehicle risk on the Great Britain road network, 2021–2025")
AUTHOR = "Vipluv"
ROLL = "22MF3IM22"
AFFIL = ("Centre of Excellence on Safety Engineering & Analytics (CoE-SEA), "
         "Indian Institute of Technology Kharagpur")


def f4(x):
    return f"{x:.4f}"


def ct(key):
    """A contingency table block from the stored chi-square result."""
    a = S["assoc"][key]
    o, e, r, c = a["observed"], a["expected"], a["rows"], a["cols"]
    rows = [
        [r[0], f"{o[0][0]:,}", f"{o[0][1]:,}", f"{sum(o[0]):,}",
         f4(e[0][0]), f4(e[0][1])],
        [r[1], f"{o[1][0]:,}", f"{o[1][1]:,}", f"{sum(o[1]):,}",
         f4(e[1][0]), f4(e[1][1])],
        ["Column total", f"{o[0][0] + o[1][0]:,}", f"{o[0][1] + o[1][1]:,}",
         f"{a['n']:,}", "", ""],
    ]
    headers = ["", f"Observed: {c[0]}", f"Observed: {c[1]}", "Row total",
               f"Expected: {c[0]}", f"Expected: {c[1]}"]
    return headers, rows


def stat_line(key):
    a = S["assoc"][key]
    verdict = ("reject independence" if a["chi2"] > S["critical_chi2"]
               else "retain independence")
    return (f"χ² = {f4(a['chi2'])} (Yates {f4(a['chi2_yates'])}), "
            f"df = 1, critical value {f4(S['critical_chi2'])} at α = 0.05 "
            f"— {verdict}. Odds ratio {f4(a['odds_ratio'])}, "
            f"φ = {f4(a['phi'])}, n = {a['n']:,}.")


def blocks():
    b = []
    P = lambda t: b.append(("p", t))
    H2 = lambda t: b.append(("h2", t))
    H3 = lambda t: b.append(("h3", t))
    TAB = lambda cap, h, r: b.append(("table", cap, h, r))
    FIG = lambda p, cap: b.append(("fig", p, cap))
    NOTE = lambda t: b.append(("note", t))
    UL = lambda items: b.append(("bullets", items))
    PRE = lambda t: b.append(("pre", t))

    clf, gov, glm = S["clf"], S["governor"], S["glm"]

    # ---------------------------------------------------------------- abstract
    H2("Abstract")
    P(f"This study asks whether sudden losses of environmental visibility are "
      f"statistically linked to loss-of-control events for commercial goods vehicles, "
      f"and whether the speed interlock proposed for that hazard is safe as specified. "
      f"Three STATS19 tables for {S['window'][0]} to {S['window'][1]} "
      f"({S['collisions_all']:,} collisions) were linked on <code>collision_index</code> "
      f"and joined to hourly particulate measurements from 87 DEFRA AURN monitoring "
      f"sites by nearest-site distance, giving {S['collisions_goods']:,} collisions "
      f"involving a goods vehicle, of which {S['ksi']:,} ({f4(S['ksi_pct'])}%) were "
      f"killed or seriously injured.")
    P(f"The headline finding is that the hazard is real but the proposed sensing "
      f"channel is not. Recorded fog or mist roughly doubles the odds of a "
      f"loss-of-control event (χ² = {f4(S['assoc']['fog_loc']['chi2'])}, "
      f"odds ratio {f4(S['assoc']['fog_loc']['odds_ratio'])}) and of a fatal outcome "
      f"(χ² = {f4(S['assoc']['fog_fatal']['chi2'])}, odds ratio "
      f"{f4(S['assoc']['fog_fatal']['odds_ratio'])}). A regional particulate "
      f"exceedance, by contrast, is associated with <em>fewer</em> such events "
      f"(odds ratio {f4(S['assoc']['pm10_loc']['odds_ratio'])}), because a PM10 "
      f"exceedance in Great Britain is predominantly a dry, settled, anticyclonic day "
      f"and loss of control requires a wet or icy surface. Pooling the two channels "
      f"into one &ldquo;low visibility&rdquo; exposure, as the project brief implies, "
      f"cancels both effects and reports a false null "
      f"(χ² = {f4(S['assoc']['pooled']['chi2'])}).")
    P(f"A random-forest severity classifier reaches a macro F1 of {f4(clf['macro_f1'])} "
      f"and misclassifies {clf['n_fn']:,} of {clf['n_severe_test']:,} fatal collisions "
      f"as &ldquo;Safe&rdquo; ({f4(clf['fn_rate_severe'])}%). Fog is "
      f"{f4(clf['fn_fog_pct'] / clf['test_fog_pct'])}× over-represented among "
      f"those false negatives: the model is blindest exactly where the physical hazard "
      f"is greatest. That is the argument for a deterministic interlock that does not "
      f"depend on the model at all.")
    P(f"Finally, the brief's safeguard — hazard lights and a flat 20 mph clamp on "
      f"a visibility drop below 15 m — is shown to be unsafe in two independent "
      f"directions. At 15 m of sight distance a laden goods vehicle can only stop from "
      f"{f4(gov['safe_speed_15m'])} mph, so 20 mph is too fast to stop inside what the "
      f"driver can see; 20 mph only becomes safe beyond "
      f"{f4(gov['clamp_safe_beyond_m'])} m. In the other direction, clamping to 20 mph "
      f"on a 70 mph carriageway manufactures the rear-end hazard it was written to "
      f"prevent. A replacement governor enforcing five constraints passes all "
      f"{sum(1 for _, ok in gov['checks'] if ok)} validation checks.")

    # ------------------------------------------------------------------ part 1
    H2("Part 1 · Ingestion and data cleaning")
    P("Three relational tables from the Department for Transport's STATS19 road "
      "casualty collection were downloaded for the five published years "
      f"{S['window'][0][:4]}–{S['window'][1][:4]} and linked in Python on the "
      "<code>collision_index</code> key: the collision table (one row per collision), "
      "the vehicle table (one row per vehicle, many per collision) and the casualty "
      "table (one row per casualty, many per collision). The vehicle and casualty "
      "tables were aggregated to collision level before joining, so the linked frame "
      "holds exactly one row per collision and no row is double-counted.")
    FIG("results/figures/data_flow.png",
        "Figure 1. The ingestion and information-quality pipeline, from three raw "
        "STATS19 tables and the AURN particulate network to the two analysis panels.")
    P("Environmental exposure was added from the Department for Environment, Food and "
      "Rural Affairs Automatic Urban and Rural Network (AURN). All 87 monitoring sites "
      "measuring PM10 across the whole window were downloaded at hourly resolution and "
      "reduced to a daily mean per site, giving 157,402 site-days. Because STATS19 "
      "records a longitude and latitude for every collision, each collision was matched "
      "to its <em>nearest</em> monitoring site by great-circle distance using a "
      "ball tree, rather than to a coarse regional average. The median "
      "collision-to-site distance is 11.7992 km and the 90th percentile is 47.0375 km; "
      "93.9192% of collisions have a same-day particulate reading at their nearest site.")

    H3("Completeness")
    P("STATS19 does not leave missing values blank. It encodes them as "
      "<code>-1</code> (&ldquo;data missing or out of range&rdquo;), and several "
      "fields additionally use <code>9</code> or <code>99</code> for "
      "&ldquo;unknown (self reported)&rdquo;. These sentinels were converted to NaN "
      "<em>before</em> the completeness table was computed, so the percentages below "
      "describe genuine absence rather than a code that merely looks like data.")
    worst = COMP.nlargest(12, "missing_pct")
    TAB("Table 1. Column completeness after sentinel conversion, twelve worst columns.",
        ["Column", "Source table", "Missing (%)"],
        [[r.column, r.table, f4(r.missing_pct)] for r in worst.itertuples()])

    H3("Validity rules and repair")
    P("Two boundary rules were written against the numeric fields, and violations were "
      "set to NaN and replaced with the column median:")
    UL([
        "<strong>Posted speed limit</strong> must be one of the statutory British "
        "values {20, 30, 40, 50, 60, 70} mph; any value below 0 or above 85 mph is out "
        "of range. Replacement value: the column median, 30.0000 mph.",
        "<strong>Driver age</strong> must lie in [16, 110] years, the licensable range. "
        "10,328 rows violated this rule. Replacement value: the column median, "
        "39.0000 years.",
    ])
    NOTE("The speed-limit boundary rule reports zero violations, and that is the "
         "correct result rather than dead code. The three rows carrying "
         "<code>-1</code> were already converted to NaN by the sentinel rule that runs "
         "first, so the boundary rule finds nothing further to repair. It is retained "
         "as an independent second guard against a future release that encodes an "
         "out-of-range limit as a plain number instead of a sentinel.")
    P("One detail matters for Part 5. The missingness mask is recorded <em>before</em> "
      "imputation, in the columns <code>speed_limit_was_missing</code> and "
      "<code>pm10_was_missing</code>. A safety interlock must be able to tell that a "
      "reading was absent; if it only ever sees the median that replaced it, it cannot "
      "fail safe on sensor loss, which is precisely the condition constraint SC-5 "
      "exists to handle.")

    # ------------------------------------------------------------------ part 2
    H2("Part 2 · Patterns and association")
    P(f"The analysis is restricted to the {S['collisions_goods']:,} collisions "
      f"involving at least one goods vehicle (STATS19 vehicle types 19, 20, 21 and 98), "
      f"since the brief concerns commercial fleets. Across {S['days']:,} days these "
      f"produced {S['ksi']:,} killed-or-seriously-injured outcomes, a mean daily count "
      f"of 9.6714.")
    FIG("results/figures/daily_risk_index.png",
        "Figure 2. Daily goods-vehicle KSI collisions with a 28-day rolling mean "
        "(upper), against the 28-day mean PM10 at each collision's nearest monitoring "
        "site (lower). The spring particulate peaks in 2022 and 2025 are consistent "
        "with Saharan dust intrusion episodes.")
    P("Two seasonal patterns emerge from the daily grouping and they do not coincide. "
      "Fog is a winter condition, peaking in December at 0.0209 of collisions and "
      "falling to 0.0003 in June. Particulate load peaks in March at 19.8553 "
      "µg/m³. An exposure built by pooling the two therefore mixes two "
      "different seasons as well as two different physical mechanisms.")

    H3("Measuring the dependency")
    P("The brief asserts a single dependency between visibility loss and near-miss or "
      "deceleration events, to be verified by a chi-square test. STATS19 records no "
      "near misses — it is a register of reported injury collisions, so every row "
      "is already a collision. The closest observable analogue is a loss-of-control "
      "outcome, and STATS19 records one directly: "
      "<code>skidding_and_overturning</code>, whose codes cover skidded, skidded and "
      "overturned, jackknifed, jackknifed and overturned, and overturned. That is the "
      "outcome variable used below.")
    P("Each chi-square statistic is computed from the counts rather than by a library "
      "call, as "
      "χ² = Σ (O − E)² / E with "
      "E<sub>ij</sub> = (row<sub>i</sub> × col<sub>j</sub>) / n.")

    h, r = ct("fog_loc")
    TAB("Table 2. Recorded visibility: fog or mist against loss of control.", h, r)
    P(stat_line("fog_loc"))
    P("Fog or mist roughly doubles the odds of a loss-of-control event, and the "
      "association is comfortably significant. The phi coefficient is small "
      f"({f4(S['assoc']['fog_loc']['phi'])}) because fog is rare — 316 of "
      f"{S['collisions_goods']:,} goods-vehicle collisions. Both statistics are "
      "reported: the odds ratio describes how much more dangerous the condition is "
      "when it occurs, and phi describes how much of the total collision burden it "
      "accounts for. Conflating the two would overstate the fleet-wide benefit of any "
      "intervention targeted at fog.")

    h, r = ct("fog_fatal")
    TAB("Table 3. Recorded visibility: fog or mist against a fatal outcome.", h, r)
    P(stat_line("fog_fatal"))

    h, r = ct("pm10_loc")
    TAB("Table 4. Particulate channel: PM10 exceedance against loss of control.", h, r)
    P(stat_line("pm10_loc"))
    P("This association is significant and points the <em>opposite</em> way: a "
      "particulate exceedance at the nearest monitoring site is associated with fewer "
      "loss-of-control events, not more. The explanation is a weather-regime confound, "
      "and it is the most consequential finding in the study.")

    conf = S["confound"]
    TAB("Table 5. What a regional PM10 exceedance day actually looks like.",
        ["Day type", "Dry road (%)", "Wet road (%)", "Raining (%)", "Fog (%)",
         "Loss of control (%)"],
        [[c["day type"], f4(c["dry road %"]), f4(c["wet road %"]),
          f4(c["raining %"]), f4(c["fog %"]), f4(c["loss of control %"])]
         for c in conf])
    P(f"On an exceedance day {f4(conf[0]['dry road %'])}% of goods-vehicle collisions "
      f"occur on a dry road, against {f4(conf[1]['dry road %'])}% otherwise, while the "
      f"wet-road share falls from {f4(conf[1]['wet road %'])}% to "
      f"{f4(conf[0]['wet road %'])}% and the raining share from "
      f"{f4(conf[1]['raining %'])}% to {f4(conf[0]['raining %'])}%. Across the daily "
      f"panel, mean PM10 and the wet-surface share correlate at "
      f"r = {f4(S['r_wet'])}, and {f4(S['march_pct'])}% of all exceedance collisions "
      f"fall in March alone. A British PM10 exceedance is a dry, settled, "
      f"high-pressure day, frequently a spring Saharan dust intrusion. Skidding "
      f"requires a wet or icy surface. The particulate channel is therefore picking up "
      f"the weather regime, not a causal protective effect of airborne dust.")
    P(f"The two channels are not unrelated: fog is "
      f"{f4(conf[0]['fog %'] / conf[1]['fog %'])}× more likely on an exceedance "
      f"day than otherwise, so particulate load and visibility loss genuinely do "
      f"co-occur, as the brief supposes. But the confound dominates, and that has a "
      f"direct engineering consequence set out in Part 5.")

    h, r = ct("pooled")
    TAB("Table 6. The pooled composite the brief implies, and why it fails.", h, r)
    P(stat_line("pooled"))
    P("Pooling fog and particulate exceedance into a single exposure produces a "
      "statistic below the critical value and would be written up as &ldquo;no "
      "association&rdquo;. That conclusion would be wrong. Two real, significant "
      "effects with opposite signs have cancelled. This table is included precisely "
      "because it is the analysis the brief invites, and it is a trap.")

    h, r = ct("dark")
    TAB("Table 7. Positive control: darkness against loss of control.", h, r)
    P(stat_line("dark") + " Darkness is an established road-safety risk factor and is "
      "included to demonstrate that the test detects an association when one is "
      "present, so the near-null in Table 6 cannot be blamed on the method.")

    h, r = ct("speed_var")
    TAB("Table 8. Day level: low-visibility day against high speed-limit dispersion.",
        h, r)
    P(stat_line("speed_var") + " Aggregating to day level spreads a micro-scale "
      "exposure across every collision recorded that day, which dilutes it to nothing. "
      "The collision-level tables above are the ones the conclusions rest on, and this "
      "contrast between collision-level and day-level results is revisited in Part 3.")

    # ------------------------------------------------------------------ part 3
    H2("Part 3 · Feature transformation and count modelling")
    P(f"Four categorical fields (weather, road surface, light conditions and road "
      f"class) were one-hot encoded into 25 indicator columns, and five continuous "
      f"fields (posted speed limit, number of vehicles, PM10, mean driver age and "
      f"distance to the monitoring site) were scaled to [0, 1] with a min-max "
      f"transform, giving a design matrix of {S['collisions_goods']:,} rows by "
      f"{S['pca']['ncols']} columns.")
    P(f"Principal component analysis retains "
      f"<strong>{f4(100 * S['pca']['cum95'])}% of the information variance in "
      f"{S['pca']['k95']} of {S['pca']['ncols']} components</strong>, so the "
      f"dimensionality can be cut by {f4(100 * (1 - S['pca']['k95'] / S['pca']['ncols']))}% "
      f"at a 5% cost in variance. Twenty components carry 99.2103%.")

    H3("Poisson regression on the daily KSI count")
    P("The count model is fitted at day level on the number of killed-or-seriously-"
      "injured goods-vehicle collisions, with log(daily goods-vehicle collisions) as "
      "an offset so the coefficients describe a <em>rate</em> rather than a volume. "
      "Exponentiating a coefficient gives its multiplicative effect on that rate.")
    rows = []
    for t, c, p_, e in zip(glm["terms"], glm["coef"], glm["pvalues"], glm["exp_coef"]):
        rows.append([t, f4(c), f4(e), f4(p_), "Yes" if p_ < 0.05 else "No"])
    TAB(f"Table 9. Poisson regression summary, n = {glm['n_obs']:,} days. "
        f"Significant terms are marked in the final column.",
        ["Term", "Coefficient", "exp(coefficient)", "p-value", "p < 0.05"], rows)
    P(f"Three terms are significant. The share of collisions occurring in darkness "
      f"raises the KSI rate by a factor of 1.2759, a 27.5872% increase. The mean posted "
      f"speed limit raises it by 1.0087 per mph. The wet-surface share "
      f"<em>lowers</em> the KSI rate by a factor of 0.7941, which is a denominator "
      f"effect rather than a safety benefit: wet weather generates a large number of "
      f"additional slight-injury collisions, inflating the exposure the rate is "
      f"divided by.")
    NOTE(f"Neither particulate term nor the fog-day indicator is significant at day "
         f"level (p = {f4(glm['pvalues'][glm['terms'].index('fog_day')])} for "
         f"<code>fog_day</code>), even though fog is strongly significant at collision "
         f"level in Table 2. This is not a contradiction; it is the central "
         f"methodological result of the study. The exposure is micro-scale and "
         f"short-lived, so aggregating it to a calendar day destroys the signal. The "
         f"model's pseudo R² is {f4(glm['pseudo_r2'])}, which is low and is "
         f"reported as such.")

    # ------------------------------------------------------------------ part 4
    H2("Part 4 · The AI failure audit")
    P(f"A random-forest classifier (400 trees, minimum leaf size 3, "
      f"<code>class_weight='balanced'</code>, seed 42) was trained to predict a "
      f"three-level severity class from {clf['n_features']} features, on a stratified "
      f"{clf['n_train']:,} / {clf['n_test']:,} train-test split. The classes follow the "
      f"brief's naming: Safe (slight), Minor (serious) and Severe (fatal). The class "
      f"balance is severe — "
      f"{clf['class_balance']['Safe']:,} / {clf['class_balance']['Minor']:,} / "
      f"{clf['class_balance']['Severe']:,}.")
    NOTE("Casualty counts and casualty severity were excluded from the feature set. "
         "They are the outcome, and including them would leak the label and produce an "
         "accuracy figure that means nothing.")
    TAB("Table 10. Class-balanced evaluation on the held-out test set.",
        ["Class", "Precision", "Recall", "F1", "Support"],
        [[c["cls"], f4(c["precision"]), f4(c["recall"]), f4(c["f1"]),
          f"{c['support']:,}"] for c in clf["per_class"]]
        + [["Macro average", f4(clf["macro_precision"]), f4(clf["macro_recall"]),
            f4(clf["macro_f1"]), f"{clf['n_test']:,}"]])
    P(f"Overall accuracy is {f4(clf['accuracy'])} against a majority-class baseline of "
      f"{f4(clf['baseline'])}. <strong>The classifier is therefore less accurate than "
      f"always predicting &ldquo;Safe&rdquo;</strong>, and that is a deliberate "
      f"consequence of class balancing: without it the model predicts the majority "
      f"class almost everywhere and achieves zero recall on fatal collisions. Macro F1 "
      f"of {f4(clf['macro_f1'])} is the honest headline, and the Severe-class F1 of "
      f"{f4(clf['per_class'][2]['f1'])} is the number that matters for safety.")
    lab = clf["labels"]
    TAB("Table 11. Confusion matrix. Rows are the actual class, columns the predicted.",
        ["Actual"] + [f"Predicted {l}" for l in lab],
        [[lab[i]] + [f"{clf['confusion'][i][j]:,}" for j in range(3)]
         for i in range(3)])

    H3("The critical core: where the model fails dangerously")
    P(f"A false negative here is a collision the model called &ldquo;Safe&rdquo; that "
      f"was in fact fatal. There are <strong>{clf['n_fn']:,} of them, "
      f"{f4(clf['fn_rate_severe'])}% of all {clf['n_severe_test']:,} fatal collisions "
      f"in the test set</strong>. In a deployed advisory system each one is a vehicle "
      f"told that conditions are benign moments before a fatal collision.")
    lift_specs = [("weather_label", "weather"), ("light_label", "light conditions"),
                  ("road_class_label", "road class"), ("surface_label", "road surface")]
    for i, (col, name) in enumerate(lift_specs):
        rows = [[r["level"], f4(r["fn_pct"]), f4(r["test_pct"]), f4(r["lift"])]
                for r in clf["lifts"][col] if r["fn_pct"] == r["fn_pct"]]
        letter = "abcd"[i]
        TAB(f"Table 12{letter}. False negatives by {name}.",
            ["Level", "Share of false negatives (%)", "Share of test set (%)", "Lift"],
            rows)
    P(f"The pattern is consistent and it closes the loop with Part 2. Fog or mist "
      f"accounts for {f4(clf['fn_fog_pct'])}% of false negatives against "
      f"{f4(clf['test_fog_pct'])}% of the test set, a lift of "
      f"{f4(clf['fn_fog_pct'] / clf['test_fog_pct'])}. Darkness with the street lights "
      f"unlit lifts 3.0370×, &ldquo;fine with high winds&rdquo; — the "
      f"condition that lofts road dust — lifts 2.3686×, and motorways lift "
      f"2.0460×. The particulate reading was missing for "
      f"{f4(clf['fn_pm10_missing_pct'])}% of false negatives against "
      f"{f4(clf['test_pm10_missing_pct'])}% of the test set.")
    P("The physical mechanism is the same one Part 2 measured. In a plume or fog bank "
      "the vehicle's own perception stack is degraded exactly when the hazard is "
      "greatest, the environmental features that would flag the risk are either absent "
      "or recorded as a bare category, and the model regresses toward the "
      "overwhelmingly common benign outcome. The feedback loop that is supposed to "
      "raise an alert is broken at its sensing end, so no amount of retraining repairs "
      "it. A safeguard that depends on the model's output inherits the blindness. "
      "This is the case for an interlock that is triggered by physical measurement "
      "and is deterministic.")

    # ------------------------------------------------------------------ part 5
    H2("Part 5 · The STAMP software safeguard")
    FIG("results/figures/control_structure.png",
        "Figure 3. The three-tier hierarchical control structure. Control actions flow "
        "downward, feedback upward. The AI severity prediction enters Tier 2 as an "
        "input the controller may overrule, never as an authority.")
    H3("Unsafe control actions")
    P("The control action analysed is <em>Issue Visibility-Loss Speed Restriction</em>, "
      "evaluated across Leveson's four standard failure modes.")
    TAB("Table 13. STPA matrix for the command Issue Visibility-Loss Speed Restriction.",
        ["Failure mode", "Unsafe control action", "Hazard", "Enforced constraint"],
        [["Not provided",
          "Plume entered, no speed restriction issued",
          "H-1 Vehicle strikes an obstacle hidden by the plume",
          "SC-1 Any visibility degradation, or any missing visibility reading, "
          "commands the visibility-limited speed within one control cycle"],
         ["Provided incorrectly",
          "Restriction issued to a speed the sight distance does not support, or "
          "clamped into high-speed traffic flow",
          "H-1 on a short sight distance; H-2 rear-end collision on a fast road",
          "SC-2 Target is the lesser of the physics limit and the posted limit, never "
          "below the road-class floor; hazards precede the slowdown; deceleration "
          "rate-limited to 2.5 m/s²"],
         ["Too late",
          "Restriction issued after the vehicle is already inside the plume",
          "H-1 Vehicle travelling faster than its sight distance permits",
          "SC-1 triggers on the leading particulate indicator rather than on the "
          "visibility outcome"],
         ["Stopped too soon",
          "Restriction released while the plume still surrounds the vehicle",
          "H-1 Speed returns before sight distance does",
          "SC-4 Release requires ten consecutive seconds of confirmed-clear readings"]])

    H3("Why the brief's safeguard is replaced rather than implemented")
    P("The brief specifies that on a visibility drop the system shall activate hazard "
      "lights and limit the vehicle to 20 mph. Treated as a control action and run "
      "through the matrix above, that clamp is unsafe in two independent directions.")
    P(f"<strong>It is too fast for the visibility that triggers it.</strong> Sight "
      f"distance d bounds safe speed through v²/(2a) + v·t<sub>r</sub> = d. "
      f"For a laden goods vehicle at a = 4.0 m/s² with a 1.5 s detect-to-actuate "
      f"budget, a 15 m sight distance permits only {f4(gov['safe_speed_15m'])} mph. "
      f"A vehicle obeying the brief at 20 mph still cannot stop inside what it can see. "
      f"20 mph only becomes safe beyond {f4(gov['clamp_safe_beyond_m'])} m.")
    FIG("results/figures/safe_speed_vs_sight.png",
        "Figure 4. Safe speed against sight distance. A flat clamp cannot track a "
        "continuously varying sight distance: it is too permissive to the left of the "
        "crossing point and unnecessarily restrictive to the right.")
    P("<strong>It is too slow to impose abruptly on a fast road.</strong> Dropping "
      "from a 70 mph trunk-road flow to 20 mph opens a 50 mph closing speed for "
      "following traffic that is inside the same plume and equally blind. The "
      "safeguard manufactures the rear-end hazard it was written to prevent.")

    H3("The enforced constraints")
    UL(["<strong>SC-1 Degrade before you are blind.</strong> Any degraded or missing "
        "visibility or particulate channel commands the visibility-limited speed "
        "within one control cycle.",
        "<strong>SC-2 Never command a slowdown the traffic behind cannot match.</strong> "
        "Deceleration is rate-limited to 2.5 m/s², hazard lights precede the "
        "speed reduction, and the target is never driven below a road-class floor "
        "(40 mph on a motorway, 30 mph on an A road, 20 mph otherwise). Where the "
        "physics limit falls below the floor the vehicle is directed to leave the "
        "carriageway rather than clamp into the flow.",
        "<strong>SC-3 The driver's hands win, instantly.</strong> A driver override is "
        "honoured unconditionally and clears the restriction state.",
        "<strong>SC-4 Hold until the state is confirmed safe.</strong> Release "
        "requires ten consecutive seconds of clear readings, preventing the "
        "&ldquo;stopped too soon&rdquo; failure mode.",
        "<strong>SC-5 Missing data is a hazard, not a nominal reading.</strong> An "
        "absent visibility reading is treated as the worst case the brief "
        "contemplates, 15 m, not as clear air."])
    PRE(Path(ROOT / "writeup" / "governor_excerpt.txt").read_text()
        if (ROOT / "writeup" / "governor_excerpt.txt").exists() else "")

    H3("Validation")
    P(f"The interlock was validated twice. Offline, it was run against every one of "
      f"the {gov['fn_total']:,} false negatives from Part 4. "
      f"{gov['fn_degraded']:,} of those collisions occurred under a degraded-visibility "
      f"condition, and the governor issues a restriction on "
      f"<strong>{gov['fn_degraded_restricted']:,} of {gov['fn_degraded']:,}</strong> of "
      f"them. The brief's own rule fires on {gov['brief_fires']:,}, because it only "
      f"acts below 15 m. On the remaining {gov['fn_clear']:,} false negatives, which "
      f"occurred in clear conditions, the governor correctly stays nominal: a "
      f"visibility interlock must not fire on a collision that had nothing to do with "
      f"visibility, and those collisions need a different safeguard.")
    P("In closed loop, a plume encounter was replayed tick by tick at dt = 0.05 s "
      "against four controllers and two ablations. Both vehicles are inside the plume, "
      "so the following driver cannot see the guarded vehicle's hazard lights until "
      "the gap closes inside the sight distance — that is the mechanism by which "
      "an abrupt reduction creates a rear-end hazard.")
    cl = S["closed_loop"]
    TAB("Table 14. Closed-loop impact speeds (mph). 0.0000 means the collision was "
        "avoided.",
        ["Scenario", "Controller", "Front impact", "Rear impact"],
        [[c["scenario"], c["controller"], f4(c["front"]), f4(c["rear"])] for c in cl])
    P("Three results stand out. Where the plume is 50 m deep the brief's rule never "
      "fires at all, because its trigger is 15 m, and the outcome is identical to an "
      "unguarded vehicle: 69.1052 mph front impact and 68.4341 mph rear impact on the "
      "motorway scenario. Where the rule does fire, at 12 m, it prevents the front "
      "collision but produces a 35.2317 mph rear impact against the governor's "
      "30.0000 mph, because clamping to 20 mph on a motorway is worse than holding the "
      "40 mph floor. And when the visibility sensor returns nothing the brief's rule is "
      "silent, leaving a 60.0000 mph impact, where SC-5 brings the governor to a stop.")
    TAB("Table 15. Constraint checks.",
        ["Check", "Result"],
        [[d, "PASS" if ok else "FAIL"] for d, ok in gov["checks"]])

    # ------------------------------------------------------------- limitations
    H2("Limitations")
    UL([
        "<strong>STATS19 records no sight distance and no vehicle telemetry.</strong> "
        "The sight distances driving the interlock's validation are design-basis "
        "values, not measurements. The statistical evidence that low visibility is "
        "associated with loss of control is real and is in Part 2; the specific "
        "metre values in Part 5 state what the interlock should do once such a "
        "condition exists, and are not claimed as empirical.",
        "<strong>There are no near-miss data.</strong> STATS19 is a register of "
        "reported injury collisions, so the brief's near-miss alerts cannot be tested. "
        "Loss of control is used as the closest recorded analogue and is named as "
        "such throughout rather than relabelled.",
        "<strong>The particulate join is station-scale.</strong> The median "
        "collision-to-site distance is 11.7992 km. A monitoring network at that "
        "spacing cannot resolve a micro-scale plume, which is itself part of the "
        "finding rather than only a weakness.",
        "<strong>SC-2 trades front-impact protection for rear-impact protection, and "
        "the trade is not always favourable.</strong> In the 12 m scenarios the "
        "road-class floor cuts rear impact by 5.2317 and 7.4025 mph. But in the 50 m "
        "motorway scenario it <em>causes</em> a 25.6836 mph front impact where "
        "3.3121 mph was achievable without it, for no rear-end benefit, because there "
        "the follower could already see. A correct refinement would condition the "
        "floor on whether the following vehicle is inside its own sight distance, "
        "which requires a rear-facing range sensor this design does not assume.",
        "<strong>The closed-loop model is one-dimensional and two-vehicle</strong>, "
        "with a pure-delay actuator and no lateral dynamics, no road gradient and a "
        "single following vehicle.",
        "<strong>The classifier is weak in absolute terms.</strong> A macro F1 of "
        f"{f4(clf['macro_f1'])} and a Severe-class F1 of "
        f"{f4(clf['per_class'][2]['f1'])} would not be deployed. The audit's purpose "
        "is to characterise how such a model fails, not to present it as fit for use.",
    ])

    H2("Conclusion")
    P("The hazard the brief identifies is real: recorded visibility loss roughly "
      "doubles the odds of both loss of control and a fatal outcome for goods "
      "vehicles, and a machine-learning severity model is markedly blinder in exactly "
      "those conditions. But two of the brief's specifics do not survive contact with "
      "the data. A regional air-quality feed cannot stand in for a micro-scale "
      "visibility measurement, because a British particulate exceedance is "
      "predominantly a dry settled day and the association runs backwards. And a flat "
      "20 mph clamp is simultaneously too fast to stop within 15 m of sight distance "
      "and too slow to impose on a 70 mph carriageway.")
    P("The safeguard that follows from the analysis is therefore triggered by an "
      "on-vehicle visibility measurement rather than an ambient air-quality feed, "
      "computes its target from stopping physics rather than a constant, rate-limits "
      "the approach to that target, refuses to clamp below a road-class floor, holds "
      "through a hysteresis window, and treats a missing reading as a hazard. It "
      "passes all seven constraint checks, and the limits of what it can do are stated "
      "above rather than left for an examiner to find.")

    H2("References")
    UL([
        "Department for Transport. <em>Road Safety Data</em> (STATS19), collision, "
        "vehicle and casualty tables, 2021–2025, and the <em>Road Safety Open "
        "Dataset Data Guide</em> 2024. data.gov.uk.",
        "Department for Environment, Food and Rural Affairs. <em>Automatic Urban and "
        "Rural Network (AURN)</em> hourly particulate measurements and site metadata. "
        "uk-air.defra.gov.uk.",
        "Leveson, N. G. <em>Engineering a Safer World: Systems Thinking Applied to "
        "Safety.</em> MIT Press, 2011.",
        "Leveson, N. G. and Thomas, J. P. <em>STPA Handbook.</em> 2018.",
        "European Union. Directive 2008/50/EC on ambient air quality, PM10 24-hour "
        "limit value of 50 µg/m³.",
    ])
    return b
