"""Part 5 validation: the STPA matrix, the offline audit of the classifier's false
negatives, and a closed-loop replay of a plume encounter."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from governor import (MPH, DECEL_CAPABILITY, REACTION_TIME, COMFORT_DECEL_LIMIT,
                      PM10_LIMIT, ROAD_CLASS_FLOOR, Telemetry, VisibilityGovernor,
                      brief_interlock, safe_speed_mph)

FIG = "results/figures"

# STATS19 records no sight distance. These are DESIGN-BASIS values used to drive
# the interlock's validation, not measurements. The statistical evidence that
# low visibility is associated with loss-of-control events is Part 2's; these
# numbers only say what the interlock should do once such a condition is present.
SIGHT_CLEAR = 500.0
SIGHT_FOG = 50.0
SIGHT_PM10 = 80.0
SIGHT_FOG_AND_PM10 = 15.0   # the condition the brief names


def sight_from_conditions(row):
    fog = bool(row.get("is_fog", False))
    pm10 = row.get("pm10", np.nan)
    exceed = (not pd.isna(pm10)) and pm10 >= PM10_LIMIT
    if fog and exceed:
        return SIGHT_FOG_AND_PM10
    if fog:
        return SIGHT_FOG
    if exceed:
        return SIGHT_PM10
    if pd.isna(pm10):
        return None          # sensor gave nothing: SC-5 territory
    return SIGHT_CLEAR


STPA_ROWS = [
    ("Not provided",
     "Plume entered, no speed restriction issued",
     "H-1 Goods vehicle strikes a vehicle or obstacle hidden by the plume",
     "SC-1 Any visibility degradation, or any missing visibility reading, must "
     "command the visibility-limited speed within one control cycle"),
    ("Provided incorrectly",
     "Restriction issued, but to a speed the sight distance does not support "
     "(the brief's flat 20 mph), or clamped into high-speed flow",
     "H-1 on a short sight distance; H-2 rear-end collision on a fast road",
     "SC-2 Target speed must be the lesser of the physics limit and the posted "
     "limit, never below the road-class floor; hazards precede the slowdown and "
     "the deceleration is rate-limited to 2.5 m/s^2"),
    ("Too late",
     "Restriction issued after the vehicle is already inside the plume",
     "H-1 The vehicle is travelling faster than its sight distance permits",
     "SC-1 trigger is the leading particulate indicator, not the visibility "
     "outcome, so the command precedes the sight-distance collapse"),
    ("Stopped too soon",
     "Restriction released while the plume still surrounds the vehicle",
     "H-1 Speed returns before sight distance does",
     "SC-4 Release requires ten consecutive seconds of confirmed-clear readings"),
]


def print_control_structure():
    print("\n\n=== PART 5: THE STAMP SOFTWARE SAFEGUARD ===\n")
    print("--- Hierarchical control structure (three tiers) ---\n")
    print("  TIER 1  Fleet Safety Management")
    print("          downward : route and dispatch policy, speed policy, ODD limits")
    print("          upward   : incident reports, interlock audit logs, exceedance trends")
    print("             |                                    ^")
    print("             v                                    |")
    print("  TIER 2  Vehicle Visibility Supervisor  (the software controller)")
    print("          downward : target-speed command, hazard-light command, advisories")
    print("          upward   : visibility and particulate readings, AI severity")
    print("                     prediction, driver override state")
    print("             |                                    ^")
    print("             v                                    |")
    print("  TIER 3  Actuators and driver")
    print("          downward : engine torque limit, service brake, hazard lamps")
    print("          upward   : wheel speed, deceleration achieved, override switch\n")

    print("--- STPA: unsafe control actions for 'Issue Visibility-Loss Speed "
          "Restriction' ---\n")
    for mode, uca, hazard, constraint in STPA_ROWS:
        print(f"  [{mode}]")
        print(f"    UCA        : {uca}")
        print(f"    Hazard     : {hazard}")
        print(f"    Constraint : {constraint}\n")


def offline_audit(fn, te):
    """Does the governor fire on the collisions the classifier called Safe?"""
    print("--- Offline audit: the interlock against the classifier's false negatives ---\n")
    print(f"  false negatives to guard : {len(fn):,}")
    print(f"  design-basis sight distances: clear {SIGHT_CLEAR:.4f} m, "
          f"fog {SIGHT_FOG:.4f} m, PM10 exceedance {SIGHT_PM10:.4f} m, "
          f"both {SIGHT_FOG_AND_PM10:.4f} m")
    print("  (STATS19 carries no sight distance; these are design-basis values, "
          "not measurements)\n")

    rows = []
    for _, r in fn.iterrows():
        sight = sight_from_conditions(r)
        tm = Telemetry(sight_distance_m=sight,
                       pm10=None if pd.isna(r.get("pm10")) else float(r["pm10"]),
                       pm10_local_mean=None, speed_mph=float(r["speed_limit"]),
                       road_class=r.get("road_class_label") or "Unclassified",
                       ai_severity_prediction="Safe")
        gov = VisibilityGovernor().step(tm)
        brf = brief_interlock(tm)
        rows.append({"collision_index": r["collision_index"],
                     "road_class": tm.road_class,
                     "posted": tm.speed_mph,
                     "sight_m": sight,
                     "pm10": tm.pm10,
                     "gov_mode": gov.mode,
                     "gov_target": gov.target_speed_mph,
                     "brief_mode": brf.mode,
                     "brief_target": brf.target_speed_mph})
    aud = pd.DataFrame(rows)
    gov_caught = int((aud.gov_mode == "RESTRICT").sum())
    brf_caught = int((aud.brief_mode == "RESTRICT").sum())
    print(f"  governor issues a restriction : {gov_caught:,} of {len(aud):,} "
          f"({100 * gov_caught / max(len(aud), 1):.4f}%)")
    print(f"  brief's interlock fires      : {brf_caught:,} of {len(aud):,} "
          f"({100 * brf_caught / max(len(aud), 1):.4f}%)")
    print("\n  The brief's rule only fires below 15 m, a condition STATS19 can only")
    print("  evidence when fog and a PM10 exceedance coincide. The governor also acts")
    print("  on a missing reading (SC-5) and on a single degraded channel (SC-1).\n")
    aud["visibility_degraded"] = aud.sight_m.isna() | (aud.sight_m < SIGHT_CLEAR)
    deg = aud[aud.visibility_degraded]
    clear = aud[~aud.visibility_degraded]
    print(f"  false negatives under degraded visibility : {len(deg):,}")
    print(f"    of which the governor restricts         : "
          f"{int((deg.gov_mode == 'RESTRICT').sum()):,}")
    print(f"  false negatives in clear conditions       : {len(clear):,}")
    print(f"    of which the governor restricts         : "
          f"{int((clear.gov_mode == 'RESTRICT').sum()):,}")
    print("")
    print("  A visibility interlock must NOT fire on a collision that had nothing to")
    print("  do with visibility. Staying nominal on the clear-condition false negatives")
    print("  is correct behaviour, not a miss: those collisions need a different")
    print("  safeguard, and this study says so rather than claiming one interlock")
    print("  covers every failure of the classifier.")
    print("")
    print("  sample of the audit, first 15 rows:")
    print(aud.head(15).to_string(index=False))
    return aud


# ---------------------------------------------------------------------------
# closed loop
# ---------------------------------------------------------------------------
class Scenario:
    def __init__(self, name, v0_mph, road_class, sight_m, obstacle_m, follower_gap_m,
                 sensor_fails=False):
        self.name = name
        self.v0 = v0_mph * MPH
        self.road_class = road_class
        self.sight = sight_m
        self.obstacle = obstacle_m
        self.gap = follower_gap_m
        self.sensor_fails = sensor_fails


def simulate(sc, controller, dt=0.05, horizon=60.0):
    """1-D two-vehicle replay inside a plume.

    Both vehicles are in the same plume, so the FOLLOWER cannot see the guarded
    vehicle's hazard lights or brake lights until the gap closes inside the sight
    distance. That is the mechanism by which an abrupt speed reduction on a fast
    road creates a rear-end hazard, and it is why SC-2 rate-limits the slowdown
    and refuses to clamp below a road-class floor.

    Returns both impact speeds in mph; 0.0 means the collision was avoided.
    """
    v = sc.v0                    # guarded vehicle speed
    x = 0.0                      # guarded vehicle position
    fv = sc.v0                   # follower speed
    fx = -sc.gap                 # follower position
    gov = controller() if callable(controller) else controller
    effective_sight = 0.0 if sc.sensor_fails else sc.sight

    seen_at = None               # when the guarded vehicle saw the obstacle
    f_react_at = None            # when the follower perceived the lead slowing
    front_impact = None
    rear_impact = None
    lead_crashed = False
    t = 0.0
    trace = []

    while t < horizon:
        rng = sc.obstacle - x
        if rng <= 0 and not lead_crashed:
            front_impact = v
            lead_crashed = True
            v = 0.0              # the lead is now a stationary obstacle

        sight = None if sc.sensor_fails else sc.sight
        tm = Telemetry(sight_distance_m=sight, pm10=120.0, pm10_local_mean=20.0,
                       speed_mph=v / MPH, road_class=sc.road_class,
                       ai_severity_prediction="Safe")
        if gov is None:
            target_mph, hazards = v / MPH, False
        elif isinstance(gov, VisibilityGovernor):
            cmd = gov.step(tm, dt=dt)
            target_mph, hazards = cmd.target_speed_mph, cmd.hazard_lights
        else:
            cmd = brief_interlock(tm)
            target_mph, hazards = cmd.target_speed_mph, cmd.hazard_lights

        a = 0.0
        if not lead_crashed:
            if seen_at is None and rng <= effective_sight:
                seen_at = t
            if seen_at is not None and (t - seen_at) >= REACTION_TIME:
                a = -DECEL_CAPABILITY          # emergency stop
            else:
                target = target_mph * MPH
                if target < v:
                    a = -min(COMFORT_DECEL_LIMIT, (v - target) / dt)
            v = max(v + a * dt, 0.0)
            x += v * dt

        # --- the follower, blind beyond its own sight distance ---------------
        gap = x - fx
        lead_slowing = a < -0.3 or lead_crashed
        if f_react_at is None and gap <= effective_sight and (lead_slowing or hazards):
            f_react_at = t
        f_braking = f_react_at is not None and (t - f_react_at) >= REACTION_TIME
        fa = -7.0 if f_braking else 0.0        # car, dry sound surface, ABS
        fv = max(fv + fa * dt, 0.0)
        fx += fv * dt

        if (x - fx) <= 0 and rear_impact is None:
            closing = fv - v
            if closing > 0:
                rear_impact = closing
                break
        if lead_crashed and fv <= 0.0:
            break
        trace.append((t, v / MPH, fv / MPH, target_mph, rng, x - fx))
        t += dt

    return {"front_impact_mph": 0.0 if front_impact is None else front_impact / MPH,
            "rear_impact_mph": 0.0 if rear_impact is None else rear_impact / MPH,
            "trace": trace}


def closed_loop():
    print("\n--- Closed-loop validation: a plume encounter, tick by tick (dt = 0.05 s) ---\n")
    scenarios = [
        Scenario("Trunk road, dense plume", 60, "A", 50, 220, 40),
        Scenario("Motorway, dense plume", 70, "Motorway", 50, 300, 50),
        Scenario("Motorway, brief's 12 m trigger", 70, "Motorway", 12, 300, 50),
        Scenario("B road, 12 m plume", 40, "B", 12, 120, 30),
        Scenario("Trunk road, sensor returns nothing", 60, "A", 50, 220, 40,
                 sensor_fails=True),
    ]
    controllers = {
        "unguarded": None,
        "brief 20 mph clamp": "brief",
        "governor": lambda: VisibilityGovernor(),
        "governor, SC-2 off": lambda: VisibilityGovernor(disabled={"SC-2"}),
        "governor, SC-1 off": lambda: VisibilityGovernor(disabled={"SC-1"}),
    }
    print(f"{'scenario':<34}{'controller':<22}{'front impact':>14}{'rear impact':>14}")
    results = {}
    for sc in scenarios:
        for cname, c in controllers.items():
            res = simulate(sc, c)
            results[(sc.name, cname)] = res
            print(f"{sc.name:<34}{cname:<22}"
                  f"{res['front_impact_mph']:>14.4f}{res['rear_impact_mph']:>14.4f}")
        print()
    return scenarios, results


def pass_fail(results, scenarios, aud):
    print("\n--- Constraint checks ---\n")
    checks = []

    g_front = [results[(s.name, "governor")]["front_impact_mph"] for s in scenarios]
    u_front = [results[(s.name, "unguarded")]["front_impact_mph"] for s in scenarios]
    checks.append(("The governor never increases a front impact speed",
                   all(g <= u + 1e-9 for g, u in zip(g_front, u_front))))

    b_rear = [results[(s.name, "brief 20 mph clamp")]["rear_impact_mph"] for s in scenarios]
    g_rear = [results[(s.name, "governor")]["rear_impact_mph"] for s in scenarios]
    checks.append(("The governor's rear-end exposure never exceeds the brief's",
                   all(g <= b + 1e-9 for g, b in zip(g_rear, b_rear))))

    sensor = [s for s in scenarios if s.sensor_fails][0]
    checks.append(("SC-5 acts on a missing reading where the brief's rule is silent",
                   results[(sensor.name, "governor")]["front_impact_mph"]
                   <= results[(sensor.name, "brief 20 mph clamp")]["front_impact_mph"]))

    checks.append(("Removing SC-1 degrades protection (ablation is not inert)",
                   any(results[(s.name, "governor, SC-1 off")]["front_impact_mph"]
                       > results[(s.name, "governor")]["front_impact_mph"] + 1e-9
                       for s in scenarios)))

    deg = aud[aud.visibility_degraded]
    checks.append(("The governor restricts on every false negative that occurred "
                   "under degraded visibility",
                   bool(len(deg) and (deg.gov_mode == "RESTRICT").all())))
    clear = aud[~aud.visibility_degraded]
    checks.append(("The governor stays nominal on clear-condition false negatives "
                   "(no spurious restriction)",
                   bool((clear.gov_mode == "NOMINAL").all())))

    checks.append((f"20 mph exceeds the safe speed at 15 m "
                   f"({safe_speed_mph(15):.4f} mph), so the brief's clamp is unsafe",
                   safe_speed_mph(15) < 20.0))

    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {desc}")
    n_pass = sum(1 for _, ok in checks if ok)
    print(f"\n  {n_pass} of {len(checks)} checks pass")
    return checks


def plot_physics():
    d = np.linspace(5, 250, 400)
    v = [safe_speed_mph(x) for x in d]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(d, v, lw=2, color="#1f4e79", label="speed from which the vehicle can stop")
    ax.axhline(20, ls="--", lw=1.1, color="#9c3d1e",
               label="the brief's flat 20 mph clamp")
    ax.axvline(15, ls=":", lw=1.1, color="#555", label="the brief's 15 m trigger")
    cross = next(x for x in d if safe_speed_mph(x) >= 20.0)
    ax.plot([cross], [20], "o", ms=5, color="#9c3d1e")
    ax.annotate(f"20 mph only becomes safe\nbeyond {cross:.1f} m",
                xy=(cross, 20), xytext=(cross + 25, 11), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=.8, color="#555"))
    ax.set_xlabel("Sight distance (m)")
    ax.set_ylabel("Safe speed (mph)")
    ax.set_title("A flat speed clamp cannot track sight distance\n"
                 "laden goods vehicle, 4.0 m/s², 1.5 s reaction",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(lw=.3, alpha=.4)
    fig.tight_layout()
    fig.savefig(f"{FIG}/safe_speed_vs_sight.png", dpi=300)
    fig.savefig(f"{FIG}/safe_speed_vs_sight.pdf")
    plt.close(fig)
    print(f"\nwrote {FIG}/safe_speed_vs_sight.png and .pdf")
    return cross
