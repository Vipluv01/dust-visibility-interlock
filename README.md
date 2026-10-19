# Sudden Environmental Visibility Loss via Particulate Dust Plumes

**Capstone Project 10** · Vipluv · 22MF3IM22
Centre of Excellence on Safety Engineering & Analytics (CoE-SEA), IIT Kharagpur

A hybrid statistical, machine-learning and STAMP software-interlock study of
goods-vehicle risk on the Great Britain road network, 2021–2025, built on **real
open data** — 513,801 STATS19 collisions joined to hourly particulate
measurements from 87 DEFRA AURN monitoring sites.

**Submit:** `Dust-Visibility-Interlock.pdf` and `Dust-Visibility-Interlock.docx` (15 pages, A4).

---

## What the study found

1. **The hazard is real.** Recorded fog or mist roughly doubles the odds of a
   loss-of-control event for a goods vehicle (χ² = 31.9761, OR = 1.9914) and of a
   fatal outcome (χ² = 7.5918, OR = 2.0474).

2. **The proposed sensing channel is not.** A regional PM10 exceedance is
   significantly associated with *fewer* loss-of-control events (OR = 0.7690),
   because a British particulate exceedance is predominantly a dry, settled,
   anticyclonic day — 88.5214% of its collisions are on dry roads against
   74.4651% otherwise, daily PM10 correlates with the wet-surface share at
   r = −0.2806, and 48.6792% of exceedance collisions fall in March alone
   (spring Saharan dust intrusion). Loss of control needs a wet or icy surface.

3. **Pooling the two channels reports a false null.** The composite exposure the
   brief implies gives χ² = 3.1020, below the 3.8415 critical value, because two
   real effects with opposite signs cancel. Table 6 in the report exists to show
   that trap.

4. **The classifier is blindest where the hazard is worst.** A random forest
   (macro F1 = 0.4215) calls 141 of 446 fatal collisions "Safe" — a 31.6143%
   false-negative rate — and fog is 3.5845× over-represented among those misses.

5. **The brief's 20 mph clamp is unsafe in two directions.** At 15 m sight
   distance a laden goods vehicle can only stop from 14.5177 mph, so 20 mph
   cannot stop inside what it can see; 20 mph only becomes safe beyond 23.4211 m.
   And clamping to 20 mph on a 70 mph carriageway manufactures the rear-end
   hazard it was written to prevent.

The replacement governor enforces five constraints and passes **7 of 7** checks.

## Files

| File | What it is |
|---|---|
| `Dust-Visibility-Interlock.pdf` / `.docx` | The report: Parts 1–5, limitations, references |
| `run_output.txt` | Full output of the run the report quotes. **Every number in the report comes from this file** |
| `run_capstone.py` | Orchestrates Parts 2–5 and writes `results/summary.json` |
| `analysis.py` | Parts 2–4: association tables, PCA, Poisson GLM, classifier, false-negative audit |
| `governor.py` | Part 5: the safety interlock, its constants, and the brief's rule for comparison |
| `validation.py` | Part 5: STPA matrix, offline audit, closed-loop replay, constraint checks |
| `scripts/build_dataset.py` | Part 1: ingestion, quality repair, nearest-site particulate join |
| `scripts/fetch_aurn.py` | Downloads AURN hourly data and reduces it to a daily panel |
| `scripts/make_diagrams.py` | The control-structure and data-flow figures |
| `writeup/content.py` | The report, written once as a block list |
| `writeup/build.py` | Renders that block list to HTML and to `.docx` |
| `writeup/cdp_print.py` | Renders the HTML to PDF via headless Brave |
| `results/summary.json` | Machine-readable headline numbers, consumed by the report |
| `data/aurn_daily_pm.csv` | 157,402 site-days of daily mean PM10/PM2.5 (committed) |

## Reproducing it

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

Two inputs are not committed and must be fetched first.

**STATS19** (~272 MB) — three tables covering 2021–2025:

```bash
mkdir -p data_raw/stats19 && cd data_raw/stats19
for t in collision vehicle casualty; do
  curl -sS -o "${t}-last5.csv" \
    "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-${t}-last-5-years.csv"
done
```

Note that `HEAD` against that host returns a 25-byte stub, so file sizes have to
be checked with a ranged `GET`. The per-year filenames only exist for the most
recent years; `last-5-years` is the reliable source and currently covers
2021–2025.

**AURN site metadata**, then the particulate data itself:

```bash
mkdir -p data_raw/aurn
curl -sS -o data_raw/aurn/AURN_metadata.RData \
  "https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData"
.venv/bin/python scripts/fetch_aurn.py     # 435 site-years, ~485 MB, a few minutes
```

The working per-site path is `uk-air.defra.gov.uk/datastore/data_files/site_data/<SITE>_<YEAR>.csv`;
the shorter `/data_files/` path 404s.

Then:

```bash
.venv/bin/python scripts/build_dataset.py          # Part 1 -> data/collision_panel.csv
.venv/bin/python scripts/make_diagrams.py
.venv/bin/python run_capstone.py | tee run_output.txt
.venv/bin/python writeup/build.py                  # -> report.html + .docx
```

## Rebuilding the PDF

`--print-to-pdf` hangs on this machine's Brave and writes nothing, so the PDF is
printed by driving headless Brave over the DevTools protocol instead.

```bash
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" --headless=new --disable-gpu \
  --no-first-run --disable-background-networking --disable-component-update --disable-sync \
  --user-data-dir=/tmp/brave-dust --remote-debugging-port=9333 about:blank &
.venv/bin/python writeup/cdp_print.py
pkill -f "user-data-dir=/tmp/brave-dust"
```

## Caveats

Read these before quoting any figure.

- **STATS19 records no sight distance and no vehicle telemetry.** The sight
  distances driving the interlock's validation are *design-basis* values, not
  measurements. The statistical evidence in Part 2 is real; the metre values in
  Part 5 state what the interlock should do once a condition exists.
- **There are no near-miss data.** STATS19 is a register of reported injury
  collisions, so the brief's near-miss alerts cannot be tested. Loss of control
  (`skidding_and_overturning`) is used as the closest recorded analogue and is
  named as such throughout.
- **The particulate join is station-scale**, median 11.7992 km from collision to
  site. A network at that spacing cannot resolve a micro-scale plume — which is
  part of the finding, not only a weakness.
- **SC-2's road-class floor is not always favourable.** It cuts rear impact by
  5.2317 and 7.4025 mph in the 12 m scenarios, but in the 50 m motorway scenario
  it *causes* a 25.6836 mph front impact where 3.3121 mph was achievable without
  it, because there the follower could already see. A correct refinement needs a
  rear-facing range sensor this design does not assume.
- **The classifier is weak in absolute terms** and would not be deployed. The
  audit characterises how such a model fails; it does not present it as fit for use.
- Fog is significant at collision level but **not** at day level (p = 0.8560).
  That is the study's central methodological point, not a contradiction: a
  micro-scale exposure does not survive aggregation to a calendar day.
