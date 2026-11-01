# Project spec — Does perp flow and positioning improve 15-minute BTC spot prediction?

## 1. Overview and research question

**Rubric:** problématique and contextualisation (Démarche scientifique); sets the thread every later section must serve.

**Research question:** Does perpetual futures flow and positioning improve short-horizon BTC spot price prediction over a spot-only baseline?

**Scope (final, after supervisor feedback):** one coin, **BTC** (BTCUSDT spot vs the BTCUSDT USD-M perpetual on Binance), and one forecast horizon, **15 minutes**. Depth goes into two places instead of breadth:

- **Descriptive analysis:** an extensive look at how BTC spot and perp interact before any model (Section 7).
- **A coin- and horizon-agnostic pipeline:** symbol and horizon are configuration parameters, so the same code runs on any Binance pair and any horizon without edits (Sections 4 and 11). Other coins or horizons are not part of the graded analysis.

**Motivation:** in crypto, a large share of price discovery is thought to happen on perpetual futures rather than spot. Perps also carry information spot cannot: leveraged positioning (open interest), the cost of holding positions (funding), and the perp-spot premium (basis). If that information reaches spot with a delay, it should improve forecasts.

**Hypotheses:**

1. H1: adding perp features to a spot-only model improves out-of-sample forecasts of the 15-minute BTC spot return.
2. H2: perp-to-spot lead-lag is visible in the descriptive analysis (cross-correlations, Granger tests) and fades within the 15-minute window.

**What a good answer looks like:** a well-documented descriptive picture of BTC spot-perp dynamics, then a clear, tested statement on whether perp features add forecasting power at 15 minutes. A null result is a valid finding if the design is sound.

## 2. Grading, deliverables and rubric cross-reference

Two graded deliverables, per the instructor's email: the **repo + PDF report**, due Nov 24, 2026 at 23:59 Paris time, worth roughly 2/3 to 3/4 of the grade; and the **defense** on Dec 1, 2026, worth the rest (exact split not yet fixed).

| Deliverable | What is required | Spec sections |
| --- | --- | --- |
| Code repository (any Git platform, public) | All code needed; runs as-is when cloned by the grader (Fernanda); downloading reasonably sized data from inside the code is fine; heavy loads discouraged | 3, 4, 11 |
| PDF report in the repo | Goal and scope, methods, results | 1, 5–10 |
| Submission | Link to the public repo sent by email | 13 |
| Defense | At most 10 min of slides, uninterrupted; then free Q&A on methods, results and extensions, including tests of our understanding of our own code | 10, 13 |

The [course evaluation page](https://pythonds.linogaliana.fr/content/annexes/evaluation.html) categories remain our quality checklist for what a good repo and report contain:

| Course category | Spec sections that earn it |
| --- | --- |
| Données (collecte et nettoyage) | 3 Data sources, 4 Cleaning and storage, 6 Features |
| Analyse descriptive | 7 Descriptive analysis, 10 Visualisation |
| Analyse et modelling | 8 Models, 9 Training and testing |
| Démarche scientifique et reproductibilité | 1 Question, 5 Horizons, 9 Testing, 11 Reproducibility, 12 Limitations |
| Format du code | 11 Repo structure and Git workflow |
| Soutenance | 10 Visualisation (slides), 13 Timeline |

**Hard requirements:**

- [ ] Public repo with frequent commits and pull requests, not a single upload.
- [ ] A fresh clone runs end to end with no manual steps beyond installing dependencies; data downloads happen inside the code.
- [ ] Download size kept reasonable (Section 3); no data files committed to Git.
- [ ] `report.pdf` committed in the repo: goal and scope, methods, results.
- [ ] Dependencies installable from `pyproject.toml` or `requirements.txt`.
- [ ] Clean code: functions in `.py` modules, no copy-pasted cells.
- [ ] Every table and figure in the report is commented and tied back to the question.
- [ ] Every team member can explain any part of the code at the defense. AI use is fine; AI use without understanding is penalised.

**Staying on topic:** each section below opens with a **Rubric** line. If a planned analysis cannot name both a rubric category and a hypothesis (H1 or H2) it serves, cut it.

## 3. Data sources and collection

**Rubric:** Données (collecte) — automated, multi-dataset collection from two access routes (static archive + REST API).

All data comes from Binance public endpoints, downloaded by the code itself, for **BTCUSDT spot** and the **BTCUSDT USD-M perpetual**. No API key is needed. The symbol is read from `config.yaml`, so swapping in another pair (e.g. ETHUSDT) requires no code change.

**Design choice: 1-minute klines instead of tick data.** Binance klines already contain the taker-buy volume per minute, so signed flow and VWAP can be computed without downloading every trade. This keeps a fresh clone fast and light, as the grader requires.

$$
\text{signed volume} = 2 \times \text{taker buy volume} - \text{volume} \qquad \mathrm{VWAP} = \frac{\text{quote volume}}{\text{volume}}
$$

**Sample window (proposal):** 2026-01-01 to 2026-06-30, about 6 months. Confirm both series cover the full window before fixing it; hardcode the dates in `config.yaml` so results never shift between runs.

| Dataset | Market | Route | Resolution | Used for |
| --- | --- | --- | --- | --- |
| `klines` 1m | Spot | Archive | 1 min | Spot VWAP returns (target), spot signed flow, volume, trade count |
| `klines` 1m | USD-M perp | Archive | 1 min | Perp VWAP, perp signed flow |
| `premiumIndexKlines` 1m | USD-M perp | Archive | 1 min | Basis / premium |
| `metrics` | USD-M perp | Archive | ≈5 min (verify) | Open interest, long/short and taker ratios |
| `liquidationSnapshot` | USD-M perp | Archive | Event (verify completeness) | Liquidation flow |
| `fundingRate` | USD-M perp | REST API `/fapi/v1/fundingRate` | 8 h (verify per symbol) | Funding level and changes |
| First candle / exchange info | Spot + perp | REST API | — | Availability check |
| `aggTrades`, a few days of BTC only | Spot + perp | Archive | Tick | Optional second-level lead-lag figure (Section 7) |

**Download size:** klines are one small monthly file per symbol and market, so the full set should stay modest. Measure the actual total after the first download and record it in the README. If it is heavier than expected, shorten the window or add a `--quick` mode (one month) for a fast check.

**Routes:**

- **Archive** (`data.binance.vision`): static zipped CSVs with a `.CHECKSUM` per file. Bulk history, no rate limits, identical files on every download.
- **REST API** (`api.binance.com`, `fapi.binance.com`): funding history and availability checks. Responses are cached locally. Open-interest and ratio statistics via REST only cover the latest 30 days, so historical positioning comes from the archive `metrics` files.

**Collection steps (all run by `python -m src.pipeline`):**

1. Availability check: earliest spot and perp candle for the configured symbol (REST); output a table for the report.
2. Build archive URLs from the config (market, dataset, symbol, month); download, verify checksums, unzip into the gitignored `data/` folder; skip files already present.
3. Fetch funding history per symbol with pagination (REST); cache raw JSON.
4. Log every download (file, size, checksum status) so provenance is documented.

## 4. Data cleaning, structuring and storage

**Rubric:** Données (nettoyage) and reproductibilité — documented cleaning choices, raw → clean pipeline, data stored outside Git.

The pipeline has three layers, each saved as parquet in the gitignored `data/` folder so later steps never re-download:

1. **Raw:** files exactly as downloaded (zipped CSV, raw JSON from the API).
2. **Clean:** typed, deduplicated, one schema per dataset, timestamps in UTC milliseconds.
3. **Panel:** one 1-minute table per symbol, spot and perp aligned on the same timestamp index.

**Cleaning tasks (document each choice in the report):**

- **Headers and units:** archive files have inconsistent headers across markets and periods; spot timestamps switch from milliseconds to microseconds on 2025-01-01. Normalise to one unit.
- **Gaps:** flag missing minutes (exchange maintenance, rare quiet minutes); keep a gap mask rather than silently forward-filling; drop windows where spot or perp is missing.
- **Derived columns:** per minute and market — VWAP (quote volume / volume), signed volume (2 × taker buy volume − volume), trade count.
- **Lower-frequency joins:** `metrics` (≈5 min) and funding (8 h) are joined to the 1-min grid **as of the last value known before each minute** (backward as-of join), never forward. This is the main leakage guard at the data stage.
- **Outliers:** inspect extreme returns (fat-finger prints, listing effects); winsorise features if needed, never the target.
- **Validation:** spot and perp close prices should track each other closely; large persistent gaps flag a join or unit bug.

**Storage:**

- No data in Git: `data/` is in `.gitignore`; the code regenerates everything from public Binance URLs.
- Local parquet cache, so re-running the notebook after the first download takes seconds.
- Tools: pandas throughout (1-min data for one symbol is small); DuckDB or polars only if a step turns out slow.

**Coin- and horizon-agnostic design:**

- `config.yaml` holds `symbol`, `horizon_minutes`, the sample window and feature lags; nothing in `src/` hardcodes BTC or 15.
- Every pipeline function takes the symbol and horizon as arguments (e.g. `build_panel(symbol)`, `make_target(panel, horizon)`), and output files are named by both, so runs never overwrite each other.
- Feature lags and windows that depend on the horizon are derived from it rather than fixed.
- A smoke test runs the whole pipeline in quick mode on a different symbol and horizon (e.g. ETHUSDT at 5 minutes) to prove the claim; its results are not part of the analysis.

## 5. Time horizon and prediction target

**Rubric:** Démarche scientifique — justified horizon choice, target defined to avoid microstructure artefacts.

All data sits on a **1-minute base grid**; we forecast at one horizon, **15 minutes** (`horizon_minutes: 15` in the config).

| Quantity | Value (6-month window) |
| --- | --- |
| Forecasts issued (one per minute) | ≈ 260,000 |
| Non-overlapping 15-min windows | ≈ 17,000 |
| Perp features usable | Signed flow, basis, open interest and ratios (`metrics`, ≈5 min), funding (slow-moving context) |

Forecasts are issued every minute, so consecutive labels overlap by 14 minutes (handled in Section 9).

**Why 15 minutes:** long enough to sit above trade-level microstructure noise and to use the ≈5-minute positioning data, short enough that the derivatives-lead-spot mechanism is still plausible, and it leaves a large sample. The descriptive analysis still looks at lags from 1 to 30 minutes to show where the lead-lag lives (H2).

**Why not seconds:** the spot archive has no historical best bid/ask, so the target is trade-based. At second scale, trades bounce between bid and ask, creating spurious negative autocorrelation. Spot-perp arbitrage also closes gaps within milliseconds. An optional BTC-only second-level cross-correlation (Section 7) shows this visually.

**Why not days:** too few observations to detect small effects, and the derivatives-lead-spot mechanism is short-horizon.

**Targets:** the forward spot log return over horizon *h*, computed from 1-min VWAPs.

$$
y_{t,h} = \log \mathrm{VWAP}^{spot}_{t+h} - \log \mathrm{VWAP}^{spot}_{t}
$$

- **Regression target:** the return itself.
- **Classification target:** three classes — up / flat / down — with a threshold (e.g. a fixed fraction of the rolling volatility), so "flat" absorbs near-zero noise. Fix the threshold on training data only.

## 6. Features

**Rubric:** Données (variables créées) and Démarche scientifique — the two feature blocks *are* the experiment for H1.

Features are split into two blocks. The baseline model uses block A only; the full model uses A + B. Every feature at time *t* uses data strictly before *t*.

| Block | Feature | Definition (computed over lags of 1, 5, 15, 60 min) | Source |
| --- | --- | --- | --- |
| A — spot only | Lagged returns | Past spot VWAP log returns | Spot klines |
| A — spot only | Spot signed flow | Sum of signed volume / total volume | Spot klines |
| A — spot only | Realised volatility | Rolling std of 1-min returns | Spot klines |
| A — spot only | Activity | Trade count and volume vs. rolling average | Spot klines |
| A — spot only | Time of day | Hour-of-day encoding (crypto trades 24/7 but activity is cyclical) | Timestamp |
| B — perp | Perp signed flow | Signed volume / total volume on the perp | Perp klines |
| B — perp | Flow divergence | Perp signed flow minus spot signed flow | Both |
| B — perp | Basis | (Perp VWAP − spot VWAP) / spot VWAP, in bps; and its change | Both / premium index |
| B — perp | Perp lead return | Past perp return minus past spot return | Both |
| B — perp | Open interest change | % change in open interest since last reading | `metrics` |
| B — perp | Taker / long-short ratios | Latest available ratios | `metrics` |
| B — perp | Liquidations | Liquidated notional by side | `liquidationSnapshot` |
| B — perp | Funding | Last settled funding rate | Funding history |

**Rules:**

- **Scale-free units:** basis in bps, flows as fractions of volume, open interest as % changes. This keeps features stable over the sample and portable to other symbols.
- **Respect resolution:** positioning features (≈5 min) are usable at the 15-minute horizon; funding (8 h) is slow context, not a short-term signal.
- **Horizon-aware lags:** lag sets are derived from `horizon_minutes`, so a different horizon gets sensible features without code changes.
- **Feature count discipline:** start with about 10–15 features per block, justify each, add more only if a hypothesis needs it.

## 7. Descriptive analysis

**Rubric:** Analyse descriptive (4 pts) — every statistic is interpreted and linked to H1 or H2; no uncommented `describe()`.

This is the centrepiece of the project. With one coin and one horizon, the descriptive part goes deep: it must give a first answer before any model (*is there visible lead-lag from perp to spot, and how long does it last?*) and justify every modelling choice that follows. It is organised in five blocks.

| Block | Analysis | What it shows | Serves |
| --- | --- | --- | --- |
| A. Data quality | Coverage: missing minutes per day, maintenance windows, gap mask | Which periods are usable; why some windows are dropped | Cleaning choices |
| A. Data quality | Spot vs perp price tracking (close-to-close difference) | Join and unit checks; sets expectations for basis size | Cleaning choices |
| B. Market structure | Perp vs spot volume share and trade counts over time | How much activity happens on the perp | Motivation |
| B. Market structure | Intraday and weekday seasonality of volume, volatility and flow | Activity cycles; justifies time-of-day features | Features |
| C. Returns and volatility | Return distributions at 1 and 15 minutes vs a normal (moments, QQ plot) | Fat tails; why a flat class and robust metrics | Target design |
| C. Returns and volatility | Autocorrelation of returns and of absolute returns | Little linear predictability; strong volatility clustering | Baseline |
| C. Returns and volatility | Stationarity tests (ADF/KPSS) on returns, basis, flows | Which series can enter models as levels vs changes | Features |
| D. Perp-specific signals | Basis over time and its distribution; persistence (half-life) | How far and how long perp deviates from spot | H1 |
| D. Perp-specific signals | Funding rate and open interest over time | Leverage build-up phases and their link to later price moves | H1 |
| D. Perp-specific signals | Spot vs perp signed flow: correlation and divergence | Whether the two markets trade in the same direction | H1 |
| D. Perp-specific signals | Feature correlation matrix (blocks A and B) | Redundancy between perp and spot features; multicollinearity risk | Models |
| E. Lead-lag | Cross-correlation of perp flow, basis and perp returns vs future spot returns, lags −30 to +30 min | Direction and length of the lead-lag; does it fade within 15 min? | H1, H2 |
| E. Lead-lag | Granger causality tests, both directions | Does perp predict spot, spot predict perp, or both? | H1, H2 |
| E. Lead-lag | Event study around large liquidation clusters (spot returns and basis before/after) | How quickly forced perp flow shows up in spot | H2 |
| E. Lead-lag | Optional: second-level cross-correlation on a few days of BTC trades | Where the fastest lead-lag lives; justifies the minute grid | H2 |

**Interpretation standard:** each item ends with one or two sentences: what we see, what it implies for H1 or H2 or for a modelling choice, and any caveat (e.g. Granger causality is predictive, not causal). The block-E results are summarised in a short "descriptive verdict" before the models, so the reader knows what to expect from Section 9.

## 8. Models

**Rubric:** Analyse et modelling — models chosen to answer the question, justified, interpreted; not a leaderboard.

The core design is an **ablation**: each model is fitted twice, on block A (spot-only baseline) and on blocks A + B (with perp features). The gain from A to A + B is the answer to H1.

| Level | Model | Library | Why it's here |
| --- | --- | --- | --- |
| 0 | Zero-return forecast (random walk); majority class | numpy | The bar to beat; short-horizon returns are close to unpredictable |
| 1 (core) | Ridge / lasso regression on returns; multinomial logistic regression on up/flat/down | scikit-learn, statsmodels | Interpretable coefficients and HAC inference: says *which* perp variables matter |
| 2 | Gradient boosting (`HistGradientBoostingRegressor` / `Classifier`, or LightGBM) | scikit-learn / lightgbm | Checks for non-linear effects and interactions (e.g. basis only matters when funding is extreme) |
| 3 (optional) | HAR-RV on realised volatility, with and without perp features | statsmodels | Secondary question: do OI and funding help forecast volatility? Reuses the GARCH/ARCH idea where it fits |

**Scope of the fits:**

- One specification: BTC, 15-minute horizon, blocks A vs A + B, for each model level.
- The fitting code takes the symbol and horizon from the config, so the same models can be re-run elsewhere later; those runs are out of scope for the report.

**Deliberately excluded (state this in the report):** polynomial regression (overfits non-stationary prices, no motivation here) and deep learning (LSTM, TLOB-style transformers): hard to justify at this data scale and would turn the project into a benchmark race.

**Interpretation outputs:** coefficient tables with HAC standard errors (level 1); permutation importance of block-B features (level 2). Treat importances as descriptive, not causal.

## 9. Training, validation and statistical testing

**Rubric:** Démarche scientifique — leakage-free validation and significance tests turn "the score went up" into a defensible claim.

**Walk-forward validation (no random splits):**

1. Split the 6-month window into monthly blocks.
2. Fold *k*: train on all months before month *k*, test on month *k* (expanding window). First test month = month 3, giving 4 test folds.
3. Tune hyperparameters (ridge/lasso penalty, boosting depth and learning rate) on the last month of each training window only.
4. Leave an **embargo** gap of at least *h* between train and test so overlapping labels can't leak.
5. Fit scalers, class thresholds and feature selection on training data only.

**Metrics (computed on each test fold, then averaged with fold-to-fold spread):**

- Regression: out-of-sample R² relative to the zero-return forecast (negative = worse than doing nothing).
- Classification: balanced accuracy, macro F1 and log loss (accuracy alone is misleading when "flat" dominates).

**Significance of the perp gain:**

- **Clark-West test:** the standard test for nested models (A is nested in A + B). This is the main test for H1.
- **Diebold-Mariano test:** for non-nested comparisons (e.g. linear vs boosting).
- **HAC (Newey-West) standard errors** for coefficients, since returns and overlapping 15-minute labels are autocorrelated.
- **Multiple testing:** with one coin and one horizon the number of tests is small; apply a Holm correction across model levels.

**Optional economic check:** a simple long/short rule on the forecasts, net of trading fees and spread (funding is negligible at these horizons unless a position is held through a settlement). Only include it if time allows and costs are modelled honestly.

## 10. Visualisation

**Rubric:** Analyse descriptive (graphical part) and Soutenance — the right chart type for each message, titled axes and legends, every figure commented.

The report keeps 12 figures, weighted towards the descriptive analysis; extras go to an appendix. The four marked ★ carry the oral presentation. Each figure is produced by one function in `plots.py` that takes the symbol and horizon from the config.

| # | Figure | Chart type | Message |
| --- | --- | --- | --- |
| 1 | Data coverage | Heatmap (day × hour, missing minutes) | Data quality and gaps |
| 2 | Perp vs spot volume share over time | Line or area chart | How much activity happens on the perp |
| 3 | Intraday seasonality of volume and volatility | Line chart by hour of day | Activity cycles behind the time-of-day features |
| 4 | Return distributions at 1 and 15 minutes | Histogram with normal overlay + QQ plot | Fat tails justify the three-class target |
| 5 | Autocorrelation of returns vs absolute returns | ACF bar charts | Weak linear predictability, strong volatility clustering |
| 6 | Basis over time, liquidation clusters marked | Line chart | Size and timing of perp-spot deviations |
| 7 | Funding rate and open interest over time | Two aligned line panels | Leverage build-up phases |
| 8 ★ | Cross-correlation of perp signals vs future spot returns, lags −30 to +30 min | Line chart with confidence band | Lead-lag exists and how fast it fades (H2) |
| 9 ★ | Event study around liquidation clusters | Mean spot-return path with confidence band | How quickly forced perp flow reaches spot (H2) |
| 10 | Feature correlation matrix, blocks A and B | Heatmap | Redundancy between spot and perp features |
| 11 ★ | Forecast gain of A + B over A per walk-forward fold, with Clark-West p-values | Dot plot with fold labels | Main result (H1) |
| 12 ★ | Coefficients / permutation importance of perp features | Horizontal bar chart with HAC intervals | Which perp signals drive the gain (H1) |

**Conventions:**

- One colour for spot and one for perp, fixed across every figure; liquidation clusters always marked the same way; colour is never the only carrier of meaning.
- Units on every axis (bps, %, minutes); dates in UTC.
- Libraries: matplotlib/seaborn for static figures; plotly for interactive ones in the notebook.
- Optional: a small Streamlit app to browse the descriptive figures; only if the core notebook is finished (a bonus, not a requirement).

**Defense slides (10 minutes, no interruptions):** about 8 slides, roughly one minute each.

1. The question and why perps might lead spot.
2. Data and pipeline: BTC spot and perp klines, one-command run, coin- and horizon-agnostic config.
3. Descriptive: market structure and basis (figures 2 and 6).
4. Descriptive: lead-lag and the liquidation event study (figures 8 and 9).
5. Method: the ablation, walk-forward validation, Clark-West test.
6. Results: figures 11 and 12 (H1).
7. Limitations and next steps: other coins and horizons are one config change away.
8. Backup slides (not presented): feature definitions, leakage checks, extra descriptive figures, to answer Q&A quickly.

Rehearse to 9 minutes; the Q&A will probe methods, extensions and our own code.

## 11. Reproducibility, repo structure and Git workflow

**Rubric:** the repo + report deliverable (about 2/3 to 3/4 of the grade) — Démarche scientifique et reproductibilité and Format du code. The grader clones the repo and runs it; anything that breaks there costs the most.

**Repo layout:**

```
project/
├── README.md            # question, one-command run, expected download size and runtime
├── report.pdf           # the graded report: goal and scope, methods, results
├── spec.md              # this project spec (exported from the shared doc)
├── pyproject.toml       # pinned dependencies (uv or pip)
├── config.yaml          # symbol, horizon_minutes, dates, lags, paths
├── .gitignore           # data/, caches, .env
├── src/
│   ├── pipeline.py      # entry point: fetch → clean → features → models → figures
│   ├── fetch.py         # archive download, checksums, REST calls + cache
│   ├── clean.py         # raw → clean → 1-min panel
│   ├── features.py      # blocks A and B, as-of joins
│   ├── models.py        # ablation fits (block A vs A + B)
│   ├── evaluate.py      # walk-forward, metrics, Clark-West, DM
│   └── plots.py         # one function per figure
├── notebooks/
│   └── analysis.ipynb   # runs top to bottom, imports src/
├── report/
│   └── report.tex       # LaTex source rendered to report.pdf
├── slides/              # defense deck
└── tests/               # small checks: no-lookahead, timestamp units
```

**Reproducibility rules:**

- One command after installing dependencies (`python -m src.pipeline`) downloads the data from public Binance URLs, builds every table and figure, and saves them for the report.
- Fixed sample window and random seeds in `config.yaml`; a `--quick` flag (one month) for a fast check; a smoke test on another symbol and horizon confirms the pipeline is truly agnostic (Section 4).
- README states the expected download size and runtime, measured on a fresh clone.
- No secrets needed (public endpoints). If a key is ever added, it goes in `.env`, never in code.
- A no-lookahead unit test: changing data after *t* must never change features at *t*.
- The report is written in Quarto from the same saved figures, then rendered to `report.pdf` and committed, so text and code agree.
- Before Nov 24: fresh clone on a different machine, fresh environment, run everything.

**Git workflow:**

- One branch per task, merged through pull requests reviewed by a teammate.
- Small, frequent commits with clear messages.
- Only one person edits the notebook at a time; strip notebook outputs before committing (e.g. `nb-clean`) to avoid merge conflicts.
- Reviews double as defense prep: each reviewer must be able to explain the code they approve.

## 12. Risks, limitations and fallbacks

**Rubric:** Démarche scientifique — naming limits honestly is part of a scientific approach; fallbacks protect the reproducibility grade.

| Risk | Impact | Mitigation / fallback |
| --- | --- | --- |
| Repo fails when the grader clones and runs it | Most of the grade lost | One-command pipeline, README with exact steps, fresh-clone test on another machine before Nov 24 |
| Download too heavy for the grader | Run abandoned or penalised | 1-min klines instead of tick data; measure size early; `--quick` mode; shorten window if needed |
| A team member can't explain part of the code at the defense | Flagged as vibe-coded | Every PR reviewed and explained by a teammate; walk-through session of all modules before Dec 1 |
| Archive dataset missing or incomplete for some period (`metrics`, `liquidationSnapshot`) | Block B partly empty | Check coverage in week 1; drop the feature or the period and say so |
| Leakage (same-time perp and spot prices) | Fake, inflated results | As-of joins, embargo, no-lookahead test; results that look too good = bug |
| Null result | Weak-looking conclusion | Hypotheses fixed in advance and an extensive descriptive analysis make a null result informative; report effect sizes with intervals |
| Binance changes archive format or endpoints | Download step breaks | Pin URLs and schemas in config; checksum checks fail loudly |
| Scope creep (other coins or horizons, TLOB, options, Streamlit) | Core unfinished by Nov 24 | Extensions only if the core runs end to end before the Nov 16 freeze |

**Limitations to state in the report:**

- Single venue, single asset, single horizon: results describe Binance BTC at 15 minutes and may not carry over to other coins, venues or horizons. The agnostic pipeline makes those checks cheap to run later.
- Trade-based target, no historical spot bid/ask, so some microstructure noise remains at 1 min.
- Predictive results, not causal ones (including Granger tests).
- Forecast gains need not be tradable after costs.
- The sample period's regime (trend, volatility) may not generalise.

## 13. Timeline and team split

**Rubric:** both deliverables — the repo + report (fresh-clone run, collaborative Git history) and the defense (time to build slides and rehearse the Q&A).

| Phase / milestone | Dates (2026) |
| --- | --- |
| Setup + topic approval | Oct 6 – Oct 12 |
| Data pipeline | Oct 12 – Oct 26 |
| Features + descriptive analysis | Oct 22 – Nov 6 |
| Models + testing | Nov 2 – Nov 16 |
| Figures + PDF report | Nov 9 – Nov 22 |
| Analysis freeze | Nov 16 |
| Fresh-clone run | Nov 21 |
| **Repo due (23:59 Paris time)** | **Nov 24** |
| Slides + rehearsal | Nov 25 – Nov 30 |
| Defense | Dec 1 |

The schedule is tight: seven weeks from today to the repo deadline. Phases overlap so nobody waits — features start on the first cleaned month while the pipeline finishes the rest. The analysis freezes on Nov 16; the last week before submission is for the PDF report, the fresh-clone test and fixes. Send the public repo link by email before Nov 24, 23:59 Paris time.

**First actions (this week):**

- [ ] Get the topic validated by the chargé de TD.
- [ ] Create the GitHub repo, `pyproject.toml` and `config.yaml`.
- [ ] Run the availability check for BTCUSDT spot and perp and fix the sample window.
- [ ] Download one month of BTC data end to end to size the pipeline.

**Team split (by pipeline stage):**

| Team Member | Role | Owns | Spec sections |
| --- | --- | --- | --- |
| Antoine | Data & Project Lead | `fetch.py`, `clean.py`, storage, availability check; project admin: repo setup, `README`, `spec`, topic approval, deadlines, sending the repo link | 1, 2, 3, 4, 11, 13 |
| Tina | Analysis Lead | `features.py`, descriptive analysis, figures, risks and limitations | 6, 7, 10, 12 |
| Arya | Modelling Lead | `models.py`, `evaluate.py`, statistical tests | 5, 8, 9 |

Everyone reviews another member's pull requests, and all three co-write the report. Because the Q&A can test any of us on any part of the code, hold a walk-through of every module in the week before the defense, where each person explains a module they did not write.