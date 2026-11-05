# Case study 1: FOMC communications

Study design for the first case study of `text-signals`: the goal, the data, the full method and
the decisions taken so far. It is a working document. When a decision changes, change it here and
record why in the [decision log](#12-decision-log). Items marked **proposed** are not decided yet.

Anything generic (document model, fetch cache, point-in-time join, extraction cache, evaluation
metrics, event study) goes into the shared library, so that later case studies (SEC filings,
earnings calls, other central banks) can reuse it.

---

## 1. Goal

### 1.1 The question

When the Federal Reserve publishes a monetary policy statement, the rate decision itself is
usually expected: markets have priced it in the days before. What moves the bond market on the
day is mostly what the text implies about the **future path** of policy. Gürkaynak, Sack and
Swanson (2005) show that a "path" factor, distinct from the surprise in the rate decision,
accounts for most of the response of longer yields to FOMC announcements.

The study asks:

> Can a local LLM, reading only the text of an FOMC communication, measure its policy stance well
> enough to explain the bond market's reaction on the release day, beyond the rate decision and
> beyond classic text baselines? And how much of that result survives once look-ahead bias is
> removed?

Example from the corpus: on 13 December 2023 the FOMC left rates unchanged, as expected, but the
statement now spoke of "the extent of **any** additional policy firming". Markets read the end of
the hiking cycle, and the 2-year yield fell 27 bp that day. (The projections and the press
conference released the same afternoon contributed too; see §4.3.)

### 1.2 What the study is not

- **Not a speed race.** The number in a statement is parsed by machines within milliseconds of
  release. A local LLM taking seconds per document cannot compete on that, and does not try to.
- **Not a trading strategy as such.** H1–H3 measure how much of the same-day reaction the text
  explains. That is an explanatory result. A predictive question, without a speed race, is the
  proposed H4.

### 1.3 Why it matters for investment research

1. **Measurement.** A structured, auditable reading of central bank communication (stance on
   inflation, labour, growth, forward guidance, wording changes with verbatim evidence) that can
   feed rates research, risk attribution and medium-frequency models.
2. **Analyst productivity.** Most of the value is in long documents that nobody reads in five
   minutes: minutes, speeches, and later 10-K filings and earnings calls. The same pipeline
   applies to them.
3. **Look-ahead bias.** An LLM trained on data after a document's date has read how history
   unfolded. Showing how much of a backtest survives after the model's training cutoff, or on
   anonymised documents, matters more than a high R².

---

## 2. Hypotheses

- **H1 (signal).** On FOMC statement days, an LLM tone score explains the change in the 2-year
  Treasury yield in the 30-minute statement window (USMPD `UST2Y`, §3.4), with a coefficient
  significantly different from zero after controlling for the policy surprise (MP1) and for the
  Loughran–McDonald (LM) and FinBERT scores. *Until 2026-11-05 the outcome was the daily ΔDGS2,
  which is now a robustness check (§12).*
- **H2 (incremental content).** The LLM score keeps explanatory power on the *change* in wording
  versus the previous statement, not only on its level.
- **H3 (look-ahead).** The H1 effect is weaker (i) on meetings after the model's training cutoff
  and (ii) on anonymised documents. The size of the gap is a result in itself, whichever way it
  goes.
- **H4 (drift), proposed.** The LLM tone score predicts the change in DGS2 over the following
  days (from the event day to event day + 5 trading days), i.e. the market takes time to absorb
  the nuances of the text. This is a genuinely out-of-sample predictive test. It must be accepted
  or dropped before any H4 number is computed.

Negative results get reported. A null H1 with a clean protocol is a valid deliverable.

---

## 3. Data

### 3.1 Inventory

| Data | Source | Coverage | Status |
|---|---|---|---|
| FOMC meeting calendar | federalreserve.gov: `fomccalendars.htm`, `fomchistoricalYYYY.htm` | 1994 → 2027 (318 meetings and calls) | fetched |
| FOMC statements | federalreserve.gov press releases (4 URL layouts, 1994 → today) | 250 statements, 1994-02-04 → 2026-10-28 | fetched, parsed |
| FOMC minutes | federalreserve.gov (`/fomc/MINUTES/…`, `/fomc/minutes/…`, `/monetarypolicy/fomcminutes…`) | links and release dates already parsed from the calendar | to fetch |
| Press conference transcripts | federalreserve.gov (`fomcpresconfYYYYMMDD.htm`, transcripts in PDF) | April 2011 → today | to fetch |
| 2-year Treasury yield, DGS2 | FRED `fredgraph.csv` (no key) | 1976 → today, daily | fetched |
| 10-year Treasury yield, DGS10 | FRED `fredgraph.csv` | 1962 → today, daily | fetched |
| 3-month Treasury yield, DGS3MO | FRED `fredgraph.csv` | 1981 → today, daily | to fetch (control, §6.2) |
| Intraday rate changes and policy surprises (USMPD) | San Francisco Fed, `sffed.us/usmpd` | 280 statements 1994 → 2026-10-28, press conferences from 2011, minutes from 2001 | fetched, parsed, joined to the event table (§3.4) |
| Loughran–McDonald dictionary | Notre Dame SRAF, master dictionary 1993-2025 (Google Drive) | 86,553 words | fetched, sha256 pinned |
| FinBERT / hawkish-dovish encoder | Hugging Face | – | to choose (§5.2) |
| Local LLMs | open-weight models served by `llama-server` | – | to choose (week 2) |
| Hand-labelled statements | written by us | ~60 statements | to do (week 2) |

Everything is fetched by code (`just fetch-fomc`) and nothing in `data/` is committed. The
hand-labelled sets are the exception: they cannot be regenerated, so they will be versioned under
`labels/`.

### 3.2 Formats

| Layer | Path | Format |
|---|---|---|
| Raw | `data/raw/<host>/<path>` | files exactly as fetched (HTML, CSV, PDF) |
| Provenance | `data/raw/manifest.jsonl` | one JSON line per file: `url`, `final_url`, `fetched_at`, `status_code`, `content_type`, `sha256`, `path`, `note` |
| Documents | `data/interim/fomc/statements.jsonl` | one `Document` per line (below) |
| Event table | `data/processed/fomc/statement_events.csv` | one row per statement: release, event and pre-event days, daily yield levels, ΔDGS2 / ΔDGS10 / Δ2s10s in bp, USMPD 30-minute changes (2y, 10y, 2s10s), MP1 (published and recomputed), MP2 and USMPD version, main-sample flags, LM scores |

`Document` fields: `doc_id`, `corpus`, `doc_type`, `event_date` (last day of the meeting),
`release_date` (the point-in-time key), `release_time_et` (when the page states it),
`release_time_source` (`document` or `none`), `unscheduled`, `url`, `raw_sha256` (links to the
manifest), `title`, `text` (paragraphs separated by a blank line).

### 3.3 Provenance rules

- Raw files are stored as fetched, with URL, fetch time and sha256. A cached file that no longer
  matches its sha256 is an error.
- Files whose content is pinned (the LM dictionary) are checked against the pinned sha256
  *before* being cached. Google Drive answers HTTP 200 with a "quota exceeded" page at times; a
  copy downloaded by hand can be imported (`just import-lm <file>`) only if it matches the pin,
  and the manifest says so in `note`.
- Polite fetching: a 2-second delay between requests and a descriptive User-Agent without a
  personal address.

### 3.4 Intraday data: the San Francisco Fed event-study database (found 2026-11-05)

The **U.S. Monetary Policy Event-Study Database** (USMPD) is published by the San Francisco Fed's
Center for Monetary Research (Acosta, Ajello, Bauer, Loria and Miranda-Agrippino, 2025). It is
free and needs no key. Its intraday prices come from LSEG Tick History, which we could not buy.
Its findings, and our own checks of it, go into the report.

**Files inspected on 2026-11-05** (version "Last update: 30 October 2026"). `USMPD.xlsx` is
fetched by the pipeline (`text_signals.sources.fomc.usmpd`) into `data/raw/` with a manifest line;
the fetched copy has the sha256 below. The ZIP was only inspected by hand:

| File | URL | sha256 |
|---|---|---|
| `USMPD.xlsx` | `https://www.frbsf.org/wp-content/uploads/USMPD.xlsx` | `2b91297c42cf692dd74f65eabf7ee97f81d452676e23fa02edb222bfa60f0d0a` |
| `monetary-policy-surprises.zip` (R code, `mps.csv`, `mps_minutes.csv`) | `https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises.zip` | `ee9f98bc6a2b49e43078dde0028413c6ca008df39fa57104ca1bf1174c27af0d` |

The page states no licence. It asks to cite the working paper (§11).

**Content.** There is one row per event, with the event's timestamp in Eastern Time. The database
gives the *change* of each price over the window, not the tick series.

| Sheet | Events | Window |
|---|---|---|
| Statements | 280, 1994-02-04 → 2026-10-28 | 10 min before → 20 min after the release |
| Press Conferences | 96, from 2011-04-27 | 10 min before → 60 min after its start |
| Monetary Events | 280 | statement and press conference together (100 min) |
| Minutes | 207, from 2001-02-01 | 10 min before → 20 min after the release |

The variables are the following. Rates are in percentage points and returns in percent.

- MP1 and MP2: fed funds futures surprises (below).
- FF1–FF6: fed funds futures.
- ED1–ED8: Eurodollar futures, then SOFR futures from 2022.
- OIS 1y and 2y.
- Treasury yields 3m, 6m, 2y, 5y, 10y and 30y, and TIPS 5y, 10y and 30y.
- S&P 500 and the E-mini future, DXY, EURUSD and USDJPY.
- Indicator flags: `SEP`, `Unscheduled` and `PC`.

**Policy surprises.**

- **MP1** is the surprise in the current decision (Kuttner, 2001). Write D for the number of days
  in the month and d for the day of the meeting. Then MP1 = FF1 × D / (D − d), or MP1 = FF2 when
  the meeting falls in the last 7 days of the month. We recomputed it from FF1 and FF2 and
  matched the database on **280 of 280** statements.
- **MP2** is the surprise in the expected rate after the next meeting (Gürkaynak, Sack and
  Swanson, 2005).
- **The PCA surprise** (`mps.csv`, from the authors' `mps.R`) is the first principal component
  of MP1, MP2 and ED2–ED4. These cover about one year ahead (Nakamura and Steinsson, 2018). It is
  scaled by a regression of the daily change in the one-year zero-coupon yield on it, so that it
  moves that yield one for one.
  - *Caveat for our question:* this surprise spans about a year, so it **includes the
    guidance**. Used as a control, it would absorb the effect the text is meant to carry. MP1
    only removes the surprise in the decision itself.
  - *Look-ahead:* the principal component and its scaling are estimated on the full sample. They
    are fine in-sample but not for the out-of-sample test (§6.4); `mps.R` has an `END_DATE`
    option for this.

**Where each concept comes from.** The report cites the paper that introduced each concept, not
only the database that computes it.

| Concept | Origin | What we take from it |
|---|---|---|
| Policy surprise from fed funds futures (MP1), scaled by D / (D − d) | Kuttner (2001) | the surprise in the current decision |
| Next-meeting surprise (MP2); target and path factors (`gss.R`) | Gürkaynak, Sack and Swanson (2005) | the surprise about the next meeting; the split between the decision and the guidance |
| The last-7-days rule (use FF2) | used in Gürkaynak, Sack and Swanson (2005), to confirm in the paper | it matches USMPD on 280 of 280 statements |
| Surprise as the first principal component of futures changes over about one year | Nakamura and Steinsson (2018) | the "policy news" surprise |
| The USMPD variant: inputs MP1, MP2 and ED2–ED4, scaled on the one-year yield; the windows and the database | Acosta et al. (2025) | the intraday changes and the surprises we use |
| One-year zero-coupon yield used for the scaling (`SVENY01`) | Gürkaynak, Sack and Wright (2007) | the yield curve behind `y1.csv` |

**Checks against our event table (§3.2).**

- **Overlap.** 248 of our 250 statements match by date.
  - The 32 USMPD events we lack are, apart from two, the 1994–1999 meetings that left rates
    unchanged. No statement was issued then, so there is no text to score.
  - The other two are the unscheduled actions of 4 October 2019 and 19 March 2020.
- **Mismatch 1: October 2019. Resolved 2026-11-05: the USMPD row is misdated.**
  - USMPD dates an event on 4 October 2019 at 11:00 ET, the day of the conference call.
  - The Fed's press release index (`federalreserve.gov/json/ne-press.json`, read 2026-11-05)
    lists no monetary policy release on 4 October 2019. The "Statement Regarding Monetary Policy
    Implementation" came out on **11 October 2019 at 11:00 AM**, the time USMPD gives. Powell's
    only remarks on 4 October were at 2:00 PM (`ne-speeches.json`).
  - For every other unscheduled event, USMPD uses the release date (e.g. 22 January 2008 for
    the call of the 21st), as we do. The row looks like the call date combined with the release
    time.
  - Which day its numbers measure cannot be known without the ticks, so the row is used for
    nothing.
  - The announcement had also been previewed: on 8 October 2019 at 2:30 PM, Powell said "my
    colleagues and I will soon announce measures to add to the supply of reserves over time"
    (`/newsevents/speech/powell20191008a.htm`). The statement itself calls the measures "purely
    technical" and "not a change in the stance of monetary policy".
- **Mismatch 2: 22 August 2025. Resolved 2026-11-05: confounded with a speech by the Chair.**
  - The revised Statement on Longer-Run Goals was released at **10:00 AM**, the minute Powell's
    Jackson Hole speech began ("Monetary Policy and the Fed's Framework Review",
    `powell20250822a.htm`). Any window, daily or intraday, mixes the two, and the speech carried
    the policy news. Our daily ΔDGS2 that day is −11 bp.
  - The 2020 revision has the same structure: statement and Powell speech both at 9:10 AM on
    27 August 2020. USMPD **includes** 2020-08-27 but not 2025-08-22, so its coverage of these
    events is not consistent and cannot settle the question for us.
- **Corpus gap to check.** USMPD's unscheduled event of 19 March 2020 at 9:00 is the dollar
  swap lines with nine central banks (`monetary20200319b.htm`). It is not in our statement
  corpus, which follows the FOMC calendar links.
- **Daily against intraday.** On the 248 common statements, our daily changes and the 30-minute
  changes are correlated at **0.61 (2y)** and **0.64 (10y)**. The standard deviation of the
  2-year change is 7.8 bp daily and 5.7 bp intraday. Example: on 18 March 2026 the 2-year yield
  rose 8 bp over the day and fell 0.9 bp in the statement window.
- **Release times.** USMPD records times of day that our pages do not give before 2016 (§4.1):
  14:15 on 125 statements, 14:00 on 109. They can be used as a cross-check, cited as USMPD, not
  as our own data.

**What was looked at.** We compared yield changes between windows and recomputed MP1. LM scores
were printed next to yield changes and surprises for six 2026 statements, to read the table. No
relation between a score and a market move was computed. The window and control decisions below
are therefore still made before seeing results.

**Decisions (fixed 2026-11-05, §7).**

1. **The main event window is the USMPD 30-minute statement window** (10 minutes before to 20
   minutes after the release). The daily window of §4.2 becomes a robustness check.
2. **The policy surprise control is MP1 from USMPD** (§6.2). ΔDGS3MO becomes a robustness
   control.

3. **Main sample (fixed 2026-11-05).** A statement is in the main sample when it has a USMPD
   30-minute change and was not released together with a speech by the Chair. This rule leaves
   out three statements, which stay in the table and enter the daily robustness check:
   - **11 October 2019**, because the USMPD row for it is misdated;
   - **27 August 2020** and **22 August 2025**, the framework statements released together with
     Powell's Jackson Hole speeches.

   That leaves 247 of 250 statements. The columns are `confounded_chair_speech` and
   `in_main_sample` in the event table.

   **Why the rule removes only these two statements (checked 2026-11-05).**
   - **Chair speeches.** We compared each USMPD statement time with every Chair speech that has
     a time in the Fed's speech index (`ne-speeches.json`; times from December 2008). Over 151
     statements, the only speech within two hours of a statement is 27 August 2020, at 0
     minutes. We checked 22 August 2025, which is not in USMPD, against our own release time:
     also 0 minutes.
   - **What this check cannot cover.** There are no speech times before December 2008, and the
     index holds speeches only, not press conferences, testimony or interviews.
   - **Press conferences** are not a confound for the main window, because it closes before the
     conference starts. From the USMPD timestamps:

     | Period | Statement → press conference | End of the +20 min window |
     |---|---|---|
     | 2011–2012 | 12:30 → 14:15, 105 min | 85 min before |
     | 2013 → 2026 | 14:00 → 14:30, 30 min | 10 min before |
     | 3 and 15 March 2020 (unscheduled) | 60 and 90 min | 40 and 70 min before |

---

## 4. Timing: the point-in-time rule

### 4.1 Three documents, three moments

| Document | Content | Released |
|---|---|---|
| Statement | one page: decision and stance | day of the meeting, early afternoon ET, all at once (2:00 p.m. on every page that states a time) |
| Press conference | the Chair answers questions, live | same day, shortly after the statement, about one hour (since April 2011; every meeting since 2019) |
| Minutes | detailed account of the discussion, 6,000–10,000 words | about three weeks later, all at once (on this schedule since January 2005) |

Every document is keyed by its **release** date, never by the meeting date.

The statement pages state a time of day only from 2016 (all 2:00 p.m.); earlier pages say "For
immediate release". The Fed's usual earlier schedule (around 2:15 p.m., and earlier in the day on
press conference days in 2011–2012) is not recorded in our data and is not imputed. It is to be
checked against the Fed's archives if a result ever depends on it; the daily window does not,
since all those times precede the 3:30 p.m. fixing.

### 4.2 Event window

**Since 2026-11-05 the main window is the USMPD 30-minute window around the statement (§3.4).**
The daily window below is kept as a robustness check, and for any statement that USMPD does not
cover.

DGS2, DGS10 and DGS3MO are daily closing quotes, taken at about **3:30 p.m. ET**.

- **Event day**: the release date if the release came before 3:30 p.m., otherwise the next day
  with an observation.
- **Pre-event day**: the last day with an observation before the event day.
- Days on which the bond market was closed (empty values in FRED) are skipped on both sides.
- A release without a stated time is assumed to precede the fixing. This holds for every
  scheduled statement.
- Change = value(event day) − value(pre-event day), in basis points.

Two statements roll forward to the next day: Sunday 9 May 2010 (9:15 p.m.) and Sunday 15 March
2020 (5:00 p.m.).

### 4.3 What a daily window cannot separate

- **Statement vs press conference.** From 2011, the window from the previous 3:30 p.m. to the
  release day's 3:30 p.m. contains the statement *and* the press conference (today 2:00 p.m.
  and 2:30–3:30 p.m.), plus the Summary of Economic Projections on quarterly meetings.
  ΔDGS2 is the reaction to all of them. The press conference transcript can be tested for
  incremental content, but the two reactions cannot be separated in time without intraday data.
- **Other news.** Any macro release on the same day is in the window too.

Both are limitations stated in the report, not hidden. An intraday window (e.g. 10 minutes before
to 20 minutes after the statement) would solve them. Tick data is paid, but the San Francisco
Fed's USMPD publishes such window changes for free (§3.4); whether it becomes the main window is
an open decision (§7).

---

## 5. Corpus and scores

### 5.1 Corpus decisions

- **Statements are irregular before May 1999.** From February 1994 to March 1999 a statement was
  issued only when policy changed. The regular sample starts in May 1999. Earlier statements are
  kept, and the main event study is also run from May 1999 onwards. Meetings without a statement
  do not get a fake "no change" document.
- **Unscheduled releases** (conference calls, intermeeting actions, notation votes) are flagged,
  not dropped: 18 of the 250 statements. Some are not rate decisions at all (swap lines in 2008
  and 2010, the FIMA repo facility, the 2020 and 2025 framework statements). The main event study
  uses scheduled meetings; unscheduled releases form a separate subsample.
- **Statements released after the meeting day.** Six follow a conference call held the day
  before (e.g. 21 January 2008, announced on the morning of the 22nd) or a week before (4 October
  2019 call, 11 October statement). The release date is the event date.
- **Text as published**, vote and related Board actions included; only page furniture is
  removed. Any further trimming is a scoring choice and lives in the scoring code.
- **Minutes, proposed: event study from 2005 only.** Before December 2004 the minutes were
  released just after the *next* meeting, when markets already knew the next decision. From
  January 2005 they come out about three weeks after the meeting, before the next one, which
  makes them a distinct event.

### 5.2 Scores

1. **LM baseline** (fixed on 2026-11-04, before any regression).
   - Master dictionary 1993-2025, sha256 pinned in `text_signals/lexicon.py`.
   - A word is in a list when its column holds a positive year; words removed in 2020 are
     excluded.
   - Tokens are alphabetic runs, upper-cased; the denominator counts every token.
   - `lm_net_negative = (negative − positive) / tokens`, `lm_uncertainty = uncertainty / tokens`,
     on the full statement text.
   - Known limitation: LM was built from 10-K filings; "inflation" or "tightening" carry no sign.
2. **Encoder baseline.** Mean sentence-level probability from a finance encoder. Candidates:
   FinBERT (general financial sentiment) and a hawkish/dovish classifier trained on FOMC text
   (e.g. Shah, Paturi and Chava, 2023). The second is closer to the task but was trained on
   labelled FOMC sentences, which is a look-ahead issue of its own: its training period must be
   recorded like an LLM's cutoff. To choose in week 1.
3. **LLM structured extraction** with a fixed Pydantic schema, settled in week 2. Candidate
   fields:
   - `policy_action`: hike / cut / hold, and size;
   - `forward_guidance_stance`: −2 … +2, dovish → hawkish;
   - `inflation_assessment`, `labour_assessment`, `growth_assessment`: −2 … +2;
   - `balance_of_risks`: categorical;
   - `changed_phrases`: wording changes versus the previous statement, each with a quote;
   - `evidence`: a verbatim span for every field, used to check faithfulness.

   Deterministic decoding (temperature 0, fixed seed). Prompt and schema versioned in the
   repository. Every output stored with model name, prompt version, schema version and decoding
   parameters, and cached by (document sha256, model, prompt version) so that long runs can
   resume.
4. **Aggregate LLM score.** A predefined combination of the fields, fixed *before* the
   regressions are run, so that no specification is fitted to the result.

---

## 6. Method

### 6.1 Pipeline

```
fetch (cache + manifest)
  → parse (one Document per release, keyed by release date)
  → score (LM, encoder, LLM extraction)
  → evaluate the extraction against hand labels
  → join market data point-in-time (event window)
  → event study, out-of-sample test, placebos
  → look-ahead tests
```

Each step is code in `src/text_signals/`, tested, and rerunnable from scratch.

### 6.2 Controlling for the rate decision

Without a control for the decision itself, a tone score may just proxy "they hiked". The
reference measure of the policy surprise (Kuttner, 2001) uses fed funds futures prices, which are
not freely available historically.

**Fixed 2026-11-05: the main control is (c), MP1. (a) is kept as a robustness control.**

- **(a) Robustness control (was the proposed main control): ΔDGS3MO on the event day.** The 3-month yield reflects the
  expected policy rate over the next three months; an expected decision is already priced and
  moves it little, a surprise moves it. Caveats: it is noisy (flight to quality, bill supply),
  and it also absorbs guidance about the next meeting. With this control, the tone coefficient
  measures what the text says *beyond* the next three months.
- **(b) Proposed robustness control: the change in the target rate** (e.g. +25 bp, 0). It is a
  known fact, not a surprise: an expected +25 bp hike shows +25 while markets barely move.
- **(c) Found 2026-11-05: MP1 from the USMPD** (§3.4). This is the Kuttner measure itself, from
  intraday fed funds futures, and it removes the need for the (a) proxy. The broader PCA surprise
  is *not* a candidate control, because it includes the guidance the text carries. It could
  instead be a second outcome: does the tone predict the surprise in the path of rates over the
  next year?

Fixed before any regression was run (§7).

### 6.3 Event study

**Model.** Estimated by OLS, one observation per statement:

  Δy_i = α + β·score_i + γ·MP1_i + δ'·baselines_i + ε_i

Δy_i is the 30-minute change in the 2-year yield (USMPD `UST2Y`, in bp). The baselines are the
LM and encoder scores.

**Primary test (one, fixed in advance).** The test is H0: β_LLM = 0 against a two-sided
alternative. It uses the 30-minute 2-year yield change, the aggregate LLM score, MP1 and the
baselines as controls, and the main sample of statements (§3.4). It is a Wald test with HAC variance.
Everything else below is secondary.

**Assumptions, and how each is justified or checked.**

| Assumption | If it fails | Check |
|---|---|---|
| Exogeneity, E[ε \| X] = 0 | β is biased | Not testable directly. It rests on the design: the text is released before the move, inside a 30-minute window, so there is no reverse causality. The main omitted variable, the decision, is controlled by MP1. The placebos (§6.5) also test it. |
| Score measured without error | attenuation, β is biased toward 0 | agreement with the hand-labelled set (§6.6). A null result is read in the light of it. |
| No strong collinearity (score, MP1, baselines) | inflated and unstable β | correlation matrix and VIF, reported. A VIF above 5 is flagged. |
| Homoskedasticity | wrong standard errors | Breusch–Pagan and White, reported |
| No residual autocorrelation | wrong standard errors | Breusch–Godfrey and Ljung–Box (4 lags), reported |
| Normal errors | tests are only asymptotic with T ≈ 250 | Jarque–Bera and QQ-plot, reported |

**Inference, fixed in advance.**

- The main standard errors are **HAC (Newey–West)**, with ⌊4·(T/100)^(2/9)⌋ lags, which is 4
  for T ≈ 250. They are chosen *a priori* and are **not** selected from the diagnostic tests:
  choosing the estimator after a pre-test distorts the size of the final test. The diagnostics
  document the data; they do not choose the method.
- **Robustness of the inference:**
  - HC3 standard errors, more reliable in small samples when autocorrelation is absent;
  - a moving-block pairs bootstrap of β, with blocks of 4 statements and 9,999 replications.
- **Influential observations:** Cook's distance, with 4/T flagged. The model is re-estimated
  without the flagged observations and with a Huber robust regression. Both are reported, and no
  observation is dropped silently.
- **Nested models** (LLM over LM and encoder): incremental R², and a HAC Wald test on the LLM
  coefficients. The test is joint when several LLM fields enter.
- **Effect size:** β is reported in bp per standard deviation of the score, next to its
  significance.
- **Multiple testing:** the secondary tests (other outcomes, scores, subsamples, H2) are reported
  with Holm-adjusted p-values. A conclusion rests on the primary test only.

**Secondary specifications.**

- Each score alone, then together.
- H2: the same regression on the change in score versus the previous statement.
- Secondary dependent variables: the 30-minute changes in the 10-year yield (`UST10Y`) and in
  the 2s10s slope.
- Robustness outcome: the daily ΔDGS2 (§4.2). Robustness control: ΔDGS3MO in place of MP1
  (§6.2).
- Subsamples, reported, never dropped silently: May 1999 onwards; scheduled only; the
  zero-lower-bound periods (December 2008 – December 2015, March 2020 – March 2022), where the
  2-year yield is pinned near zero; with and without press conference days.

### 6.4 Out-of-sample test

- Expanding window, refitted every year; the score's coefficient is estimated on past meetings
  only.
- Out-of-sample R² against a zero forecast and against the baselines.

### 6.5 Placebos

- Shuffled event dates: the score should explain nothing.
- Score of the previous meeting instead of the current one.

### 6.6 Evaluation of the extraction

- Hand-label about 60 statements, stratified by decade and policy regime, **before** looking at
  model outputs.
- Per-field accuracy or MAE; Cohen's κ against a second labelling pass on a subset; rate of
  `evidence` spans that do not appear verbatim in the document (hallucination).
- Compare 2–3 local models on quality, latency and tokens per document.

### 6.7 Look-ahead tests (H3)

- **Cutoff split.** Find each model's training cutoff (model card, or probing with dated
  events), then compare β and the out-of-sample R² before and after it. Every score records the
  model, its cutoff and whether the meeting is before or after.
- **Anonymisation.** Remove dates, names of Chairs and governors, and explicit year markers;
  score again; compare scores and β.
- **Recognition probe.** Ask the model to date an anonymised statement. High accuracy means the
  model recognises documents rather than reading them.

### 6.8 Retrieval agent (week 4)

- Question answering over the corpus ("how did the inflation wording change between 2021 and
  2023?"), with verbatim citations (document, paragraph).
- About 50 hand-written questions with reference answers and passages.
- Metrics: retrieval recall@k, citation faithfulness (does the cited passage support the claim?),
  correct refusals.
- Stack, decided in week 4: `kbase` from AgenticEnv (pgvector, hybrid retrieval) or a light
  in-repo index. The corpus is small.

---

## 7. Specification freeze

Fixed before seeing any regression output, and dated here when fixed:

| Item | Status |
|---|---|
| Daily event window and join rule | fixed 2026-11-04 (§4.2); robustness check since 2026-11-05 |
| Main event window: USMPD 30-minute statement window | fixed 2026-11-05 (§3.4) |
| LM score definition | fixed 2026-11-04 (§5.2) |
| Policy surprise control: MP1 from USMPD, with ΔDGS3MO as robustness | fixed 2026-11-05 (§6.2) |
| Primary test, inference (HAC, 4 lags), diagnostics and robustness | fixed 2026-11-05 (§6.3) |
| H4 in or out | proposed (§2) |
| Minutes sample start | proposed (§5.1) |
| Encoder baseline | to choose |
| LLM schema and aggregate score | week 2 |
| Hand-labelling guide | week 2, before labelling |

If anything changes after results are known, the change and its reason go into the decision log
and the report.

---

## 8. Limitations known in advance

- Intraday changes and surprises are taken from USMPD as published. We cannot rebuild them from
  ticks (paid LSEG data); we can only recompute MP1 from the futures changes (§3.4). The daily
  window remains as an independent check.
- The PCA surprise of USMPD is estimated on the full sample, so it is not used out of sample
  without refitting (§3.4).
- Small sample: about 220 scheduled statements from May 1999, fewer with press conferences;
  standard errors will be wide.
- Pinned rates at the zero lower bound flatten the 2-year yield's response for several years.
- LLM look-ahead: most open-weight models have read the whole history of the corpus (§6.7).

---

## 9. Deliverables

- A 5–6 page research report (`reports/fomc.md` or PDF): question, data, method, results,
  look-ahead analysis, limitations.
- A README with the main results table.
- A pipeline rerunnable from scratch: `fetch → parse → score → evaluate → regress`.

---

## 10. Schedule

| Week | Deliverable |
|---|---|
| 1 | Fed texts and FRED data ingested, versioned dataset, LM and encoder baselines |
| 2 | Pydantic extraction schema, LLM extraction, hand-labelled set, extraction metrics |
| 3 | Event study, out-of-sample test, placebos, cutoff and anonymisation tests |
| 4 | Retrieval agent and its evaluation, research report, final README |

---

## 11. References

- Acosta, M., Ajello, A., Bauer, M., Loria, F. and Miranda-Agrippino, S. (2025). Financial
  Market Effects of FOMC Communication: Evidence from a New Event-Study Database. *Federal
  Reserve Bank of San Francisco Working Paper* 2025-30, doi:10.24148/wp2025-30. Database:
  `https://sffed.us/usmpd`.
- Gürkaynak, R., Sack, B. and Swanson, E. (2005). Do Actions Speak Louder Than Words? The
  Response of Asset Prices to Monetary Policy Actions and Statements. *International Journal of
  Central Banking* 1(1), 55–93. MP2, target and path factors.
- Gürkaynak, R., Sack, B. and Wright, J. (2007). The U.S. Treasury Yield Curve: 1961 to the
  Present. *Journal of Monetary Economics* 54(8), 2291–2304. Zero-coupon yields (`SVENY01`),
  data at `https://www.federalreserve.gov/data/yield-curve-tables/feds200628_1.html`.
- Kuttner, K. (2001). Monetary policy surprises and interest rates: Evidence from the Fed funds
  futures market. *Journal of Monetary Economics* 47(3), 523–544. MP1.
- Loughran, T. and McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis,
  Dictionaries, and 10-Ks. *Journal of Finance* 66(1), 35–65.
- Nakamura, E. and Steinsson, J. (2018). High-Frequency Identification of Monetary
  Non-Neutrality: The Information Effect. *Quarterly Journal of Economics* 133(3), 1283–1330.
  Surprise as a principal component of futures changes.
- Shah, A., Paturi, S. and Chava, S. (2023). Trillion Dollar Words: A New Financial Dataset, Task
  & Market Analysis. *ACL*.

---

## 12. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-04 | Keep pre-1999 statements, flag them, also run from May 1999 | statements were issued only on policy changes before 1999 |
| 2026-11-04 | Release time recorded only when the page states it | imputing 2:15 p.m. would invent data; a daily window needs only the date |
| 2026-11-04 | Unscheduled releases flagged, separate subsample | several are not rate decisions |
| 2026-11-04 | Text stored as published | trimming is a scoring choice |
| 2026-11-04 | Event window anchored on the 3:30 p.m. Treasury fixing | after-close and weekend releases must roll forward |
| 2026-11-04 | FRED fetched in-repo, not read from `data-ingest` | the pipeline must be rerunnable by a reader; the two copies matched exactly |
| 2026-11-04 | LM score definition fixed, dictionary sha256 pinned | fix the baseline before seeing results |
| 2026-11-05 | Protocol rewritten as a complete study design | one document for goal, data and method; new items marked proposed |
| 2026-11-05 | USMPD found and inspected; window and control left open (§3.4) | free intraday changes and futures-based surprises would lift two known limitations; no score–outcome relation looked at yet |
| 2026-11-05 | Main window changed from daily to the USMPD 30-minute statement window; the daily window becomes robustness | it isolates the statement from the press conference and same-day news; it replaces an item fixed on 2026-11-04, before any regression was run |
| 2026-11-05 | Main policy surprise control: MP1 from USMPD; ΔDGS3MO becomes robustness | the futures-based measure of the literature (Kuttner, 2001); the PCA surprise is rejected as a control because it includes the guidance |
| 2026-11-05 | H1 outcome restated as the 30-minute 2-year change | follows the window decision above |
| 2026-11-05 | USMPD row of 4 October 2019 not used; the 11 October 2019 statement leaves the main sample | no Fed release on 4 October; the row has the call date with the release time (§3.4) |
| 2026-11-05 | Statements released with a speech by the Chair (27 August 2020, 22 August 2025) leave the main sample | no window separates the two; USMPD covers 2020 and not 2025, so its coverage cannot decide |
| 2026-11-05 | One primary test; HAC with 4 lags fixed a priori; diagnostics reported, not used to choose; HC3, block bootstrap and Huber as robustness; Holm for secondary tests | each element tied to an assumption; no pre-testing; limits false positives from many specifications |
