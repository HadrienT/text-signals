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
  Treasury yield (ΔDGS2), with a coefficient significantly different from zero after controlling
  for the policy surprise and for the Loughran–McDonald (LM) and FinBERT scores.
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
| Event table | `data/processed/fomc/statement_events.csv` | one row per statement: release, event and pre-event days, yield levels, ΔDGS2 / ΔDGS10 / Δ2s10s in bp, LM scores |

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
to 20 minutes after the statement) would solve them but needs paid tick data.

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

- **(a) Proposed main control: ΔDGS3MO on the event day.** The 3-month yield reflects the
  expected policy rate over the next three months; an expected decision is already priced and
  moves it little, a surprise moves it. Caveats: it is noisy (flight to quality, bill supply),
  and it also absorbs guidance about the next meeting. With this control, the tone coefficient
  measures what the text says *beyond* the next three months.
- **(b) Proposed robustness control: the change in the target rate** (e.g. +25 bp, 0). It is a
  known fact, not a surprise: an expected +25 bp hike shows +25 while markets barely move.

To be fixed before any regression is run.

### 6.3 Event study

- Regression: ΔDGS2_t = α + β·score_t + γ·controls_t + ε_t, with HAC (Newey–West) standard
  errors.
- Each score alone, then together: incremental R² and nested F-tests (LLM over LM and encoder).
- H2: the same regression on the change in score versus the previous statement.
- Secondary dependent variables: ΔDGS10 and Δ2s10s.
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
| Event window and join rule | fixed 2026-11-04 (§4.2) |
| LM score definition | fixed 2026-11-04 (§5.2) |
| Policy surprise control | proposed (§6.2) |
| H4 in or out | proposed (§2) |
| Minutes sample start | proposed (§5.1) |
| Encoder baseline | to choose |
| LLM schema and aggregate score | week 2 |
| Hand-labelling guide | week 2, before labelling |

If anything changes after results are known, the change and its reason go into the decision log
and the report.

---

## 8. Limitations known in advance

- Daily data: the statement, the press conference, the projections and other same-day news are
  mixed in one window (§4.3).
- The policy surprise is proxied by the 3-month yield, not measured from futures (§6.2).
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

- Gürkaynak, R., Sack, B. and Swanson, E. (2005). Do Actions Speak Louder Than Words? The
  Response of Asset Prices to Monetary Policy Actions and Statements. *International Journal of
  Central Banking*.
- Kuttner, K. (2001). Monetary policy surprises and interest rates: Evidence from the Fed funds
  futures market. *Journal of Monetary Economics*.
- Loughran, T. and McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis,
  Dictionaries, and 10-Ks. *Journal of Finance*.
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
