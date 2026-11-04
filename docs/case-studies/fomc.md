# Case study 1: FOMC communications

Protocol for the first case study. It is a working document: update it when a decision
changes, and record why. Anything generic (document model, point-in-time join, extraction
cache, evaluation metrics, event study) goes into the shared library so that the next case
studies can reuse it.

## 1. Hypotheses

- **H1 (signal).** On FOMC announcement days, an LLM tone score explains the change in the
  2-year yield (ΔDGS2), with a coefficient significantly different from zero after controlling for
  the Loughran–McDonald (LM) and FinBERT scores.
- **H2 (incremental content).** The LLM score keeps explanatory power on the *change* in wording
  versus the previous statement, not only on its level.
- **H3 (look-ahead).** The H1 effect is weaker (i) on meetings after the model's training cutoff
  and (ii) on anonymised documents. The size of that gap is a result in itself, whichever way it
  goes.

Negative results get reported. A null H1 with a clean protocol is still a valid deliverable.

## 2. Corpus

| Document | Period | Use |
|---|---|---|
| FOMC statements | 1994 → today | main event study |
| FOMC minutes | 1993 → today (released ~3 weeks later) | robustness, separate event date |
| Press conference transcripts | 2011 → today | extension |

Each document carries its **release timestamp** (not the meeting date). Every join with market
data goes through that timestamp. This is the point-in-time rule. Store raw HTML/PDF as fetched,
together with the URL, fetch time and sha256.

## 3. Market variables

- Main dependent variable: ΔDGS2 between the day before and the release day (daily FRED data).
- Secondary: ΔDGS10, the 2s10s slope.
- Known limitation: a daily window includes other news. An intraday window would be cleaner but
  needs paid data. Write this limitation down rather than hide it.
- Possible control: unscheduled meetings and the zero-lower-bound period (2009–2015, 2020–2021)
  are treated as subsamples, not dropped silently.

## 4. Scores

1. **LM baseline:** (negative − positive) / total words, plus an uncertainty word count.
2. **FinBERT baseline:** mean sentence-level hawkish/dovish or sentiment probability.
3. **LLM structured extraction** with a fixed Pydantic schema. Fields to settle in week 2, for
   example:
   - `policy_action` (hike / cut / hold, size)
   - `forward_guidance_stance` (−2 … +2, dovish → hawkish)
   - `inflation_assessment`, `labour_assessment`, `growth_assessment` (−2 … +2)
   - `balance_of_risks` (categorical)
   - `changed_phrases`: wording changes versus the previous statement, each one with a quote
   - `evidence`: a verbatim span for every field (used to check faithfulness)
   Use deterministic decoding (temperature 0, fixed seed), and version the prompt and the schema
   in the repository.
4. **Aggregate LLM score:** a predefined combination of the fields, fixed *before* the
   regressions are run, so that no specification gets fitted to the result.

## 5. Evaluation of the extraction

- Hand-label about 60 statements, stratified by decade and by policy regime. Label them before
  looking at the model outputs.
- Metrics: per-field accuracy or MAE, Cohen's κ against a second labelling pass on a subset,
  and the rate of `evidence` spans that do not appear verbatim in the document (hallucination).
- Compare 2–3 local models on quality, latency and tokens per document.

## 6. Event study

- Regression: ΔDGS2_t = α + β·score_t + γ·controls_t + ε_t, with HAC standard errors.
- Run it with each score alone and then together (incremental R², nested F-test).
- Out-of-sample: an expanding window, refitted every year, reporting the out-of-sample R² against a
  zero forecast and against the baselines.
- Placebos: shuffled dates, and scores taken from the previous meeting.

## 7. Look-ahead tests (H3)

- **Cutoff split:** find the training cutoff of each model, from the model card or by probing
  with dated events, then compare β and the out-of-sample R² before and after it.
- **Anonymisation:** remove dates, the names of chairs and governors, and explicit year markers,
  then score the documents again and compare the scores and β.
- **Recognition probe:** ask the model to date an anonymised statement. High accuracy means the
  model recognises the documents rather than reading them.

## 8. Retrieval agent (week 4)

- A question-answering agent over the corpus, for questions such as "how did the inflation
  wording change between 2021 and 2023?", answering with verbatim citations (document,
  paragraph).
- About 50 hand-written questions with reference answers and reference passages.
- Metrics: recall@k of the retrieval, citation faithfulness (does the cited passage support the
  claim?), and the rate of answers that refuse when they should.
- Retrieval stack: decide in week 4 between reusing `kbase` from AgenticEnv (pgvector, hybrid
  retrieval, provenance, at the cost of a Postgres setup) and a light in-repo index. The corpus is
  small, a few thousand documents.

## 9. Deliverables

- A 5–6 page research report (`reports/fomc.md` or PDF) covering the question, data, method, results, the
  look-ahead analysis and the limitations.
- A README with the main results table.
- A pipeline that can be rerun from scratch: `fetch → parse → score → evaluate → regress`.

## 10. Schedule

| Week | Deliverable |
|---|---|
| 1 | Ingestion of the Fed texts and FRED data, a versioned dataset, LM and FinBERT baselines |
| 2 | Pydantic extraction schema, LLM extraction, hand-labelled set, extraction metrics |
| 3 | Event study, out-of-sample test, placebos, cutoff and anonymisation tests |
| 4 | Retrieval agent and its evaluation, research report, final README |
