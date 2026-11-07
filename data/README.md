# data/

Nothing in this directory is versioned except this file. The code in `src/text_signals/` rebuilds
everything.

```
raw/        files exactly as fetched (HTML, PDF, CSV) + a manifest (url, fetched_at, sha256)
interim/    parsed text, one record per document, keyed by release timestamp
processed/  scores, market variables, labelled sets ready for analysis
```

`labels/` is the exception: research inputs that cannot be regenerated, versioned in git.

```
labels/fomc_statements_sample_v1.csv     extraction evaluation sample (65 statements)
labels/fomc_confirm_sample_v1.csv        confirmation sample (30 statements)
labels/fomc_statements_v1/practice/      the five practice labels (not used)
labels/fomc_counterfactual_v1.json       counterfactual guidance edits of the look-ahead test B
labels/fomc_qa_v1.json                   retrieval agent questions, references and split
labels/fomc_qa_v1_split.json             development / test split
labels/fomc_qa_v1_judgments_test.json    judgments of the test answers, with reasons
```
