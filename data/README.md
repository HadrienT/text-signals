# data/

Nothing in this directory is versioned except this file. The code in `src/fedspeak/` rebuilds
everything.

```
raw/        files exactly as fetched (HTML, PDF, CSV) + a manifest (url, fetched_at, sha256)
interim/    parsed text, one record per document, keyed by release timestamp
processed/  scores, market variables, labelled sets ready for analysis
```

The hand-labelled evaluation sets are the exception. They are research inputs that cannot be
regenerated, so they will be versioned under `labels/` (not ignored) when they exist.
