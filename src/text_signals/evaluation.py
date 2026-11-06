"""Agreement between two readers of the same documents, field by field (protocol §6.6).

A reader is a model or the reference. For each field, every document falls into one of four
cases: both readers filled it, both left it empty, or only one of them filled it. The metrics
keep the two questions apart:

- **presence**: do the readers agree on *whether* the text says something on the field?
- **value**, on the documents where both filled it: exact agreement, agreement within one step
  (ordinal scales), mean absolute error and Cohen's κ (quadratic weights for ordinal scales,
  so a two-step disagreement costs four times a one-step one).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass


def cohen_kappa(
    a: Sequence[Hashable], b: Sequence[Hashable], *, ordinal: bool = False
) -> float | None:
    """Cohen's κ between two label sequences; quadratic weights when ``ordinal``.

    Ordinal labels must be numbers. ``None`` when κ is undefined (fewer than two items, or a
    single category used by both readers, so that chance agreement is 1).
    """
    if len(a) != len(b):
        raise ValueError("the two readers must label the same items")
    n = len(a)
    if n < 2:
        return None
    cats = sorted(set(a) | set(b), key=repr)
    if len(cats) < 2:
        return None
    index = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[index[x]][index[y]] += 1 / n
    row = [sum(r) for r in observed]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    if ordinal:
        values = [float(c) for c in cats]  # type: ignore[arg-type]
        span = max(values) - min(values)

        def weight(i: int, j: int) -> float:
            return ((values[i] - values[j]) / span) ** 2
    else:

        def weight(i: int, j: int) -> float:
            return 0.0 if i == j else 1.0

    disagree_obs = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    disagree_exp = sum(weight(i, j) * row[i] * col[j] for i in range(k) for j in range(k))
    if disagree_exp == 0:
        return None
    return 1 - disagree_obs / disagree_exp


@dataclass(frozen=True)
class FieldAgreement:
    field: str
    n: int
    both_filled: int
    both_empty: int
    only_a: int
    only_b: int
    exact: float | None  # share of exact agreement where both filled
    within_one: float | None  # ordinal only
    mae: float | None  # ordinal only
    kappa: float | None  # on the documents where both filled

    @property
    def presence_agreement(self) -> float:
        return (self.both_filled + self.both_empty) / self.n if self.n else float("nan")


def field_agreement(
    field: str,
    a: Sequence[Hashable | None],
    b: Sequence[Hashable | None],
    *,
    ordinal: bool,
) -> FieldAgreement:
    if len(a) != len(b):
        raise ValueError("the two readers must label the same documents")
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    both_empty = sum(x is None and y is None for x, y in zip(a, b, strict=True))
    only_a = sum(x is not None and y is None for x, y in zip(a, b, strict=True))
    only_b = sum(x is None and y is not None for x, y in zip(a, b, strict=True))
    exact = within = mae = kappa = None
    if pairs:
        exact = sum(x == y for x, y in pairs) / len(pairs)
        kappa = cohen_kappa([x for x, _ in pairs], [y for _, y in pairs], ordinal=ordinal)
        if ordinal:
            gaps = [abs(float(x) - float(y)) for x, y in pairs]  # type: ignore[arg-type]
            within = sum(g <= 1 for g in gaps) / len(gaps)
            mae = sum(gaps) / len(gaps)
    return FieldAgreement(
        field, len(a), len(pairs), both_empty, only_a, only_b, exact, within, mae, kappa
    )
