import numpy as np
import pandas as pd

from text_signals.sources.fomc.claude_signal import compare


def test_compare_ranks_an_informative_score_and_reports_each_reader() -> None:
    rng = np.random.default_rng(13)
    n = 200
    tone = rng.normal(0, 1, n)
    table = pd.DataFrame(
        {
            "doc_id": [f"d{i:03d}" for i in range(n)],
            "release_date": pd.date_range("1995-01-01", periods=n, freq="45D").astype(str),
            "llm_tone": tone + rng.normal(0, 0.3, n),
            "anon_tone": tone + rng.normal(0, 0.3, n),
            "claude_tone": rng.normal(0, 1, n),  # uninformative
            "MP1_bp": rng.normal(0, 5, n),
            "lm_net_negative": rng.normal(0, 0.01, n),
            "finbert_net_negative": rng.normal(0, 0.2, n),
            "in_main_sample": True,
        }
    )
    table["d_UST2Y_30m_bp"] = 2 * tone + rng.normal(0, 1, n)
    rows = {row["score"]: row for row in compare(table).to_dict("records")}
    assert set(rows) >= {"llm_tone", "anon_tone", "claude_tone"}
    assert float(rows["llm_tone"]["p_hac"]) < 0.01 < float(rows["claude_tone"]["p_hac"])
    assert float(rows["llm_tone"]["oos_r2_vs_baseline"]) > float(
        rows["claude_tone"]["oos_r2_vs_baseline"]
    )
    assert all(int(row["n"]) == n for row in rows.values())
