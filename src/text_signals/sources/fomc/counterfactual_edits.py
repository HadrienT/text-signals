"""The hand-written counterfactual edits of protocol §6.7 (test B), and the builder of
``data/labels/fomc_counterfactual_v1.json``.

Rewrites are keyed by the start of the original passage; identical passages get identical
rewrites. Written by Claude (claude-opus-5-5) on 2026-11-07, before any edited text was read.

Usage: ``python -m text_signals.sources.fomc.counterfactual_edits``.
"""

from __future__ import annotations

import json

import pandas as pd

from text_signals.sources.fomc import counterfactual as cf
from text_signals.sources.fomc.extract import all_docs
from text_signals.sources.fomc.extraction import StatementExtraction

GEN_H_POST = (
    "The Committee anticipates that increases in the target range will be appropriate in "
    "coming meetings, and it will act forcefully to return inflation to 2 percent."
)
GEN_D_POST = (
    "The Committee anticipates that further reductions in the target range will be "
    "appropriate in coming meetings, and it expects to keep rates low to support employment."
)

# key: start of the original passage -> (hawkish, dovish, neutral)
R: dict[str, tuple[str, str, str]] = {
    "Nonetheless, to date the easing of demand pressures": (
        "Nonetheless, the Committee judges that further firming of policy will be needed in "
        "coming months to contain heightened inflation pressures in the foreseeable future, and "
        "it is prepared to act forcefully.",
        "Nonetheless, the Committee judges that the easing of demand pressures will allow it to "
        "hold the federal funds rate at its current level for an extended period, and it is "
        "prepared to ease if growth weakens further.",
        "Even so, the easing of demand pressures so far has not been enough to change the "
        "Committee's judgment that, given its long-run goals of price stability and sustainable "
        "economic growth and the information now available, the risks remain weighted mainly "
        "toward conditions that may generate heightened inflation pressures in the foreseeable "
        "future.",
    ),
    "The associated easing of pressures on labor": (
        "The Committee nonetheless anticipates that a firming of policy will likely be needed "
        "soon to keep inflation contained.",
        "With inflation contained, the Committee expects to ease policy further in coming months "
        "and to keep rates low for an extended period.",
        "The resulting easing of pressures in labor and product markets should keep inflation "
        "contained.",
    ),
    "The Committee expects the rate of increase of final demand": (
        "The Committee expects final demand to strengthen over coming quarters and anticipates "
        "that it will need to begin raising the federal funds rate soon and at a steady pace.",
        "The Committee expects that weak final demand will warrant keeping the federal funds rate "
        "at its current low level for an extended period, and further easing if needed.",
        "The Committee anticipates that growth of final demand will pick up in coming quarters, "
        "helped partly by robust underlying productivity growth, though how much it will "
        "strengthen remains uncertain.",
    ),
    "The current accommodative stance of monetary policy, coupled": (
        "With the business climate improving, the Committee anticipates that the current "
        "accommodative stance will soon need to be withdrawn through a series of increases in "
        "the federal funds rate.",
        "The Committee anticipates keeping the federal funds rate at its current accommodative "
        "level for an extended period and stands ready to ease further to foster an improving "
        "business climate.",
        "The present accommodative stance of monetary policy, together with still-robust "
        "underlying productivity growth, should be enough to support a better business climate "
        "over time.",
    ),
    "The Committee believes that, even after this action": (
        "The Committee believes that, even after this action, the stance of monetary policy "
        "remains too accommodative. It anticipates that substantial further increases in the "
        "federal funds rate will be needed promptly to maintain price stability.",
        "The Committee believes that, after this action, no further firming is likely to be "
        "needed for some time. It expects to keep the federal funds rate at its new level for an "
        "extended period and would reverse the action if economic prospects weaken.",
        "The Committee judges that, even after this action, monetary policy remains "
        "accommodative and, together with robust underlying productivity growth, continues to "
        "support economic activity. Even so, the Committee will respond to changes in the "
        "economic outlook as needed to meet its obligation to maintain price stability.",
    ),
    "The Committee judges that some further policy firming may yet be needed": (
        "The Committee judges that substantial further policy firming will be needed at coming "
        "meetings to address inflation risks and that it should proceed promptly, whatever the "
        "evolution of the economic outlook implied by incoming information.",
        "The Committee judges that no further policy firming is likely to be needed and expects "
        "to hold the federal funds rate at its current level for an extended period, easing if "
        "the economic outlook implied by incoming information weakens.",
        "The Committee judges that some additional policy firming may still be needed to address "
        "inflation risks but stresses that how much firming occurs, and when, will depend "
        "importantly on how the economic outlook evolves as implied by incoming information.",
    ),
    "The Committee is monitoring the situation and is prepared": (
        "The Committee judges that inflation remains its predominant concern and expects that "
        "further increases in the federal funds rate will be needed despite the disruptions in "
        "financial markets.",
        "The Committee is prepared to lower the federal funds rate promptly and substantially, "
        "and to keep it low for an extended period, to offset the adverse effects of the "
        "disruptions in financial markets.",
        "The Committee is watching the situation closely and stands ready to take action as "
        "needed to limit the adverse effects on the economy of the disruptions in financial "
        "markets.",
    ),
    "Today’s action, combined with the policy action taken in September": (
        "Following today’s action and the action taken in September, the Committee expects "
        "that the easing will need to be reversed soon, with increases in the federal funds rate "
        "as inflation risks build.",
        "Today’s action, combined with the policy action taken in September, is likely to be "
        "followed by further reductions in the federal funds rate, and the Committee expects to "
        "keep rates low for an extended period.",
        "Together with the policy action taken in September, today’s action should help "
        "prevent some of the adverse effects on the broader economy that might otherwise result "
        "from the disruptions in financial markets and support moderate growth over time.",
    ),
    "The Committee will continue to monitor economic and financial developments": (
        "The Committee expects that the recent easing will need to be reversed, and that "
        "increases in the federal funds rate will be required in coming months to ensure price "
        "stability.",
        "The Committee expects to reduce the federal funds rate further in coming months and to "
        "keep it at exceptionally low levels for an extended period to promote sustainable "
        "economic growth.",
        "The Committee will keep monitoring economic and financial developments and will act as "
        "needed to foster sustainable economic growth and price stability.",
    ),
    "The Committee will closely monitor incoming information on economic": (
        "The Committee will closely monitor incoming information and anticipates ending its "
        "purchases of Treasury and agency mortgage-backed securities promptly and beginning to "
        "raise the federal funds rate soon, at a steady pace.",
        "The Committee will closely monitor incoming information and will continue and, if "
        "needed, expand its purchases of Treasury and agency mortgage-backed securities, keeping "
        "the federal funds rate near zero for an extended period, until the outlook for the "
        "labor market has improved substantially.",
        "The Committee will watch incoming information on economic and financial developments "
        "closely in the coming months and will keep up its purchases of Treasury and agency "
        "mortgage-backed securities, using its other policy tools as appropriate, until the "
        "labor market outlook has improved substantially in a context of price stability.",
    ),
    "This action supports the Committee's view": (
        "The Committee anticipates that this action will be reversed soon and that increases in "
        "the target range will be appropriate at coming meetings as the expansion continues.",
        "The Committee anticipates that further reductions in the target range will be "
        "appropriate at coming meetings, and it expects to keep rates low for an extended period.",
        "This action is consistent with the Committee's view that a sustained expansion of "
        "economic activity, strong labor market conditions, and inflation near its symmetric 2 "
        "percent objective are the most likely outcomes, though uncertainties about this outlook "
        "remain.",
    ),
    "The Committee is closely monitoring developments and their implications": (
        "The Committee anticipates that this reduction will be reversed promptly and that "
        "increases in the target range will be appropriate in coming months to keep inflation in "
        "check.",
        "The Committee anticipates further reductions in the target range and expects to keep "
        "rates at very low levels for an extended period to support the economy.",
        "The Committee is watching developments and their implications for the economic outlook "
        "closely and will use its tools and act as appropriate to support the economy.",
    ),
    "The Committee will continue to closely monitor market conditions": (
        "The Committee will continue to closely monitor market conditions, and anticipates "
        "reducing its securities purchases soon and raising the federal funds rate at a steady "
        "pace thereafter.",
        "The Committee will continue to closely monitor market conditions, and expects to keep "
        "the federal funds rate near zero for an extended period while expanding its securities "
        "purchases as needed.",
        "The Committee will keep monitoring market conditions closely, and will decide on the "
        "appropriate pace of its securities purchases at upcoming meetings.",
    ),
    "The Committee expects to maintain this target range until it is confident": (
        "The Committee expects that increases in the target range will be appropriate soon and "
        "will proceed at a steady pace once the effects of recent events fade.",
        "The Committee expects to maintain this target range for an extended period, well beyond "
        "the point at which the economy has weathered recent events, and stands ready to provide "
        "further accommodation.",
        "The Committee anticipates keeping this target range in place until it is confident that "
        "the economy has come through recent events and is on track to meet its maximum "
        "employment and price stability goals.",
    ),
    "The Committee expects to maintain an accommodative stance of monetary policy until": (
        "The Committee expects that ongoing increases in the target range will soon be "
        "appropriate to remove policy accommodation.",
        "The Committee expects to maintain a highly accommodative stance of monetary policy for "
        "an extended period, well after these outcomes are achieved.",
        "The Committee anticipates keeping monetary policy accommodative until these outcomes "
        "have been achieved.",
    ),
    "With inflation having exceeded 2 percent for some time": (
        "With inflation having exceeded 2 percent for some time, the Committee expects that "
        "ongoing increases in the target range will be appropriate, beginning soon.",
        "Even with inflation having exceeded 2 percent for some time, the Committee expects to "
        "maintain this target range for an extended period, well after labor market conditions "
        "have reached maximum employment.",
        "Because inflation has been above 2 percent for some time, the Committee anticipates that "
        "keeping this target range will be appropriate until labor market conditions reach "
        "levels consistent with its assessments of maximum employment.",
    ),
    "The Committee anticipates that ongoing increases in the target range": (
        "The Committee anticipates that large and ongoing increases in the target range will be "
        "appropriate in order to attain a stance of monetary policy that is sufficiently "
        "restrictive to return inflation to 2 percent quickly.",
        "The Committee anticipates that no further increases will be needed and that reductions "
        "in the target range will be appropriate soon to support economic activity.",
        "The Committee expects that continued increases in the target range will be appropriate "
        "to reach a monetary policy stance restrictive enough to bring inflation back to 2 "
        "percent over time.",
    ),
    "The Committee does not expect it will be appropriate to reduce": (
        "The Committee expects that further increases in the target range will be appropriate "
        "until it has gained greater confidence that inflation is moving sustainably toward 2 "
        "percent.",
        "The Committee expects that a series of reductions in the target range will be "
        "appropriate soon, as it has gained confidence that inflation is moving sustainably "
        "toward 2 percent.",
        "The Committee does not anticipate that lowering the target range will be appropriate "
        "until it has more confidence that inflation is moving sustainably toward 2 percent.",
    ),
    "In considering additional adjustments": (
        GEN_H_POST,
        GEN_D_POST,
        "In weighing additional adjustments to the target range for the federal funds rate, the "
        "Committee will carefully evaluate incoming data, the evolving outlook, and the balance "
        "of risks.",
    ),
    "In considering the extent and timing": (
        GEN_H_POST,
        GEN_D_POST,
        "In weighing the extent and timing of additional adjustments to the target range for the "
        "federal funds rate, the Committee will carefully evaluate incoming data, the evolving "
        "outlook, and the balance of risks.",
    ),
    "The Committee will deliver price stability.": (
        "The Committee expects further increases in the target range to be appropriate.",
        "The Committee expects to ease policy further and to keep rates low.",
        "The Committee is committed to delivering price stability.",
    ),
    "Today's policy action will support a timelier return": (
        "The Committee expects that further increases in the target range will be appropriate to "
        "ensure a timely return to its 2 percent goal. The Committee will deliver price "
        "stability.",
        "The Committee expects that reductions in the target range will be appropriate soon and "
        "anticipates keeping rates low to support employment. The Committee remains attentive to "
        "price stability.",
        "Today's policy action will help inflation return sooner to the Committee's 2 percent "
        "goal. The Committee is committed to delivering price stability.",
    ),
    "The accumulation of more information will help": (
        "The Committee expects that further increases in the target range will be appropriate to "
        "ensure a timely return of inflation to its 2 percent goal.",
        "The Committee expects that reductions in the target range will be appropriate soon, and "
        "it anticipates keeping policy accommodative for an extended period.",
        "Gathering more information will help the Committee decide on the policy stance needed to "
        "ensure a timely return of inflation to its 2 percent goal.",
    ),
}


# Statements with another sentence about the future path of the policy rate (any lean), left
# untouched by the edit; judged by reading the texts before any edited reading (protocol §6.7).
OTHER_RATE_GUIDANCE = {
    "2004-06-30": "policy accommodation can be removed at a pace that is likely to be measured",
    "2014-07-30": "maintain the current target range ... for a considerable time",
    "2019-10-30": "timing and size of future adjustments to the target range",
    "2020-03-15": "timing and size of future adjustments to the stance of monetary policy",
    "2020-04-29": "timing and size of future adjustments to the stance of monetary policy",
    "2021-03-17": "expects it will be appropriate to maintain this target range until ...",
    "2021-11-03": "expects it will be appropriate to maintain this target range until ...",
    "2022-11-02": "In determining the pace of future increases in the target range",
    "2024-05-01": "In considering any adjustments to the target range",
    "2024-07-31": "In considering any adjustments to the target range",
}


def main() -> None:
    scores = pd.read_csv("data/processed/fomc/llm_scores_phi-4.csv")
    docs = {d.doc_id: d for d in all_docs()}
    orig = cf.original_readings()
    edits = []
    for doc_id, stratum in cf.draw_sample(scores):
        ext = StatementExtraction.model_validate(orig[doc_id].output)
        evidence = ext.forward_guidance.evidence
        assert evidence is not None
        passage = cf.guidance_text(docs[doc_id].text, evidence)
        assert passage is not None, doc_id
        keys = [k for k in R if passage.startswith(k)]
        assert len(keys) == 1, (doc_id, passage[:80], keys)
        h, d, n = R[keys[0]]
        edits.append(
            {
                "doc_id": doc_id,
                "release_date": docs[doc_id].release_date.isoformat(),
                "stratum": stratum,
                "phi4_evidence": evidence,
                "original": passage,
                "hawkish": h,
                "dovish": d,
                "neutral": n,
                "other_rate_guidance": OTHER_RATE_GUIDANCE.get(
                    docs[doc_id].release_date.isoformat()
                ),
            }
        )
    out = {
        "version": cf.EDITS_VERSION,
        "author": "claude-opus-5-5",
        "written": "2026-11-07",
        "written_before_any_edited_reading": True,
        "seed": cf.SEED,
        "edits": edits,
    }
    with open(cf.EDITS, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(len(edits), "edits ->", cf.EDITS)


if __name__ == "__main__":
    main()
