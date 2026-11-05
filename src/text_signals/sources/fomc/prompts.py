"""Versioned prompts for the FOMC statement extraction (protocol §5.2).

A prompt is never edited in place: a change is a new version, so that every stored output can be
traced to the exact instructions that produced it.

The model sees the statement text only: no title, no date, no URL. The text itself may still
contain dates and names; the anonymisation test (protocol §6.7) removes them.
"""

from __future__ import annotations

from text_signals.llm import Prompt

STATEMENT_V1 = Prompt(
    version="fomc-statement-prompt-v1",
    system=(
        "You read one statement of the Federal Open Market Committee (FOMC) and fill a JSON "
        "form that describes it. Rules:\n"
        "1. Use only the text of the statement. Do not use what you may know about the date, "
        "the economy at the time, later events, or how markets reacted.\n"
        "2. Every scale runs from -2 to +2. Positive means hawkish (pointing to higher policy "
        "rates); negative means dovish (pointing to lower rates). Follow the definition given "
        "for each field in the schema.\n"
        "3. When the statement says nothing on a field, set its value and its evidence to null. "
        "Do not guess.\n"
        "4. For every value you give, copy into 'evidence' the words of the statement that "
        "support it, exactly as written: one sentence or part of one. Never paraphrase.\n"
        "5. Judge the statement as a whole as written, not single words: 'elevated' inflation "
        "is a concern, while 'elevated' uncertainty is not about inflation."
    ),
    user_template="Statement:\n\n{text}",
)
