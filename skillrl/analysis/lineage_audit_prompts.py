"""Judge prompts for the skill-lineage audit (two-measure split).

The audit asks an LLM judge to annotate what happened to each entry between two
versions of a guidance document (= two `skill_bank_evolving_N.json` snapshots).

TWO SEPARATE MEASURES. They are never summed and never reported under one
heading:

  Measure A -- INCORPORATION (composition).
      A later entry carries content traceable to a SECOND, different earlier
      entry, over and above its own predecessor. This is the measure that
      speaks to hierarchical composition.

  Measure B -- ABSTRACTION (generalisation).
      A later entry is a more general form of its SINGLE predecessor. This is
      a weaker, separate claim: entries abstract over time. It is NOT
      composition.

Each measure has an explicit counter-direction (FRAGMENTATION for A, NARROWING
for B) so the instrument cannot only move one way.

BLINDING CONTRACT -- the caller MUST enforce all of this before formatting:
  1. Strip real skill_ids. Relabel earlier entries P1..Pk, later entries C1..Cm.
  2. Shuffle both lists independently. Position correlates with lineage in the
     raw banks; an unshuffled list lets the judge match by index and silently
     defeats the negative controls.
  3. Drop `confidence` and `generation` (both correlate with round number).
  4. Keep `source_seeds` -- legitimate evidence -- but keep it IDENTICALLY
     across every condition, including the controls.
  5. Never state the research hypothesis, the project name, or that
     composition is expected or hoped for.
  6. Run at low temperature, fresh context per call.

CONDITIONS these prompts serve:
  TRANSITION_AUDIT_PROMPT     main; a pair of versions (k=1, 4 or 8 apart),
                              plus NEG-APPEND and NEG-MISMATCH controls
  HISTORY_ATTRIBUTION_PROMPT  Measure A with donors from ANY earlier version
  VERIFY_EVENT_PROMPT         pass-2 adversarial check of one proposed event
  DIRECTION_BLIND_PROMPT      control: is the generality verdict just reading
                              verbosity?
"""

# --------------------------------------------------------------------------
# Formatting helper. The caller supplies already-anonymised, already-shuffled
# entries as dicts with: label, title, principle, when_to_apply, example,
# source_seeds.
# --------------------------------------------------------------------------

ENTRY_TEMPLATE = """\
[{label}]
  title:         {title}
  principle:     {principle}
  when_to_apply: {when_to_apply}
  example:       {example}
  observed_on:   {source_seeds}
"""


def format_entries(entries) -> str:
    return "\n".join(
        ENTRY_TEMPLATE.format(
            label=e["label"],
            title=e.get("title") or "",
            principle=e.get("principle") or "",
            when_to_apply=e.get("when_to_apply") or "",
            example=e.get("example") or "(none)",
            source_seeds=e.get("source_seeds") or [],
        )
        for e in entries
    )


# --------------------------------------------------------------------------
# 1. MAIN TRANSITION AUDIT
# --------------------------------------------------------------------------

TRANSITION_AUDIT_PROMPT = """\
You are auditing how a technical guidance document was revised. The document is
a list of entries, each giving advice for solving a physics puzzle. You are
shown an EARLIER version and a LATER version. The entries have been relabelled
and reordered, so labels and ordering carry no information -- judge only by
content.

EARLIER VERSION
{earlier_block}

LATER VERSION
{later_block}

Your job has three parts: establish lineage, then annotate two independent
relations, then report the counter-directions. Work only from the text shown.

-------------------------------------------------------------------------
PART 1 -- LINEAGE

For every later entry, assign exactly ONE lineage label:

  UNCHANGED      -- same entry as one earlier entry, text materially identical
  EDITED_FROM    -- same entry as one earlier entry, text changed
  SPLIT_FROM     -- one earlier entry has become two or more later entries;
                    this later entry is one of them
  NEW            -- no earlier entry is a predecessor

"Same entry" means same subject and same advice, not merely the same topic.
Two entries that both discuss drop height are not the same entry unless they
give the same advice about it.

For every earlier entry that no later entry claims as a predecessor, record it
as REMOVED.

Most entries are UNCHANGED or EDITED_FROM. NEW and SPLIT_FROM are less common.

-------------------------------------------------------------------------
PART 2 -- MEASURE A: INCORPORATION

An incorporation is when a later entry C has taken content from a DONOR entry D
that is NOT C's own predecessor.

Report an incorporation only if ALL FOUR hold:

  A1. C has a dominant predecessor P (its lineage label is EDITED_FROM or
      SPLIT_FROM, and P is that predecessor).
  A2. C contains a SPECIFIC element -- a numeric value, a named condition, a
      mechanism, a named case or regime, a variable the advice depends on --
      that also appears in some earlier entry D, where D is not P.
  A3. That element does NOT appear anywhere in P.
  A4. You can quote the element verbatim from D and quote the corresponding
      text verbatim from C.

A3 IS THE TEST THAT MATTERS. If the element was already in P, C did not take it
from D, and this is not an incorporation no matter how similar D and C look.

These do NOT qualify:
  - C and D discuss the same object, parameter or tool.
  - C and D share a value that P also contained.
  - C and D are both about the same puzzle.
  - You think C's author was probably aware of D.

D may still be present in the LATER version. D does not have to have been
removed. A later entry may incorporate from more than one donor. Most later
entries incorporate from nothing; reporting none is a normal and expected
result.

-------------------------------------------------------------------------
PART 2b -- MEASURE A, THEMATIC FORM: ABSORPTION

Part 2 asks about a specific element moving between entries. Content can also
move WITHOUT any quotable constant: an entry is dropped because what it taught
is now handled by another entry that grew to cover it.

For EVERY entry you listed as REMOVED in Part 1, decide between exactly two
possibilities:

  ABSORBED_INTO(Cj) -- the advice that entry gave is now available to a reader
      of surviving entry Cj. A reader who follows Cj would do what the removed
      entry told them to do, in the situation it was about.

  DELETED -- the advice is simply gone. No surviving entry tells the reader
      what that entry told them. It was dropped as wrong, redundant with
      something that already existed unchanged, or no longer relevant.

DELETED IS THE DEFAULT. Choose ABSORBED_INTO only if you can point to the
specific part of Cj that now does the removed entry's job, and quote it. It is
not enough that Cj is about the same topic, or that Cj is the longest entry, or
that Cj is where you would have put it. Ask: has Cj CHANGED to take this on, or
was it always like that? If Cj carries the removed entry's job only because it
was already general enough before the removal, that is DELETED, not absorption
-- say so in "covering_entry_changed": false.

Report absorption even when the overlap is thematic rather than numeric. A
removed entry whose lesson was "small parameter changes can flip the outcome"
is absorbed if a surviving entry now tells the reader to expect and test for
that, in whatever words.

-------------------------------------------------------------------------
PART 3 -- MEASURE B: ABSTRACTION

An abstraction is when a later entry is a MORE GENERAL form of its SINGLE
predecessor. This is about one entry becoming broader. It is unrelated to
Part 2 and must not be reported there.

Two mechanisms only:

  CONSTANT_TO_VARIABLE -- a fixed value in P is replaced in C by the quantity
      it was an instance of, so the advice now applies at any value.
      e.g. "check that height 2.5 is reachable" -> "check that the chosen
      height is reachable"

  SCOPE_WIDENING -- the set of situations the advice applies to is strictly
      larger in C than in P.
      e.g. advice about one specific parameter -> the same advice stated for
      any parameter

A LONGER ENTRY IS NOT A MORE GENERAL ENTRY. None of the following is
abstraction: added caveats, added examples, added hedging, more careful
wording, more detail, extra conditions, a longer list of values. Several of
those are the opposite -- see Part 4.

Quote P and quote C for every abstraction you report.

-------------------------------------------------------------------------
PART 4 -- COUNTER-DIRECTIONS

Report these with the same care and the same quoting standard:

  FRAGMENTATION -- one earlier entry has become two or more later entries,
      each covering part of what it covered. (The counter-direction of A.)

  NARROWING -- a later entry is a MORE SPECIFIC form of its predecessor: a
      variable replaced by a fixed value, or the set of situations it applies
      to made strictly smaller. (The counter-direction of B.)

If you find more narrowing than abstraction, or more fragmentation than
incorporation, say so. That is a legitimate result.

-------------------------------------------------------------------------
OUTPUT -- a single JSON object, no prose outside it:

{{
  "lineage": [
    {{"later": "C1", "label": "EDITED_FROM", "predecessor": "P3",
      "note": "<one clause: what changed>"}}
  ],
  "removed": ["P5"],
  "incorporations": [
    {{"later": "C1", "predecessor": "P3", "donor": "P7",
      "element": "<the specific element taken>",
      "quote_donor": "<verbatim from P7>",
      "quote_later": "<verbatim from C1>",
      "absent_from_predecessor": "<why this element is nowhere in P3>",
      "donor_still_present": true,
      "confidence": "high" | "medium" | "low"}}
  ],
  "absorptions": [
    {{"removed_entry": "P5", "disposition": "ABSORBED_INTO" | "DELETED",
      "covering_entry": "<later label, or null if DELETED>",
      "what_it_taught": "<the job the removed entry did, one clause>",
      "quote_removed": "<verbatim from the removed entry>",
      "quote_covering": "<verbatim from the covering entry that now does it>",
      "covering_entry_changed": true | false,
      "confidence": "high" | "medium" | "low"}}
  ],
  "abstractions": [
    {{"later": "C2", "predecessor": "P1",
      "mechanism": "CONSTANT_TO_VARIABLE" | "SCOPE_WIDENING",
      "quote_predecessor": "<verbatim from P1>",
      "quote_later": "<verbatim from C2>",
      "why_not_just_longer": "<one clause>",
      "confidence": "high" | "medium" | "low"}}
  ],
  "fragmentations": [
    {{"predecessor": "P4", "later": ["C5", "C6"],
      "quote_predecessor": "<verbatim>", "quotes_later": ["<verbatim>", "<verbatim>"]}}
  ],
  "narrowings": [
    {{"later": "C7", "predecessor": "P2",
      "quote_predecessor": "<verbatim>", "quote_later": "<verbatim>"}}
  ]
}}

Every quote must appear character-for-character in the entry you attribute it
to. Do not paraphrase inside a quote field. Empty lists are valid and common.
"""


# --------------------------------------------------------------------------
# 2. MEASURE A ACROSS THE WHOLE HISTORY (donors from any earlier version)
# --------------------------------------------------------------------------

HISTORY_ATTRIBUTION_PROMPT = """\
You are auditing one entry in a technical guidance document that was revised
many times. Below is the FINAL form of the entry, then its OWN revision history
(the same entry as it appeared in earlier versions), then a POOL of other
entries that existed in those earlier versions. Entries are relabelled and the
pool is in random order.

FINAL ENTRY
{final_block}

ITS OWN REVISION HISTORY (earliest first)
{own_history_block}

POOL OF OTHER ENTRIES FROM EARLIER VERSIONS
{pool_block}

QUESTION: does the final entry contain content that came from the POOL rather
than from its own history?

Report a contribution only if ALL of the following hold:

  1. You can point to a SPECIFIC element in the final entry -- a numeric value,
     a named condition, a mechanism, a named case or regime, or a quantity the
     advice now depends on.
  2. That element appears in a pool entry D.
  3. That element appears NOWHERE in the entry's own revision history before
     the version where it first shows up in the entry.
  4. You can quote it verbatim from D and from the final entry.

Condition 3 is the test. The entry's own history is the default explanation for
everything in it. Only content that its own history cannot account for may be
attributed to the pool.

Shared vocabulary is not a contribution. Discussing the same object, the same
tool or the same puzzle is not a contribution. A value that already appears in
the entry's own history is not a contribution even if a pool entry also has it.

An entry whose content is fully explained by its own history is the normal
case. Reporting no contributions is an expected result.

OUTPUT -- a single JSON object, no prose outside it:

{{
  "contributions": [
    {{"donor": "<pool label>",
      "element": "<the specific element>",
      "quote_donor": "<verbatim from the donor>",
      "quote_final": "<verbatim from the final entry>",
      "first_appears_in_own_history_at": "<version label, or 'never before'>",
      "why_own_history_cannot_account_for_it": "<one or two clauses>",
      "confidence": "high" | "medium" | "low"}}
  ],
  "verdict": "SELF_CONTAINED" | "HAS_CONTRIBUTIONS"
}}
"""


# --------------------------------------------------------------------------
# 3. PASS-2 ADVERSARIAL VERIFICATION
# --------------------------------------------------------------------------

VERIFY_EVENT_PROMPT = """\
A previous reviewer claimed that one entry in a technical document took content
from another entry. Your job is to find out whether that claim is wrong. Assume
it is wrong until the text forces you to conclude otherwise.

PREDECESSOR (P) -- the later entry's own earlier form
{predecessor_block}

DONOR (D) -- the entry the content was supposedly taken from
{donor_block}

LATER ENTRY (C)
{later_block}

THE CLAIM
  Element said to have moved from D into C: {element}
  Quoted from D: {quote_donor}
  Quoted from C: {quote_later}

Check each of these in order and stop at the first failure:

  1. VERBATIM. Does the quote from D appear character-for-character in D? Does
     the quote from C appear character-for-character in C? If either does not,
     the claim fails.

  2. ABSENT FROM P. Is the element genuinely absent from P -- from its title,
     principle, when_to_apply and example? If any form of it is present in P,
     the claim fails: C got it from its own predecessor, not from D.

  3. SPECIFIC. Is the element specific enough that two independent authors
     would be unlikely to write it by coincidence? A shared parameter name, a
     shared tool name, a common round number, or a shared topic is NOT
     specific. A particular value tied to a particular condition, a named
     regime, or a stated dependency between two quantities IS specific.

  4. SAME MEANING. Does the element mean the same thing in D and in C, or has
     it been reused for a different purpose? If the meaning differs, the claim
     fails.

The most common reason a claim like this is wrong: C and D are both about the
same part of the puzzle, so they share vocabulary and values, and the reviewer
mistook that overlap for transfer.

OUTPUT -- a single JSON object, no prose outside it:

{{
  "verbatim_ok": true | false,
  "absent_from_predecessor": true | false,
  "specific": true | false,
  "same_meaning": true | false,
  "verdict": "CONFIRMED" | "REJECTED",
  "failing_check": "<name of the first check that failed, or null>",
  "reason": "<one or two sentences>"
}}

Return CONFIRMED only if all four checks pass.
"""


# --------------------------------------------------------------------------
# 4. DIRECTION-BLIND CONTROL (is the generality verdict reading verbosity?)
# --------------------------------------------------------------------------

DIRECTION_BLIND_PROMPT = """\
Below are two versions of the same entry from a technical guidance document.
They are labelled VERSION ONE and VERSION TWO in a random order that carries no
information about which was written first.

VERSION ONE
{version_one_block}

VERSION TWO
{version_two_block}

Which version is MORE GENERAL -- that is, which one applies to a strictly wider
set of situations?

Judge applicability only. Length, level of detail, number of examples, number
of caveats, and confidence of tone are all irrelevant. A short blunt rule can
be more general than a long qualified one, and a longer entry is very often
narrower because the added text restricts when the advice applies.

If neither applies to a wider set of situations than the other, answer EQUAL.
EQUAL is a common and acceptable answer.

OUTPUT -- a single JSON object, no prose outside it:

{{
  "more_general": "ONE" | "TWO" | "EQUAL",
  "basis": "<the situations covered by the broader one that the other excludes,
             or why they are equal>",
  "word_count_one": <int>,
  "word_count_two": <int>,
  "confidence": "high" | "medium" | "low"
}}
"""

# --------------------------------------------------------------------------
# 5. PASS-2 FOR ABSORPTION (counter-hypothesis: it was just deleted)
# --------------------------------------------------------------------------

VERIFY_ABSORPTION_PROMPT = """\
A previous reviewer claimed that an entry removed from a technical guidance
document was not simply deleted, but that its job is now done by another entry.
Your job is to find out whether that claim is wrong. Assume it is wrong until
the text forces you to conclude otherwise.

THE REMOVED ENTRY
{removed_block}

THE ENTRY SAID TO HAVE TAKEN OVER ITS JOB, IN THE EARLIER VERSION
{covering_before_block}

THE SAME ENTRY IN THE LATER VERSION
{covering_after_block}

THE CLAIM
  What the removed entry taught: {what_it_taught}
  Quoted from the removed entry: {quote_removed}
  Quoted from the later form of the covering entry: {quote_covering}

Check each of these in order and stop at the first failure:

  1. VERBATIM. Do both quotes appear character-for-character in the entries
     they are attributed to? If not, the claim fails.

  2. ACTUALLY COVERED. Would a reader who follows ONLY the later form of the
     covering entry do what the removed entry told them to do, in the
     situation the removed entry was about? If the covering entry merely
     mentions the same topic, or is about the same parameter, the claim fails.

  3. THE COVERING ENTRY CHANGED. Compare the covering entry's EARLIER and
     LATER forms. Did it take this job on, or did it already do it before the
     removal? If it already did it, then the removed entry was redundant and
     was deleted -- the claim fails. This is the most common way the claim is
     wrong.

  4. NOT A SHRINK. Is the removed entry's advice preserved, rather than
     narrowed away to a special case or contradicted? If the later form gives
     opposing advice, the claim fails.

OUTPUT -- a single JSON object, no prose outside it:

{{
  "verbatim_ok": true | false,
  "actually_covered": true | false,
  "covering_entry_changed": true | false,
  "not_a_shrink": true | false,
  "verdict": "CONFIRMED" | "REJECTED",
  "failing_check": "<name of the first check that failed, or null>",
  "reason": "<one or two sentences>"
}}

Return CONFIRMED only if all four checks pass.
"""
