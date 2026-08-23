# Abandoned at round 5/30 — do not report this run

Cancelled deliberately (job 62865576), not crashed. First MathGAP run:

    PIPELINE_MAXTOK=60000  n_train=60  round_target=30  budget=600
    no MAXINJECT cap, no TRUNCGUIDE

No eval was run, so there is no accuracy number here and none should be quoted.

## Why it was killed

The evolver learned a **wrong semantics** for the task's central statement form in
its first distillation, and every later round was solved through it.

Top-ranked skill in the round-3 bank (confidence 0.52, so injected first):

> **Treat 'has more than' in equality statements as absolute difference** —
> interpret both sides as magnitudes: |X - Y| = |A - B|

A second skill at 0.50 ("Absolute difference for 'difference between'") repeated it.
This is incorrect: the relation is *signed*. Verified against the item whose
published `text_target` is 23
(`mathgap_num_20-seed_0-d_5-l_20-irr-a_1-s_0-idx_0`):

- signed, `(Adolfo - Taylor) = (James - Myles)` → `-5 = James - 15` → James = 10;
  carrying the chain gives Katheryn = 6, then `Micheal gives 17` → **23 = target**.
- absolute, `|Adolfo - Taylor| = |James - Myles|` → James in {10, 20}. Every
  comparison-transfer forks in two; on Katheryn's chain alone Mikaela in {13,1},
  Maya in {20,10}, Marisa in {7,3}, so the answer is no longer determined.

Those statements are the backbone of the long items (2,609 across the dataset), so
discarding the sign corrupts exactly the reasoning the task is testing.

Compounding it: the one skill encoding the genuinely load-bearing rule,
`Process statements in correct order`, sat at confidence **0.10**, last in the bank.
Transfer order does matter — using Robert's pre-transfer count yields 26, not 23.

## Measured effect

Truncation was only 2%, so unlike Matchstick this was not a length artifact.

| rnd | acc | b_q | lift | added_tok |
|---|---|---|---|---|
| 0 | 0.10 | 0.10 | +0.000 | 0 (empty bank) |
| 1 | 0.35 | 0.50 | -0.150 | 1012 |
| 2 | 0.15 | 0.10 | +0.050 | 1028 |
| 3 | 0.10 | 0.60 | -0.500 | 996 |
| 4 | 0.30 | 0.70 | -0.400 | 999 |

Over the four injected rounds: accuracy 0.225 vs b_q 0.475, lift **-0.250**. Treat
that number as indicative only — 2 problems per round means each `b_q` is an average
of two 5-sample values quantised to 0.2, and training lift has a documented
counterexample as an eval predictor (WEAKNESSES.md, Hanoi run0: lift +0.133, eval
delta 0.000). The bank-content diagnosis above does not depend on it.

## Fix applied before the rerun

`CORE_TASK_DESCRIPTIONS["mathgap"]` described the comparison-transfer form without
saying it was signed, leaving the evolver to guess. It now states the signed reading
plainly — `(A - B) = (C - D)`, order preserved, either side may be negative — and
notes that transfers take effect in the order stated.

**The first version of that repair over-reached** and was itself abandoned: it also
argued that the absolute reading is incorrect and supplied an answer-space prior
("every answer lies between 10 and 30"). See `../run0_privinfo_ABANDONED` for the
audit and the trimmed final wording.

Only the `mathgap` entry changed; `matchstick`, `zebralogic` and `tower-of-hanoi`
descriptions are byte-identical (hashes checked before and after), and the evolution
prompt interpolates a single `{level_description}` looked up by the running task, so
no other task's prompts are affected.

Superseded by `../run0`.
