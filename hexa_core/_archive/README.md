# Archived: CORE-parity machinery

Moved here, not deleted, when the project stopped claiming an exact comparison with
CORE (see the SCOPE CHANGE box at the top of `../PHASE3_DESIGN.md`). CORE now supplies
only the tasks -- splits, prompts, verifiers. Nothing in this directory is imported by
the live code path; `hexa_core/` is untracked in git, so archiving rather than deleting
is the only way to keep these recoverable.

| file | why it is here |
|---|---|
| `logging_schema.py` | Wrote CORE's `validation.jsonl` / `_summary.jsonl` so their plotting code could read our runs. Had **zero importers** even before the scope change. |
| `nim_client.py` | Thin wrapper over CORE's `NvidiaClient`, kept so the NIM provider could be A/B'd against local vLLM. We serve locally and never construct it. |
| `baseline_runner.py` | The Phase 2 reproduction gate: no-memory eval versus the paper's published numbers, with PASS*/MISS* verdicts. Superseded by the paired no-memory arm inside `passive_loop`, which runs the same problems and seeds through the same code path. Its seed helpers were the one live dependency and now live in `hexa_core/seeds.py`. |
| `run_repro_gate.py` | The driver for the above. Its own docstring says it is "a calibration run, not a reproduction". |
| `test_*.py` | Suites for the above. |

Delete this directory once you are sure none of it is wanted.
