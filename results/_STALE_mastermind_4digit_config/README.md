# Stale: Mastermind results on the pre-ORBIT-paper task config

`hexa_orbit.mastermind.TASK_CONFIG` was changed to match the ORBIT paper:
`code_length` 3, `num_numbers` 6 (120 codes), from 4/8 (1680 codes).

Everything listed below was measured on 4/8 and is not comparable to any
current Mastermind number. Kept for provenance; cite none of it.

| artifact | what it was |
|---|---|
| `results/orbit_mastermind_baseline_oss` | Design B no-bank baseline, n=256. ep1 .031 / ep3 .367 / any .379 |
| `results/orbit_mastermind_hexa_v3` | Design B HExA, the "+17.6 ep3 / +16.4 any" result |
| `results/design_c/mastermind_baseline` | partial (n=226), never completed |

`orbit_mastermind_hexa_v3` was additionally compromised: its bank changed in
only rounds 0, 8 and 14 of 20 -- the teacher-truncation signature from
`HEXA_TEACHER_MAX_TOKENS=16384`. So the headline it carried came from a bank
that was frozen for 17 of 20 rounds.

Reruns on the current config: `scripts/run_orbit_mastermind_rerun.sh` (Design B)
and `scripts/run_design_c_mastermind_redo.sh` (Design C Phase 3+4). The matched
no-bank baseline is `results/design_c/phase4/mastermind_nobank/run{0,1,2}`.
