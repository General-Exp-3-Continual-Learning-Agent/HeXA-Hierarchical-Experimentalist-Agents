"""Episodic RAG and MemRL on CORE's four single-turn tasks.

Neither baseline ships with CORE's own released repo (`hexa_core/CORE_ENV_RESULTS.md`
§6 -- their numbers for these methods are read off the paper, not rerun on our
provider). This package is the missing rerun: it points the same generic memory
machinery `baselines/memory/` already built for InterPhyre at
`hexa_core.adapter.Problem` instead, so Matchstick / MathGAP / ZebraLogic /
Tower-of-Hanoi get a provider-matched, budget-matched Episodic RAG and MemRL arm
of their own, directly comparable to the HeXA numbers already in
`hexa_core/CORE_ENV_RESULTS.md`.
"""
