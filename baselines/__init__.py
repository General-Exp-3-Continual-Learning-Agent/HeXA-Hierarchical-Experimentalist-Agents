"""Memory baselines for InterPhyre: Episodic RAG and MemRL.

Both are lifelong/runtime learners that share one seed-stream loop
(`baselines.runner.stream`) and one prompt-injection seam
(`baselines.runner.memory_runner`), mirroring how HeXA injects its skill bank
in `skillrl/runner/augmented_runner.py`.

Episodic RAG follows the CORE paper's definition (retrieve the nearest past
successful and nearest past unsuccessful rollout, verbatim, with verifier
feedback).  MemRL follows the official release (MemTensor/MemRL) with the
ALFWorld configuration: Intent-Experience-Utility triplets, two-phase
retrieval, Monte-Carlo utility updates.
"""
