"""Episodic-memory baselines on DiscoverPhysics.

`baselines/memory/` holds the methods (`rag.py`, `memrl.py`, `store.py`) and is
untouched; everything DiscoverPhysics-specific is here, the same split
`baselines/orbit/` uses for ORBIT.

The arms run under HeXA's own DiscoverPhysics protocol so they land paired,
per seed, against the existing no-bank baselines in
`discovery-agents/hexa_discovery/data/runs/`:

  within-world   offline init on the world's 5 bank-free episodes (seeds
                 100-104), 5 build rounds x 3 seeds (200-214), freeze,
                 evaluate on seeds 300-314
  cross-world    build over 9 training worlds x seeds 100-102 in HeXA's own
                 world order, freeze, evaluate on hubble seeds 100-114
"""
