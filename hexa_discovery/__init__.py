"""HeXA <-> DiscoverPhysics bridge (Axis B: cross-world skill evolution).

Replaces HeXA's four InterPhyre-specific seams while reusing the rest of the
skill distillation/evolution engine unchanged:
  - reward.py                    graded blend reward (replaces iteration reward)
  - traj_adapter.py              DiscoveryAgent run -> HeXA trajectory JSON
  - discovery_runner.py          run episodes + inject skills (replaces run_skill_augmented)
  - teacher_prompts_discovery.py cross-world teacher prompts + formatter
  - cross_world_loop.py          world-outer evolving loop + transfer eval
  - run_offline.py               offline seed-trajectory bootstrap
"""
