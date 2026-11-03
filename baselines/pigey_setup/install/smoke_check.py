"""Check the actual simulator stack without API calls or a policy checkpoint."""
import importlib.metadata
import os
from pathlib import Path

print('Loading API clients and simulation dependencies...', flush=True)
import anthropic
import imageio
import litellm
import numpy as np
import openai
from openpi_client import image_tools, websocket_client_policy
print('Loading LIBERO-PRO and robosuite...', flush=True)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

for package in ('numpy', 'torch', 'robosuite', 'mujoco', 'litellm', 'transformers', 'openpi-client'):
    print(f'{package}=={importlib.metadata.version(package)}', flush=True)
print(f'MUJOCO_GL={os.environ.get("MUJOCO_GL")}', flush=True)
registry = benchmark.get_benchmark_dict()
for suite_name in ('libero_goal', 'libero_goal_task'):
    suite = registry[suite_name]()
    print(f'{suite_name}: {suite.n_tasks} tasks', flush=True)
    assert suite.n_tasks > 0
    task = suite.get_task(0)
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    assert bddl.is_file(), bddl
    states = suite.get_task_init_states(0)
    assert len(states) > 2
    print(f'Creating simulator: {bddl}', flush=True)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64, camera_depths=True)
    try:
        env.seed(7)
        env.reset()
        obs = env.set_init_state(states[2])
        obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
        assert obs['agentview_image'].shape == (64, 64, 3)
        assert np.isfinite(obs['robot0_eef_pos']).all()
        print(f'{suite_name}: reset, initial state, action, RGB/depth observations OK', flush=True)
    finally:
        env.close()
print('Simulator smoke check passed; no LLM or policy calls were made.', flush=True)
