# Source this file to activate the Pigey simulator environment.
export PIGEY_ROOT="${PIGEY_ROOT:-/scratch3/workspace/udhanuka_umass_edu-pigey}"
export PIGEY_REPO="${PIGEY_REPO:-/home/udhanuka_umass_edu/Scalar_Research/HeXA/baselines/Pigey}"
export LIBERO_CONFIG_PATH="$PIGEY_ROOT/config/libero"
# A shell profile may export HF_HUB_CACHE/HF_DATASETS_CACHE/TRANSFORMERS_CACHE
# pointed at a different project's cache (e.g. via --export=ALL); those take
# precedence over HF_HOME in huggingface_hub's resolution, so they are cleared
# here rather than merely shadowed.
unset HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_HUB_TEMP_DIR
export HF_HOME="$PIGEY_ROOT/cache/huggingface"
export OPENPI_DATA_HOME="$PIGEY_ROOT/cache/openpi"
export XDG_CACHE_HOME="$PIGEY_ROOT/cache/xdg"
export NUMBA_CACHE_DIR="$PIGEY_ROOT/cache/numba"
export UV_CACHE_DIR="$PIGEY_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PIGEY_ROOT/python"
export PIP_CACHE_DIR="$PIGEY_ROOT/cache/pip"
export PYTHONPATH="$PIGEY_ROOT/LIBERO-PRO${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
source "$PIGEY_ROOT/.venv/bin/activate"
