#!/bin/bash
#SBATCH --job-name=evolving_grpo_tbp
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH --mem=128G
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH -A pi_sniekum_umass_edu

# ─── Continuous-Process Evolving Skill Bank + GRPO Training (TBP) ─────────────
# Level: two_body_problem
# Variant: evolving-skilled (single verl process, file-based bank)
#
#   Phase 0 (once, before verl starts):
#     5 base-model trajectories (FirstFiveTrajs/) → Qwen 7B teacher → skill_bank_round_0.json
#
#   Continuous training:
#     One verl process for all 500 steps. The dataset reads skill_bank.json on
#     every __getitem__ and injects the current bank into the system prompt.
#     Every SKILL_EVOLVE_FREQ steps, a synchronous trainer hook tails
#     trajectories.jsonl, calls the teacher, and atomically rewrites the bank.
#     No checkpoint reload, no Adam reset, no Ray/FSDP re-init.
#
# Pre-requisites:
#   1. FirstFiveTrajs available: SKILL-RL/FirstFiveTrajs/tbp_epoch0_trajectories.jsonl
#   2. vLLM teacher running:     sbatch slurm_vllm_7b_teacher.sh  (writes endpoint file)
#      OR set VLLM_ENDPOINT env var: e.g. http://gpu014:8000
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
SKILL_EVOLVE_FREQ="${SKILL_EVOLVE_FREQ:-3}"        # steps between teacher calls
INIT_TRAJ_SEEDS="${INIT_TRAJ_SEEDS:-1 2 3 4 5}"     # seeds assigned to FirstFiveTrajs lines
TRAIN_SEED_START="${TRAIN_SEED_START:-1}"
NUM_TRAIN_SEEDS="${NUM_TRAIN_SEEDS:-50}"
VAL_SEED_START="${VAL_SEED_START:-51}"
NUM_VAL_SEEDS="${NUM_VAL_SEEDS:-50}"
MAX_SKILLS="${MAX_SKILLS:-10}"
MAX_MISTAKES="${MAX_MISTAKES:-5}"
LEVEL="two_body_problem"

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/project/pi_sniekum_umass_edu/vgandhi"
VERL_TOOL_DIR="$PROJECT_DIR/HeXA-Hierarchical-Experimentalist-Agents/training"
export PYTHONPATH="$(dirname "$VERL_TOOL_DIR")/interphyre:${PYTHONPATH:-}"   # makes repo-level interphyre package importable
CONDA_ENV="$PROJECT_DIR/conda/envs/VerlToolInterphyre"
PYTHON="$CONDA_ENV/bin/python"

INIT_TRAJ_JSONL="$VERL_TOOL_DIR/SKILL-RL/FirstFiveTrajs/tbp_epoch0_trajectories.jsonl"
SKILLRL_INIT_DIR="$VERL_TOOL_DIR/SKILL-RL/data/initial/$LEVEL"
SKILL_BANK_DIR="$VERL_TOOL_DIR/SKILL-RL/Skill_banks/evolving_grpo_tbp"
DATA_DIR="$VERL_TOOL_DIR/data/interphyre_tbp_evolving_continuous"

# The dataset reads this path at every __getitem__; the hook rewrites it.
SKILL_BANK_LIVE="$SKILL_BANK_DIR/skill_bank_live.json"

# ─── Model / Training hyperparameters (mirror static-skilled script) ─────────
model_name="Qwen/Qwen2.5-3B-Instruct"
rl_alg=grpo
n=4
batch_size=1
ppo_mini_batch_size=4
max_prompt_length=3072
max_action_length=1024
max_response_length=4096
max_obs_length=512
temperature=1.0
top_p=1.0
lr=1e-6
total_epochs=10
total_training_steps=500
kl_loss_coef=0.001
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
reward_manager=interphyre
ppo_micro_batch_size_per_gpu=2
log_prob_micro_batch_size_per_gpu=4
tensor_model_parallel_size=1
gpu_memory_utilization=0.3
do_offload=False
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=False
rollout_mode=async
agent_loop_workers=4
max_turns=25
enable_agent=True
action_stop_tokens=$'\nObservation:'
strategy=fsdp

# ─────────────────────────────────────────────────────────────────────────────

module load conda/latest
module load cuda/12.8
export PATH="$CONDA_ENV/bin:${PATH:-}"
export HF_HOME="/scratch4/workspace/svaidyanatha_umass_edu-phyre/hf_cache"

cd "$VERL_TOOL_DIR"
mkdir -p logs "$SKILL_BANK_DIR" "$SKILLRL_INIT_DIR"

echo "================================================================"
echo "  Evolving Skill Bank + GRPO (continuous process) — two_body_problem"
echo "  Job ID         : $SLURM_JOB_ID"
echo "  Node           : $SLURMD_NODENAME"
echo "  Skill freq     : every $SKILL_EVOLVE_FREQ steps"
echo "  Total steps    : $total_training_steps"
echo "================================================================"

# ─── Wait for vLLM teacher endpoint ──────────────────────────────────────────
ENDPOINT_FILE="$VERL_TOOL_DIR/logs/vllm_teacher_endpoint.txt"
if [ -n "${VLLM_ENDPOINT:-}" ]; then
    TEACHER_ENDPOINT="$VLLM_ENDPOINT"
    echo "Using VLLM_ENDPOINT from env: $TEACHER_ENDPOINT"
else
    echo "Waiting for vLLM teacher endpoint (logs/vllm_teacher_endpoint.txt)..."
    for i in $(seq 1 60); do
        [ -f "$ENDPOINT_FILE" ] && { TEACHER_ENDPOINT=$(cat "$ENDPOINT_FILE"); break; }
        [ $i -eq 60 ] && { echo "[ERROR] no teacher endpoint after 5min"; exit 1; }
        sleep 5
    done
    echo "Found teacher endpoint: $TEACHER_ENDPOINT"
    for i in $(seq 1 120); do
        if curl -sf "$TEACHER_ENDPOINT/health" >/dev/null 2>&1; then
            echo "Teacher server ready after $((i * 5))s."
            break
        fi
        [ $i -eq 120 ] && { echo "[ERROR] Teacher server unreachable after 10min."; exit 1; }
        sleep 5
    done
fi

# ─── Sanity checks ───────────────────────────────────────────────────────────
[ ! -f "$INIT_TRAJ_JSONL" ] && { echo "[ERROR] Missing: $INIT_TRAJ_JSONL"; exit 1; }

export VLLM_USE_V1=1
export WANDB_API_KEY="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"

model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="interphyre-a100-1gpu-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-evolving-tbp"
export VERL_RUN_ID=$run_name

# ─── Incremental run directory ───────────────────────────────────────────────
run_num=1
while [ -d "logs/run${run_num}" ]; do run_num=$((run_num + 1)); done
RUN_DIR="logs/run${run_num}"
mkdir -p "$RUN_DIR"
ln -sfn "run${run_num}" logs/latest
export TRAJ_LOG_FILE="$RUN_DIR/trajectories.jsonl"
export TRAJ_LOG_ALL=1   # log all n=4 rollouts so the hook sees full GRPO group
# Per-rollout full-text dump: canonical input for the skill-evolution hook.
# Reward manager writes step_{S}/{phase}/seed{N}_w{w}.json files here.
export FULL_TRAJ_DIR="$RUN_DIR/full_trajectories"
mkdir -p "$FULL_TRAJ_DIR"
echo "Run directory: $RUN_DIR"

# ─── Phase 0: Initial skill bank distillation (once, before verl starts) ─────
INIT_SKILL_BANK="$SKILL_BANK_DIR/skill_bank_round_0.json"

echo ""
echo "════════════════════════════════════════════"
echo "  PHASE 0: Initial skill bank distillation"
echo "════════════════════════════════════════════"

if [ ! -f "$INIT_SKILL_BANK" ]; then
    echo "Converting initial trajectories (seeds: $INIT_TRAJ_SEEDS)..."
    $PYTHON SKILL-RL/tools/convert_training_trajs.py \
        --input "$INIT_TRAJ_JSONL" \
        --output "$SKILLRL_INIT_DIR" \
        --seeds $INIT_TRAJ_SEEDS

    echo "Distilling initial skill bank..."
    $PYTHON -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from skillrl.distillation.distill import run_distillation
run_distillation(
    traj_dirs={'$LEVEL': Path('$SKILLRL_INIT_DIR')},
    levels=['$LEVEL'],
    output_path=Path('$INIT_SKILL_BANK'),
    teacher_model='vllm:$TEACHER_ENDPOINT',
    skip_general=True,
)
print(f'Initial skill bank saved: $INIT_SKILL_BANK')
"
else
    echo "Initial skill bank already exists: $INIT_SKILL_BANK"
fi

# Seed the live bank file from the round-0 distillation. The hook will
# overwrite this in place every SKILL_EVOLVE_FREQ steps.
cp "$INIT_SKILL_BANK" "$SKILL_BANK_LIVE"
echo "Live skill bank: $SKILL_BANK_LIVE"

# Snapshot the starting bank into the run's skill_evolution tree as step_0.
# Subsequent step_{K,2K,...}/{level}/bank_out.json snapshots are written by
# the in-process hook after each evolve.
STEP0_DIR="$RUN_DIR/skill_evolution/step_0/$LEVEL"
mkdir -p "$STEP0_DIR"
cp "$SKILL_BANK_LIVE" "$STEP0_DIR/bank_out.json"

# ─── Generate training parquet (no skills baked in) ──────────────────────────
TRAIN_DATA="$DATA_DIR/train.parquet"
VAL_DATA="$DATA_DIR/val.parquet"

if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
    echo ""
    echo "Generating training/val parquet (skills injected at runtime)..."
    $PYTHON examples/data_preprocess/interphyre_data.py \
        --output_dir "$DATA_DIR" \
        --levels "$LEVEL" \
        --num_train_per_level $NUM_TRAIN_SEEDS \
        --num_val_per_level $NUM_VAL_SEEDS \
        --train_seed_start $TRAIN_SEED_START \
        --val_seed_start $VAL_SEED_START \
        --runtime_skill_bank_path "$SKILL_BANK_LIVE"
else
    echo "Training/val parquet already exists at $DATA_DIR"
fi

# ─── Cleanup trap ────────────────────────────────────────────────────────────
action_stop_tokens_file=$(mktemp /tmp/action_stop_tokens.XXXXXX)
echo -e -n "$action_stop_tokens" > "$action_stop_tokens_file"

TOOL_SERVER_PID=""
METRICS_PID=""
cleanup() {
    [ -n "$TOOL_SERVER_PID" ] && kill -9 "$TOOL_SERVER_PID" 2>/dev/null || true
    [ -n "$METRICS_PID" ]     && kill    "$METRICS_PID"     2>/dev/null || true
    rm -f "$action_stop_tokens_file"
    [ -f "logs/tool_server_${SLURM_JOB_ID}.log" ] && \
        cp "logs/tool_server_${SLURM_JOB_ID}.log" "$RUN_DIR/tool_server.log" 2>/dev/null || true
}
trap cleanup EXIT

# ─── Start metrics monitor ───────────────────────────────────────────────────
$PYTHON "$(pwd)/examples/train/interphyre/parse_metrics.py" \
    --log_file "$(pwd)/logs/train_${SLURM_JOB_ID}.out" \
    --output_csv "$RUN_DIR/metrics.csv" \
    > "$RUN_DIR/metrics_monitor.log" 2>&1 &
METRICS_PID=$!

# ─── Start tool server ───────────────────────────────────────────────────────
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url="http://$host:$port/get_observation"

echo "[$(date)] Starting interphyre tool server on $tool_server_url..."
$PYTHON -m verl_tool.servers.serve \
    --host "$host" --port "$port" \
    --tool_type interphyre --workers_per_tool 4 \
    > logs/tool_server_${SLURM_JOB_ID}.log 2>&1 &
TOOL_SERVER_PID=$!

for i in $(seq 1 60); do
    curl -sf "http://$host:$port/health" >/dev/null 2>&1 && { echo "Tool server ready."; break; }
    ! kill -0 "$TOOL_SERVER_PID" 2>/dev/null && { echo "[ERROR] Tool server died."; exit 1; }
    [ $i -eq 60 ] && { echo "[ERROR] Tool server timeout."; exit 1; }
    sleep 5
done

# ─── Run PPO training (single continuous invocation) ─────────────────────────
SKILL_EVOLVE_WORK_DIR="$RUN_DIR/skill_evolution"
mkdir -p "$SKILL_EVOLVE_WORK_DIR"

echo "[$(date)] Starting PPO training (continuous evolving)..."
PYTHONUNBUFFERED=1 $PYTHON -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$batch_size \
    data.val_batch_size=50 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.agent.num_workers=$agent_loop_workers \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=interphyre_rl \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/scratch4/workspace/svaidyanatha_umass_edu-phyre/checkpoints/interphyre/${run_name}/run${run_num} \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=$total_training_steps \
    +trainer.skill_evolve_freq=$SKILL_EVOLVE_FREQ \
    +trainer.skill_bank_path=$SKILL_BANK_LIVE \
    +trainer.skill_evolve_level=$LEVEL \
    +trainer.skill_evolve_teacher_endpoint=$TEACHER_ENDPOINT \
    +trainer.skill_evolve_work_dir=$SKILL_EVOLVE_WORK_DIR \
    +trainer.skill_evolve_max_skills=$MAX_SKILLS \
    +trainer.skill_evolve_max_mistakes=$MAX_MISTAKES

echo "[$(date)] Training complete."
