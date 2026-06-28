#!/bin/bash
#SBATCH --job-name=interphyre_3b_1gpu
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH --mem=128G
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH -A pi_sniekum_umass_edu
#SBATCH --reservation=sniekum

# ─── Interphyre Physics Puzzle RL Training on 1x A100 80GB ───────────────────
# Model: Qwen2.5-3B-Instruct
# Level: two_body_problem  [SKILL-AUGMENTED]
# Train seeds: 1–50, Val seeds: 51–100
# batch_size=1 (1 seed per gradient update), 10 epochs = 500 steps
# Skills injected into system prompt from SKILL-RL/Skill_banks/skill_bank_evolving_two_body.json
#
# Setup:
#   python examples/data_preprocess/interphyre_data.py \
#     --output_dir data/interphyre_tbp_s1_50_skilled \
#     --levels two_body_problem \
#     --num_train_per_level 50 --num_val_per_level 50 \
#     --train_seed_start 1 --val_seed_start 51 \
#     --skill_bank SKILL-RL/Skill_banks/skill_bank_evolving_two_body.json
# Then:
#   sbatch examples/train/interphyre/slurm_train_3b_a100_1gpu_tbp_bsz1_skilled.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR="/project/pi_sniekum_umass_edu/vgandhi"
VERL_TOOL_DIR="$PROJECT_DIR/HeXA-Hierarchical-Experimentalist-Agents/training"
export PYTHONPATH="$(dirname "$VERL_TOOL_DIR")/interphyre:${PYTHONPATH:-}"   # makes repo-level interphyre package importable
CONDA_ENV="$PROJECT_DIR/conda/envs/VerlToolInterphyre"
DATA_DIR="$VERL_TOOL_DIR/data/interphyre_tbp_s1_50_skilled"

# ─── Model & Data ────────────────────────────────────────────────────────────
model_name="Qwen/Qwen2.5-3B-Instruct"
train_data="$DATA_DIR/train.parquet"
val_data="$DATA_DIR/val.parquet"

# ─── Hyperparameters ─────────────────────────────────────────────────────────
rl_alg=grpo
n_gpus_per_node=1
n_nodes=1
n=4                         # rollouts per prompt (GRPO group size)
total_epochs=10
total_training_steps=500    # 50 train samples / batch_size 1 = 50 steps/epoch × 10 epochs
batch_size=1                # 1 seed per gradient update
ppo_mini_batch_size=4       # all n=4 rollouts from the single seed in one mini-batch
max_prompt_length=3072      # increased from 2048 — skilled prompt is ~2200 tokens
max_action_length=1024
max_response_length=4096
max_obs_length=512
temperature=1.0
top_p=1.0
enable_agent=True
strategy=fsdp
max_turns=25
kl_loss_coef=0.0001
kl_coef=0
entropy_coeff=0.001
kl_loss_type=low_var_kl
lr=1e-6
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
agent_loop_workers=4        # batch_size*n=4 must be divisible by this; 4 works with batch_size=1,n=4

# Stop token: after Action Input the model writes "\nObservation:"
# vLLM halts there; tool server injects the real observation.
action_stop_tokens=$'\nObservation:'

# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  Interphyre 3B Training on 1x A100 80GB  [SKILL-AUGMENTED]"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURMD_NODENAME"
echo "  GPUs   : $CUDA_VISIBLE_DEVICES"
echo "  n=$n  batch=$batch_size  max_response_length=$max_response_length"
echo "================================================================"

module load conda/latest
module load cuda/12.8
export PATH="$CONDA_ENV/bin:${PATH:-}"
PYTHON="$CONDA_ENV/bin/python"

# ─── Sanity checks ───────────────────────────────────────────────────────────
if [ ! -f "$train_data" ]; then
    echo "[ERROR] Training data not found: $train_data"
    echo "        Run: python examples/data_preprocess/interphyre_data.py \\"
    echo "               --output_dir data/interphyre_tbp_s1_50_skilled \\"
    echo "               --levels two_body_problem \\"
    echo "               --num_train_per_level 50 --num_val_per_level 50 \\"
    echo "               --train_seed_start 1 --val_seed_start 51 \\"
    echo "               --skill_bank SKILL-RL/Skill_banks/skill_bank_evolving_two_body.json"
    exit 1
fi

cd "$VERL_TOOL_DIR"
mkdir -p logs

export VLLM_USE_V1=1
export WANDB_API_KEY="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"

# ─── Run name ────────────────────────────────────────────────────────────────
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="interphyre-a100-1gpu-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-skilled-tbp"
export VERL_RUN_ID=$run_name

# ─── Incremental run directory ───────────────────────────────────────────────
run_num=1
while [ -d "logs/run${run_num}" ]; do
    run_num=$((run_num + 1))
done
RUN_DIR="logs/run${run_num}"
mkdir -p "$RUN_DIR"
echo "[$(date)] Run directory: $RUN_DIR (run #${run_num})"

cat > "$RUN_DIR/run_info.txt" << EOF
job_id:     $SLURM_JOB_ID
run_num:    $run_num
run_name:   $run_name
node:       $SLURMD_NODENAME
gpus:       $CUDA_VISIBLE_DEVICES
model:      $model_name
algorithm:  $rl_alg
batch_size: $batch_size
max_turns:  $max_turns
n_gpus:     $n_gpus_per_node
level:      two_body_problem
train_seeds: 1-50
val_seeds:   51-100
skill_bank: skill_bank_evolving_two_body.json
start_time: $(date)
EOF

ln -sf "$(pwd)/logs/train_${SLURM_JOB_ID}.out" "$RUN_DIR/train.out" 2>/dev/null || true
ln -sf "$(pwd)/logs/train_${SLURM_JOB_ID}.err" "$RUN_DIR/train.err" 2>/dev/null || true
ln -sfn "run${run_num}" logs/latest
export TRAJ_LOG_FILE="$RUN_DIR/trajectories.jsonl"

# ─── Write action stop token to temp file ────────────────────────────────────
action_stop_tokens_file=$(mktemp /tmp/action_stop_tokens.XXXXXX)
echo -e -n "$action_stop_tokens" > "$action_stop_tokens_file"

# ─── Cleanup trap ────────────────────────────────────────────────────────────
TOOL_SERVER_PID=""
METRICS_PID=""

cleanup() {
    echo "[$(date)] Cleaning up..."
    [ -n "$TOOL_SERVER_PID" ] && kill -9 "$TOOL_SERVER_PID" 2>/dev/null || true
    [ -n "$METRICS_PID" ]     && kill    "$METRICS_PID"     2>/dev/null || true
    rm -f "$action_stop_tokens_file"
    [ -f "logs/tool_server_${SLURM_JOB_ID}.log" ] && \
        cp "logs/tool_server_${SLURM_JOB_ID}.log" "$RUN_DIR/tool_server.log" 2>/dev/null || true
    echo "[$(date)] Cleanup done. Logs in $RUN_DIR"
}
trap cleanup EXIT

# ─── Start Metrics Monitor ───────────────────────────────────────────────────
$PYTHON "$(dirname "$0")/parse_metrics.py" \
    --log_file "$(pwd)/logs/train_${SLURM_JOB_ID}.out" \
    --output_csv "$RUN_DIR/metrics.csv" \
    > "$RUN_DIR/metrics_monitor.log" 2>&1 &
METRICS_PID=$!
echo "[$(date)] Metrics monitor started (PID=$METRICS_PID) → $RUN_DIR/metrics.csv"

# ─── Start Tool Server ───────────────────────────────────────────────────────
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url="http://$host:$port/get_observation"

echo "[$(date)] Starting interphyre tool server on $tool_server_url..."
$PYTHON -m verl_tool.servers.serve \
    --host "$host" \
    --port "$port" \
    --tool_type interphyre \
    --workers_per_tool 4 \
    > logs/tool_server_${SLURM_JOB_ID}.log 2>&1 &
TOOL_SERVER_PID=$!

echo "[$(date)] Waiting for tool server (PID=$TOOL_SERVER_PID)..."
for i in $(seq 1 60); do
    if curl -sf "http://$host:$port/health" >/dev/null 2>&1; then
        echo "[$(date)] Tool server ready after $((i * 5))s"
        break
    fi
    if ! kill -0 "$TOOL_SERVER_PID" 2>/dev/null; then
        echo "[ERROR] Tool server died. Check logs/tool_server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if [ $i -eq 60 ]; then
        echo "[ERROR] Tool server not ready after 300s"
        exit 1
    fi
    sleep 5
done

# ─── Run PPO Training ────────────────────────────────────────────────────────
echo "[$(date)] Starting PPO training..."
echo "  Model:       $model_name"
echo "  Algorithm:   $rl_alg"
echo "  Batch size:  $batch_size (n=$n rollouts per prompt)"
echo "  Tool server: $tool_server_url"
echo "  Run name:    $run_name"

PYTHONUNBUFFERED=1 $PYTHON -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
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
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=$total_training_steps

echo "[$(date)] Training complete."
