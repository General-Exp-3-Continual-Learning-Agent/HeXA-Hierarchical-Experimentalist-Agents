set -x

# ---- Environment ----
# Activate the VerlToolInterphyre conda environment
source "$(conda info --base 2>/dev/null || echo $HOME/miniconda3)"/etc/profile.d/conda.sh 2>/dev/null || true
conda activate VerlToolInterphyre 2>/dev/null || source /project/pi_sniekum_umass_edu/vgandhi/conda/envs/VerlToolInterphyre/bin/activate 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$(pwd):$(cd "$(pwd)/.." && pwd)/interphyre:$PYTHONPATH   # makes repo-level `interphyre` package importable (sibling of training/)

# ---- Paths ----
train_data=$(pwd)/data/interphyre/train.parquet
val_data=$(pwd)/data/interphyre/val.parquet

# ---- Model ----
model_name=Qwen/Qwen2.5-3B-Instruct

# ---- Algorithm ----
rl_alg=grpo

# ---- Hyperparameters ----
n_gpus_per_node=4
n_nodes=1
n=8                         # rollouts per prompt for GRPO group normalization
batch_size=64
ppo_mini_batch_size=64
max_prompt_length=2048      # system prompt is long (~800 tokens)
max_response_length=4096    # 10 turns × ~400 tokens/turn
max_obs_length=512
max_action_length=2048
temperature=1.0
top_p=1.0
max_turns=10

# ---- Training knobs ----
lr=1e-6
kl_loss_coef=0.0
kl_coef=0.0
entropy_coeff=0.0
kl_loss_type=low_var_kl
gpu_memory_utilization=0.6
do_offload=True
use_dynamic_bsz=True
strategy=fsdp
fsdp_size=-1
ulysses_sequence_parallel_size=1
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=8
tensor_model_parallel_size=1
mask_observations=True
enable_mtrl=False

# <|im_end|> token id for Qwen2.5
additional_eos_token_ids=[151645]

reward_manager=interphyre
wandb_project=interphyre_rl

# ---- Run name ----
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="${reward_manager}-${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1

rollout_mode=async

# ---- Stop token (written to temp file — verl can't pass special chars as params) ----
# The ReAct agent writes "\nObservation:" after each Action Input.
# vLLM stops generation at this token and hands control to the tool server.
action_stop_tokens=$'\nObservation:'
action_stop_tokens_file="$(pwd)$(mktemp)"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

# ---- Tool server ----
mkdir -p logs
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

python -m verl_tool.servers.serve \
    --host $host \
    --port $port \
    --tool_type interphyre \
    --workers_per_tool 8 \
    --use_ray=True \
    > logs/tool_server_${run_name}.log 2>&1 &
server_pid=$!
echo "Tool server (pid=$server_pid) starting at $tool_server_url"

# Wait for tool server to be ready
sleep 15

# ---- Training ----
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=32 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation=right \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
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
    actor_rollout_ref.agent.enable_agent=True \
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
    +actor_rollout_ref.agent.retokenization=True \
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
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=$temperature \
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
    trainer.project_name=$wandb_project \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.rollout_data_dir=$(pwd)/verl_step_records/$run_name \
    trainer.validation_data_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=20

# ---- Cleanup ----
pkill -P -9 $server_pid
kill -9 $server_pid
rm -f $action_stop_tokens_file
