set -x


export VLLM_USE_V1=1
RAY_DEBUG=legacy


export CUDA_VISIBLE_DEVICES=2,3
TASK="medium"
ROOT_FOLDER=''
collab_train_path="$ROOT_FOLDER/data/collab/${TASK}/rl_train.parquet"
collab_val_path="$ROOT_FOLDER/data/collab/${TASK}/rl_validation.parquet"

train_files="['$collab_train_path']"
test_files="['$collab_val_path']"
SAVE_PATH="$ROOT_FOLDER/verl/checkpoints/collab_math_rl"

RESUME_PATH="$ROOT_FOLDER/"

model_path="Qwen/Qwen2.5-3B-Instruct"

nohup
python3 -m recipe.itpo.main_prime \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.filter_accuracy=False \
    data.accuracy_lower_bound=0.2 \
    data.accuracy_upper_bound=0.8 \
    data.oversample_factor=1 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=3 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.adv_estimator=rloo_itponorm \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.reward_gt_coef=0 \
    algorithm.reward_dpo_coef=15 \
    +algorithm.reward_turn_coef=0 \
    reward_model.model.path=$model_path \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.model.update=itponorm \
    reward_model.model.beta_train=0.005 \
    reward_model.model.optim.lr=1e-6 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.model.input_tokenizer=null \
    reward_model.mini_batch_size=4 \
    reward_model.ulysses_sequence_parallel_size=1 \
    +reward_model.model.temp=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.path=$model_path \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_mini_batch_size=4 \
    critic.enable=False \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verlxcollabllm' \
    trainer.experiment_name=$TASK-rloo-itponorm \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=500 \
    trainer.total_epochs=20 $@ \
    reward_model.reward_manager=collabllm \
    +reward_model.reward_kwargs.metric_weights.bleu_score=1 \
    +reward_model.reward_kwargs.metric_weights.token_amount=-0.00005 \
    +reward_model.reward_kwargs.llm_judge_kwargs.model="hosted_vllm/Qwen/Qwen2.5-14B-Instruct" \
    +reward_model.reward_kwargs.llm_judge_kwargs.max_tokens=2048 \
    +reward_model.reward_kwargs.llm_judge_kwargs.temperature=0 \
    +reward_model.reward_kwargs.llm_judge_kwargs.api_base="http://0.0.0.0:8001/v1"\
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.trace.token2text=True \
    custom_reward_function.path=recipe/itpo/reward_function.py \
    custom_reward_function.name=conversation_level_reward_func \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$ROOT_FOLDER/verl/recipe/itpo/config/itpo_interaction_collab.yaml" \
    trainer.resume_mode=disable \
    trainer.resume_from_path=$RESUME_PATH \
    reward_model.enable_resource_pool=True \
    >./${TASK}_rloo_itponorm.log 2>&1 & </dev/null &
