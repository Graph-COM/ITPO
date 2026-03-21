# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import statistics
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional, Union

import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

# below are for _validate
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    _compute_response_info,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)

# from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import (
    need_critic,
    need_reference_policy,
    need_reward_model,
    Role,
    WorkerType,
)
from verl.utils.checkpoint.checkpoint_manager import (
    find_latest_ckpt_path,
    should_save_ckpt_esi,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn, RLHFDataset
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import simple_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

from . import itpo_core_algos, prime_core_algos


def compute_advantage(data: DataProto, adv_estimator, config):
    if "rloo" in adv_estimator:
        response_mask = data.batch["response_mask"]
        advantages, returns = itpo_core_algos.compute_rloo_advantage_return(
            data, response_mask, config.actor_rollout_ref.rollout.n, config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def compute_data_metrics(batch, use_critic=True):
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    # response_length = response_info["response_length"]
    response_length = torch.sum(batch.batch["response_mask"], dim=-1).float()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5))
                .detach()
                .item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(
            torch.eq(response_length, max_response_length).float()
        )
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(
            torch.eq(prompt_length, max_prompt_length).float()
        )
        .detach()
        .item(),
    }
    return metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{
            name: num_overall_tokens
            for name in ["ref", "values", "adv", "update_critic", "update_actor"]
        },
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name]
            * 1000
            / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


class RayPRIMETrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        device_name="cuda",
    ):
        # assert get_torch_device().is_available(), 'cuda must be available on driver'

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=device_name,
        )

        if "gae" not in config.algorithm.adv_estimator:
            self.use_critic = False
        else:
            self.use_critic = True

    def _create_dataloader(self, *args, **kwargs):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            config=self.config.data,
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=int(
                self.config.data.train_batch_size * self.config.data.oversample_factor
            ),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            config=self.config.data,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )
            self.config.critic.optim.total_training_steps = total_training_steps

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
        )

        if self.use_rm:
            reward_local_path = os.path.join(local_global_step_folder, "reward")
            reward_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "reward",
                )
            )
            self.rm_wg.save_checkpoint(
                reward_local_path,
                reward_remote_path,
                self.global_steps,
            )
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "critic",
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        import dill

        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = (
                self.config.trainer.default_local_dir
            )  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(
                checkpoint_folder
            )  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(
                    self.config.trainer.resume_from_path, str
                ), "resume ckpt must be str type"
                assert (
                    "global_step_" in self.config.trainer.resume_from_path
                ), "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        reward_path = os.path.join(global_step_folder, "reward")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )
        # load rm
        if self.use_rm:
            self.rm_wg.load_checkpoint(
                reward_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
        # load critic
        if self.use_critic:
            critic_path = os.path.join(global_step_folder, "critic")
            self.critic_wg.load_checkpoint(
                critic_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        self.train_dataloader = torch.load(dataloader_local_path, weights_only=False)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to
        construct the PPO dataflow. The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                """gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"]
                )"""
                gen_batch = self._get_gen_batch(batch)

                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                with simple_timer("step", timing_raw):
                    # generate a batch
                    with simple_timer("gen", timing_raw):
                        """gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch
                        )"""
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(
                                gen_batch
                            )
                        else:
                            gen_batch_output = (
                                self.async_rollout_manager.generate_sequences(gen_batch)
                            )
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == "remax":
                        with simple_timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = (
                                self.actor_rollout_wg.generate_sequences(
                                    gen_baseline_batch
                                )
                            )

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    batch = batch.union(gen_batch_output)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # post process the inference result
                    self.process_inference_metrics(batch, metrics=metrics)

                    batch = self.filter_and_downsample(
                        batch.batch["scores"],
                        batch,
                    )
                    batch.meta_info["n"] = self.config.actor_rollout_ref.rollout.n
                    n_samples = self.config.actor_rollout_ref.rollout.n

                    # recompute old_log_probs
                    with simple_timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        loss_agg_mode = (
                            self.config.actor_rollout_ref.actor.loss_agg_mode
                        )
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=batch.batch["response_mask"],
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with simple_timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with simple_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    if self.config.algorithm.reward_dpo_coef != 0:
                        with simple_timer("adv", timing_raw):
                            if self.use_rm:
                                update_style = self.config.reward_model.model.get(
                                    "update", "none"
                                )
                                if (
                                    update_style == "before"
                                ):  # update reward model, and then run forward
                                    reward_output = self.rm_wg.update_rm_mt(batch)
                                    if "metrics" in reward_output.meta_info.keys():
                                        reward_output_metrics = reduce_metrics(
                                            reward_output.meta_info["metrics"]
                                        )
                                        metrics.update(reward_output_metrics)
                                    batch.batch["before_update_rm_scores"] = (
                                        reward_output.batch["rm_scores"]
                                        .detach()
                                        .clone()
                                    )

                                    reward_output = self.rm_wg.compute_rm_score_mt(
                                        batch
                                    )
                                    _, token_proportions = (
                                        itpo_core_algos.calculate_segment_proportions(
                                            reward_output.batch["rm_scores"],
                                            batch.batch["response_mask"],
                                            self.config.reward_model.model.get(
                                                "temp", 0.7
                                            ),
                                        )
                                    )
                                    batch.non_tensor_batch["token_proportions"] = (
                                        token_proportions
                                    )

                                    # here should also consider the metrics after update
                                    reward_output_metrics_after = reduce_metrics(
                                        reward_output.meta_info["metrics"]
                                    )
                                    new_dict = {
                                        key + "_afterupdate": value
                                        for key, value in reward_output_metrics_after.items()
                                    }
                                    metrics.update(new_dict)

                                elif update_style == "itponorm":
                                    # update reward model, and then run forward
                                    # in ITPO way
                                    reward_output = self.rm_wg.update_rm_mt(batch)
                                    if "metrics" in reward_output.meta_info.keys():
                                        reward_output_metrics = reduce_metrics(
                                            reward_output.meta_info["metrics"]
                                        )
                                        metrics.update(reward_output_metrics)
                                    batch.batch["before_update_rm_scores"] = (
                                        reward_output.batch["rm_scores"]
                                        .detach()
                                        .clone()
                                    )
                                    reward_output = self.rm_wg.compute_rm_score_mt(
                                        batch
                                    )
                                    # here should also consider the metrics after update
                                    reward_output_metrics_after = reduce_metrics(
                                        reward_output.meta_info["metrics"]
                                    )
                                    new_dict = {
                                        key + "_afterupdate": value
                                        for key, value in reward_output_metrics_after.items()
                                    }
                                    metrics.update(new_dict)
                                    turn_level_proportions, token_proportions = (
                                        itpo_core_algos.calculate_segment_proportions_itponorm(
                                            reward_output.batch["rm_scores"],
                                            batch.batch["response_mask"],
                                            self.config.reward_model.model.get(
                                                "temp", 0.7
                                            ),
                                        )
                                    )
                                    batch.non_tensor_batch["turn_level_proportions"] = (
                                        turn_level_proportions
                                    )
                                    batch.non_tensor_batch["token_proportions"] = (
                                        token_proportions
                                    )
                                    variances = [
                                        np.var(arr["proportions"])
                                        for arr in turn_level_proportions
                                    ]
                                    metrics.update(
                                        {
                                            "rm/proportion_variance": np.mean(
                                                variances
                                            ).item()
                                        }
                                    )
                                elif update_style == "itpo":
                                    # update reward model, and then run forward
                                    reward_output = self.rm_wg.update_rm_mt(batch)
                                    if "metrics" in reward_output.meta_info.keys():
                                        reward_output_metrics = reduce_metrics(
                                            reward_output.meta_info["metrics"]
                                        )
                                        metrics.update(reward_output_metrics)
                                    batch.batch["before_update_rm_scores"] = (
                                        reward_output.batch["rm_scores"]
                                        .detach()
                                        .clone()
                                    )
                                    reward_output = self.rm_wg.compute_rm_score_mt(
                                        batch
                                    )
                                    # here should also consider the metrics after update
                                    reward_output_metrics_after = reduce_metrics(
                                        reward_output.meta_info["metrics"]
                                    )
                                    new_dict = {
                                        key + "_afterupdate": value
                                        for key, value in reward_output_metrics_after.items()
                                    }
                                    metrics.update(new_dict)
                                    turn_level_proportions, token_proportions = (
                                        itpo_core_algos.calculate_segment_proportions_itpo(
                                            reward_output.batch["rm_scores"],
                                            batch.batch["response_mask"],
                                            self.config.reward_model.model.get(
                                                "temp", 0.7
                                            ),
                                        )
                                    )
                                    batch.non_tensor_batch["turn_level_proportions"] = (
                                        turn_level_proportions
                                    )
                                    batch.non_tensor_batch["token_proportions"] = (
                                        token_proportions
                                    )
                                    variances = [
                                        np.var(arr["proportions"])
                                        for arr in turn_level_proportions
                                    ]
                                    metrics.update(
                                        {
                                            "rm/proportion_variance": np.mean(
                                                variances
                                            ).item()
                                        }
                                    )
                                else:
                                    raise NotImplementedError
                                batch.pop(batch_keys=["rm_scores"])
                                batch = batch.union(reward_output)
                                # Here post-process the rm-scores for statistics
                                self.process_token_level_reward_metrics(batch, metrics)
                                # Here we need to log the statistics of the rm_scores
                                if "metrics" in reward_output.meta_info.keys():
                                    reward_output_metrics = reduce_metrics(
                                        reward_output.meta_info["metrics"]
                                    )
                                    metrics.update(reward_output_metrics)
                    else:
                        token_proportions = itpo_core_algos.calculate_token_proportions(
                            batch.batch["response_mask"],
                        )
                        batch.non_tensor_batch["token_proportions"] = token_proportions
                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        config=self.config,
                    )
                    if self.use_critic:
                        with simple_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(
                                critic_output.meta_info["metrics"]
                            )
                            metrics.update(critic_output_metrics)

                    # update actor
                    with simple_timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(
                        actor_output.meta_info["metrics"]
                    )
                    metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with simple_timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if (
                        self.config.trainer.save_freq > 0
                        and self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with simple_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    if (
                        self.config.trainer.save_freq > 0
                        and (self.global_steps - 1) % self.config.trainer.save_freq != 0
                    ):
                        with simple_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    return

    def filter_and_downsample(self, scores, batch: DataProto):
        """
        downsample the batch according to oversample_factor
        samples passing the filters will be prioritized
        """
        n_samples = int(self.config.actor_rollout_ref.rollout.n)
        reward_matrix = torch.tensor(scores).reshape(-1, n_samples)

        filter_mask = torch.ones((reward_matrix.shape[0]), dtype=torch.bool)

        if self.config.data.filter_accuracy:
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            filter_mask[
                (acc_tensor > self.config.data.accuracy_upper_bound)
                | (acc_tensor < self.config.data.accuracy_lower_bound)
            ] = False

        if self.config.data.filter_truncate:
            length_matrix = (
                batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1] :]
                .sum(dim=-1)
                .reshape(-1, n_samples)
            )
            length_tensor = torch.max(length_matrix, dim=-1)[0]
            filter_mask[length_tensor >= self.config.data.max_response_length - 1] = (
                False
            )

        reorder_index = torch.argsort(filter_mask, descending=True)
        reorder_index = (
            reorder_index.unsqueeze(-1) * n_samples
            + torch.arange(0, n_samples).unsqueeze(0)
        ).view(-1)
        batch.reorder(
            reorder_index[: int(len(batch) // self.config.data.oversample_factor)]
        )  # this operation is inplace

        return batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                    dtype=object,
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in test_batch
            ]
            sample_gts.extend(ground_truths)
            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, size_divisor
            )
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                    test_gen_batch_padded
                )
            else:
                test_output_gen_batch_padded = (
                    self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                )

            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            tmp_metrics = {}
            self.process_inference_metrics(test_batch, metrics=tmp_metrics)

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(
                f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}"
            )
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(
                        f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}"
                    )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(
                sample_scores
            ), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(
            data_sources, sample_uids, reward_extra_infos_dict
        )
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "scores" if "scores" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max(
                    [
                        int(name.split("@")[-1].split("/")[0])
                        for name in metric2val.keys()
                    ]
                )
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(
                            metric_name.startswith(pfx)
                            for pfx in ["mean", "maj", "best"]
                        )
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def process_inference_metrics(self, batch: DataProto, metrics: dict):
        # verify
        scores = torch.sum(batch.batch["rm_scores"], dim=-1)
        batch.batch["scores"] = scores
        # The score is used as the trajectory Reward
        metrics["scores"] = torch.mean(scores).item()
        # extra scores process and consider LLM judge process
        for metric in batch.non_tensor_batch["reward_extra_info"][0].keys():
            # process the metric (not attributed)
            if "attribute" not in metric:
                batch.batch["extra_" + metric] = torch.tensor(
                    batch.non_tensor_batch[metric]
                )
                metrics[f"extra_{metric}"] = float(
                    np.mean(batch.non_tensor_batch[metric])
                )
            elif (
                ("attribute" in metric)
                and ("token_amount" not in metric)
                and ("scores" not in metric)
            ):
                # count the variance and #valid judge assignment
                var = 0
                val_judge = 0
                for i in range(len(batch.non_tensor_batch[metric])):
                    if np.sum(batch.non_tensor_batch[metric][i]["value"]) != 0:
                        tmp_var = np.var(
                            batch.non_tensor_batch[metric][i]["value"]
                            / np.sum(batch.non_tensor_batch[metric][i]["value"])
                        )
                    else:
                        tmp_var = np.var(batch.non_tensor_batch[metric][i]["value"])
                    var += tmp_var
                    if (
                        np.all(batch.non_tensor_batch[metric][i]["value"][:-1] == 0)
                        and batch.batch[("extra_" + metric).removesuffix("_attribute")][
                            i
                        ]
                        != 0
                    ):
                        pass
                    else:
                        val_judge += 1
                var = var / len(batch.non_tensor_batch[metric])
                val_judge = val_judge / len(batch.non_tensor_batch[metric])
                metrics[f"extra_{metric}_var"] = var
                metrics[f"extra_{metric}_val_rate"] = val_judge

        # then define the metric to learn with PRIME as batch['scores_to_learn']
        # Note that all the metrics are already weighted accoridng to the reward manager
        # So Here only need to sum and  / sum_weights
        metrics_to_learn = []
        all_weights = 0
        for metric in batch.non_tensor_batch["reward_extra_info"][0].keys():
            if len(metrics_to_learn) == 0:
                batch.batch["scores_to_learn"] = torch.zeros_like(
                    batch.batch["extra_" + metric]
                )
            if metric != "token_amount" and ("attribute" not in metric):
                metrics_to_learn.append(metric)
                weight = self.config.reward_model.reward_kwargs.metric_weights[metric]
                all_weights += weight
                batch.batch["scores_to_learn"] += batch.batch["extra_" + metric]
        batch.batch["scores_to_learn"] = batch.batch["scores_to_learn"] / all_weights
        batch.batch["scores_to_learn"] = batch.batch["scores_to_learn"].float()

        # Here remove the keys with attribute in reward_extra_info, otherwise will cause bug in _validate()
        for idx in range(len(batch.non_tensor_batch["reward_extra_info"])):
            batch.non_tensor_batch["reward_extra_info"][idx] = {
                k: v
                for k, v in batch.non_tensor_batch["reward_extra_info"][idx].items()
                if "attribute" not in k
            }
        # filter the batch. 1/oversample_factor samples will be kept.
        # If there is a filter, prompts passing it will be prioritized.

    def process_token_level_reward_metrics(self, batch: DataProto, metrics: dict):
        rm_scores = batch.batch["rm_scores"]
        response_mask = batch.batch["response_mask"]
        N, D = rm_scores.shape
        rm_scores_pad = F.pad(rm_scores, (0, 1), value=0)  # Shape: [N, D+1]
        response_mask_pad = F.pad(response_mask, (0, 1), value=0)  # Shape: [N, D+1]

        row_ids = torch.arange(N, device=rm_scores.device).unsqueeze(1).expand(N, D + 1)
        flat_rm_score = rm_scores_pad.reshape(-1)
        flat_response_mask = response_mask_pad.reshape(-1)
        flat_row_ids = row_ids.reshape(-1)

        shifted_response_mask = F.pad(flat_response_mask, (1, 0), value=0)[:-1]
        is_start = (flat_response_mask == 1) & (shifted_response_mask == 0)
        segment_ids = is_start.cumsum(dim=0) - 1
        valid_response_mask = flat_response_mask == 1

        valid_rm_scores = flat_rm_score[valid_response_mask]
        valid_ids = segment_ids[valid_response_mask]
        valid_row_indices = flat_row_ids[valid_response_mask]

        num_segments = valid_ids.max().item() + 1

        segment_sums = torch.zeros(
            num_segments, device=rm_scores.device, dtype=rm_scores.dtype
        )
        segment_row_ids = torch.zeros(
            num_segments, device=rm_scores.device, dtype=torch.long
        )

        segment_sums.scatter_add_(0, valid_ids, valid_rm_scores)
        segment_row_ids.scatter_(0, valid_ids, valid_row_indices)

        row_counts = torch.zeros(
            N, device=segment_sums.device
        )  # how many turns in each row
        row_sums = torch.zeros(
            N, device=segment_sums.device, dtype=segment_sums.dtype
        )  # sum in each turn
        row_sq_sums = torch.zeros(
            N, device=segment_sums.device, dtype=segment_sums.dtype
        )

        ones = torch.ones_like(segment_sums)
        row_counts.scatter_add_(0, segment_row_ids, ones)
        row_sums.scatter_add_(0, segment_row_ids, segment_sums)
        row_sq_sums.scatter_add_(0, segment_row_ids, segment_sums**2)
        safe_counts = row_counts.clamp(min=1)

        row_means = row_sums / safe_counts
        row_mean_sq = row_sq_sums / safe_counts
        row_vars = row_mean_sq - (row_means**2)

        metrics["rm/turn_level_sum"] = torch.mean(row_sums).item()
        metrics["rm/turn_level_mean"] = torch.mean(row_means).item()
        metrics["rm/turn_level_var"] = torch.mean(row_vars).item()
