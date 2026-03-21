import numpy as np
import torch
import torch.nn.functional as F
import verl
import verl.utils.torch_functional as verl_F


def compute_rloo_advantage_return(
    data: verl.DataProto, response_mask: torch.Tensor, n_samples, config
):
    # define RLOO adv estimator
    def masked_rloo(reward_tensor_original, mask_tensor):
        # Return advantage (without GAE)
        reward_tensor = reward_tensor_original.clone()
        reward_tensor[~mask_tensor] = 0
        for start_pos in range(0, reward_tensor.shape[0], n_samples):
            cur_rewards_mean = torch.cat(
                [
                    reward_tensor[pos : pos + 1][mask_tensor[pos : pos + 1]].mean(
                        dim=0, keepdim=True
                    )
                    for pos in range(start_pos, start_pos + n_samples)
                ],
                dim=0,
            )
            cur_rewards_sum = cur_rewards_mean.sum()
            cur_reward_baseline = cur_rewards_sum / (n_samples - 1)
            reward_tensor[start_pos : start_pos + n_samples][
                mask_tensor[start_pos : start_pos + n_samples]
            ] = (
                reward_tensor[start_pos : start_pos + n_samples][
                    mask_tensor[start_pos : start_pos + n_samples]
                ]
                * (n_samples / (n_samples - 1))
                - cur_reward_baseline
            )
        return reward_tensor

    # main body of the function
    reward_tensors = []

    with torch.no_grad():
        # This is for PRIME /ITPO
        if "rm_scores" in data.batch.keys() and config.algorithm.reward_dpo_coef != 0.0:
            if config.algorithm.adv_estimator == "rloo_itponorm":
                reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
                reward_mask = torch.zeros_like(response_mask, dtype=torch.bool)
                response_mask_padded = F.pad(response_mask, (0, 1), value=0)
                reward_mask = (response_mask == 1) & (response_mask_padded[:, 1:] == 0)
                proportions = data.non_tensor_batch["turn_level_proportions"]

                for metric in data.non_tensor_batch["reward_extra_info"][0].keys():
                    if "token_amount" in metric:
                        for row_id in range(reward_tensor.shape[0]):
                            reward_idxs = torch.nonzero(reward_mask[row_id]).reshape(-1)
                            dict_id = 0
                            for reward_id in reward_idxs:
                                reward_tensor[row_id, reward_id] += (
                                    data.non_tensor_batch["token_amount"][row_id]
                                    * data.non_tensor_batch["token_proportions"][
                                        row_id
                                    ]["proportions"][dict_id]
                                )
                                dict_id += 1
                    else:
                        for row_id in range(reward_tensor.shape[0]):
                            dict_id = 0
                            reward_idxs = torch.nonzero(reward_mask[row_id]).reshape(-1)
                            for reward_id in reward_idxs:
                                tmp_reward = (
                                    data.non_tensor_batch[metric][row_id]
                                    * proportions[row_id]["proportions"][dict_id]
                                )
                                reward_tensor[row_id, reward_id] += tmp_reward
                                dict_id += 1
                reward_tensors.append(
                    masked_rloo(reward_tensor, reward_mask)
                    * config.algorithm.reward_dpo_coef
                )

            elif config.algorithm.adv_estimator == "rloo_itpo":
                reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
                reward_tensor_tokens = torch.zeros_like(
                    response_mask, dtype=torch.float32
                )
                reward_mask = torch.zeros_like(response_mask, dtype=torch.bool)
                response_mask_padded = F.pad(response_mask, (0, 1), value=0)
                reward_mask = (response_mask == 1) & (response_mask_padded[:, 1:] == 0)
                proportions = data.non_tensor_batch["turn_level_proportions"]

                for metric in data.non_tensor_batch["reward_extra_info"][0].keys():
                    if "token_amount" in metric:
                        for row_id in range(reward_tensor.shape[0]):
                            reward_idxs = torch.nonzero(reward_mask[row_id]).reshape(-1)
                            dict_id = 0
                            for reward_id in reward_idxs:
                                reward_tensor_tokens[row_id, reward_id] += (
                                    data.non_tensor_batch["token_amount"][row_id]
                                    * data.non_tensor_batch["token_proportions"][
                                        row_id
                                    ]["proportions"][dict_id]
                                )
                                dict_id += 1
                    else:
                        for row_id in range(reward_tensor.shape[0]):
                            dict_id = 0
                            reward_idxs = torch.nonzero(reward_mask[row_id]).reshape(-1)
                            for reward_id in reward_idxs:
                                tmp_reward = proportions[row_id]["proportions"][dict_id]
                                reward_tensor[row_id, reward_id] += tmp_reward
                                dict_id += 1
                reward_tensors.append(
                    masked_rloo(reward_tensor_tokens, reward_mask)
                    * config.algorithm.reward_dpo_coef
                )
                reward_tensors.append(
                    masked_rloo(reward_tensor, reward_mask)
                    * config.algorithm.reward_dpo_coef
                )

            else:
                reward_tensor = data.batch["rm_scores"]
                reward_tensor_tokens = torch.zeros_like(
                    response_mask, dtype=torch.float32
                )
                reward_mask = response_mask.bool()
                reward_mask_tokens = torch.zeros_like(response_mask, dtype=torch.bool)
                response_mask_padded = F.pad(response_mask, (0, 1), value=0)
                reward_mask_tokens = (response_mask == 1) & (
                    response_mask_padded[:, 1:] == 0
                )

                reward_tensors.append(
                    masked_rloo(reward_tensor, reward_mask)
                    * config.algorithm.reward_dpo_coef
                )

                for metric in data.non_tensor_batch["reward_extra_info"][0].keys():
                    if "token_amount" in metric:
                        for row_id in range(reward_tensor.shape[0]):
                            reward_idxs = torch.nonzero(
                                reward_mask_tokens[row_id]
                            ).reshape(-1)
                            dict_id = 0
                            for reward_id in reward_idxs:
                                reward_tensor_tokens[row_id, reward_id] += (
                                    data.non_tensor_batch["token_amount"][row_id]
                                    * data.non_tensor_batch["token_proportions"][
                                        row_id
                                    ]["proportions"][dict_id]
                                )
                                dict_id += 1
                reward_tensors.append(
                    masked_rloo(reward_tensor_tokens, reward_mask_tokens)
                    * config.algorithm.reward_dpo_coef
                )

        if "scores" in data.batch.keys() and config.algorithm.reward_gt_coef != 0.0:

            reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
            reward_mask = torch.zeros_like(response_mask, dtype=torch.bool)

            prompt_ids = data.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(
                -1
            )
            response_mask_padded = F.pad(response_mask, (0, 1), value=0)
            temp_reward_mask = (response_mask == 1) & (response_mask_padded[:, 1:] == 0)
            cumsum_mask = temp_reward_mask.cumsum(dim=-1)
            total_sum = temp_reward_mask.sum(dim=-1, keepdim=True)
            reward_mask = (
                (temp_reward_mask == 1) & (cumsum_mask == total_sum) & (total_sum > 0)
            )
            reward_tensor[reward_mask] = data.batch["scores"]
            reward_tensors.append(
                masked_rloo(reward_tensor, reward_mask)
                * config.algorithm.reward_gt_coef
            )

        final_reward_tensor = sum(reward_tensors)

        returns = (
            (final_reward_tensor * response_mask)
            .flip(dims=[-1])
            .cumsum(dim=-1)
            .flip(dims=[-1])
        )
        advantages = returns.clone()
        advantages = verl_F.masked_whiten(advantages, response_mask)

        return advantages, returns


def calculate_segment_proportions(predictions, mask, temperature=0.7):
    n, d = predictions.shape

    mask_padded = F.pad(mask, (1, 0), value=0)  # (n, d+1)
    diff = mask_padded[:, 1:] - mask_padded[:, :-1]
    starts = (diff == 1).long()

    num_segments_per_row = starts.sum(dim=1)  # shape: (n,)
    intra_row_ids = starts.cumsum(dim=1) * mask.long()
    row_offsets = torch.cat(
        [
            torch.zeros(1, device=mask.device, dtype=torch.long),
            num_segments_per_row.cumsum(dim=0)[:-1],
        ]
    )

    global_segment_ids = (intra_row_ids + row_offsets.unsqueeze(1)) - 1
    flat_preds = predictions.view(-1)
    flat_mask = mask.view(-1).bool()
    flat_ids = global_segment_ids.view(-1)

    valid_preds = flat_preds[flat_mask]
    valid_ids = flat_ids[flat_mask]
    total_segments = num_segments_per_row.sum().item()
    segment_sums = torch.zeros(
        total_segments, device=predictions.device, dtype=predictions.dtype
    )
    segment_counts = torch.zeros(
        total_segments, device=predictions.device, dtype=predictions.dtype
    )
    segment_sums.index_add_(0, valid_ids, valid_preds)
    segment_counts.index_add_(0, valid_ids, torch.ones_like(valid_preds))
    # segment_means = segment_sums / (segment_counts + 1e-8)
    split_sections = num_segments_per_row.cpu().tolist()
    # row_means_tuple = torch.split(segment_means, split_sections)
    row_sums_tuple = torch.split(segment_sums, split_sections)
    row_counts_tuple = torch.split(segment_counts, split_sections)
    final_results = []
    length_probs_list = []

    for sums, counts in zip(row_sums_tuple, row_counts_tuple):
        if sums.numel() > 0:
            row_total_count = counts.sum()
            # row_avg_count = counts.mean()
            # probs = F.softmax(means * row_avg_count / temperature, dim=0)
            probs = F.softmax(sums / temperature, dim=0)
            l_probs = counts / row_total_count
            final_results.append({"proportions": probs.detach().cpu().numpy()})
            length_probs_list.append({"proportions": l_probs.detach().cpu().numpy()})
        else:
            final_results.append({"proportions": np.array([])})
            length_probs_list.append({"proportions": np.array([])})

    return np.array(final_results), np.array(length_probs_list)


def calculate_segment_proportions_itponorm(predictions, mask, temperature=0.7):
    n, d = predictions.shape

    mask_padded = F.pad(mask, (1, 0), value=0)  # (n, d+1)
    diff = mask_padded[:, 1:] - mask_padded[:, :-1]
    starts = (diff == 1).long()

    num_segments_per_row = starts.sum(dim=1)  # shape: (n,)
    intra_row_ids = starts.cumsum(dim=1) * mask.long()
    row_offsets = torch.cat(
        [
            torch.zeros(1, device=mask.device, dtype=torch.long),
            num_segments_per_row.cumsum(dim=0)[:-1],
        ]
    )

    global_segment_ids = (intra_row_ids + row_offsets.unsqueeze(1)) - 1
    flat_preds = predictions.view(-1)
    flat_mask = mask.view(-1).bool()
    flat_ids = global_segment_ids.view(-1)

    valid_preds = flat_preds[flat_mask]
    valid_ids = flat_ids[flat_mask]
    total_segments = num_segments_per_row.sum().item()
    segment_sums = torch.zeros(
        total_segments, device=predictions.device, dtype=predictions.dtype
    )
    segment_counts = torch.zeros(
        total_segments, device=predictions.device, dtype=predictions.dtype
    )
    segment_sums.index_add_(0, valid_ids, valid_preds)
    segment_counts.index_add_(0, valid_ids, torch.ones_like(valid_preds))
    segment_means = segment_sums / (segment_counts)
    split_sections = num_segments_per_row.cpu().tolist()
    row_means_tuple = torch.split(segment_means, split_sections)
    row_counts_tuple = torch.split(segment_counts, split_sections)
    final_results = []
    length_probs_list = []

    global_avg_count = segment_counts.mean()

    for means, counts in zip(row_means_tuple, row_counts_tuple):
        if means.numel() > 0:
            row_total_count = counts.sum()
            probs = F.softmax(means * global_avg_count / temperature, dim=0)
            l_probs = counts / row_total_count
            final_results.append({"proportions": probs.detach().cpu().numpy()})
            length_probs_list.append({"proportions": l_probs.detach().cpu().numpy()})
        else:
            final_results.append({"proportions": np.array([])})
            length_probs_list.append({"proportions": np.array([])})

    return np.array(final_results), np.array(length_probs_list)


def calculate_segment_proportions_itpo(predictions, mask, temperature=0.7):
    n, d = predictions.shape

    mask_padded = F.pad(mask, (1, 0), value=0)  # (n, d+1)
    diff = mask_padded[:, 1:] - mask_padded[:, :-1]
    starts = (diff == 1).long()

    num_segments_per_row = starts.sum(dim=1)  # shape: (n,)
    intra_row_ids = starts.cumsum(dim=1) * mask.long()
    row_offsets = torch.cat(
        [
            torch.zeros(1, device=mask.device, dtype=torch.long),
            num_segments_per_row.cumsum(dim=0)[:-1],
        ]
    )

    global_segment_ids = (intra_row_ids + row_offsets.unsqueeze(1)) - 1
    flat_preds = predictions.view(-1)
    flat_mask = mask.view(-1).bool()
    flat_ids = global_segment_ids.view(-1)

    valid_preds = flat_preds[flat_mask]
    valid_ids = flat_ids[flat_mask]
    total_segments = num_segments_per_row.sum().item()
    segment_sums = torch.zeros(
        total_segments, device=predictions.device, dtype=predictions.dtype
    )
    segment_counts = torch.zeros(
        total_segments, device=predictions.device, dtype=predictions.dtype
    )
    segment_sums.index_add_(0, valid_ids, valid_preds)
    segment_counts.index_add_(0, valid_ids, torch.ones_like(valid_preds))
    segment_means = segment_sums / (segment_counts)
    split_sections = num_segments_per_row.cpu().tolist()
    row_sums_tuple = torch.split(segment_sums, split_sections)
    row_means_tuple = torch.split(segment_means, split_sections)
    row_counts_tuple = torch.split(segment_counts, split_sections)
    final_results = []
    length_probs_list = []

    global_avg_count = segment_counts.mean()

    for sums, counts in zip(row_sums_tuple, row_counts_tuple):
        if sums.numel() > 0:
            row_total_count = counts.sum()
            l_probs = counts / row_total_count
            final_results.append({"proportions": sums.detach().cpu().numpy()})
            length_probs_list.append({"proportions": l_probs.detach().cpu().numpy()})
        else:
            final_results.append({"proportions": np.array([])})
            length_probs_list.append({"proportions": np.array([])})

    return np.array(final_results), np.array(length_probs_list)


def calculate_token_proportions(mask):
    n, d = mask.shape

    mask_padded = F.pad(mask, (1, 0), value=0)  # (n, d+1)
    diff = mask_padded[:, 1:] - mask_padded[:, :-1]
    starts = (diff == 1).long()

    num_segments_per_row = starts.sum(dim=1)  # shape: (n,)
    intra_row_ids = starts.cumsum(dim=1) * mask.long()
    row_offsets = torch.cat(
        [
            torch.zeros(1, device=mask.device, dtype=torch.long),
            num_segments_per_row.cumsum(dim=0)[:-1],
        ]
    )

    global_segment_ids = (intra_row_ids + row_offsets.unsqueeze(1)) - 1
    flat_mask = mask.view(-1).bool()
    flat_ids = global_segment_ids.view(-1)

    valid_ids = flat_ids[flat_mask]
    total_segments = num_segments_per_row.sum().item()

    segment_counts = torch.zeros(total_segments, device=mask.device, dtype=torch.float)
    segment_counts.index_add_(0, valid_ids, torch.ones_like(valid_ids.float()))

    split_sections = num_segments_per_row.cpu().tolist()

    row_counts_tuple = torch.split(segment_counts, split_sections)
    length_probs_list = []

    for counts in row_counts_tuple:

        row_total_count = counts.sum()
        l_probs = counts / row_total_count
        length_probs_list.append({"proportions": l_probs.detach().cpu().numpy()})

    return np.array(length_probs_list)
