import torch
from torch.utils.data import DataLoader
from wilds.common.data_loaders import GroupSampler, WeightedRandomSampler


# modified from
# https://github.com/gibipara92/learning-explanations-hard-to-vary/blob/e45a7b01b87f0bc2f9a583a2c051e901efd66800/and_mask/run_cifar.py#L76
def get_grads(
    agreement_threshold,
    batch_size,
    loss_fn,
    n_agreement_envs,
    params,
    output,
    target,
    method,
    scale_grad_inverse_sparsity,
):
    """
    Use the and mask or the geometric mean to put gradients together.
    Modifies gradients wrt params in place (inside param.grad).
    Returns mean loss and masks for diagnostics.
    Args:
        agreement_threshold: a float between 0 and 1 (tau in the paper).
            If 1 -> requires complete sign agreement across all environments (everything else gets masked out),
             if 0 it requires no agreement, and it becomes essentially standard sgd if method == 'and_mask'. Values
             in between are fractional ratios of agreement.
        batch_size: The original batch size per environment. Needed to perform reshaping, so that grads can be computed
            independently per each environment.
        loss_fn: the loss function
        n_agreement_envs: the number of environments that were stacked in the inputs. Needed to perform reshaping.
        params: the model parameters
        output: the output of the model, where inputs were *all examples from all envs stacked in a big batch*. This is
            done to at least compute the forward pass somewhat efficiently.
        method: 'and_mask' or 'geom_mean'.
        scale_grad_inverse_sparsity: If True, rescale the magnitude of the gradient components that survived the mask,
            layer-wise, to compensate for the reduce overall magnitude after masking and/or geometric mean.
    Returns:
        mean_loss: mean loss across environments
        masks: a list of the binary masks (every element corresponds to one layer) applied to the gradient.
    """

    param_gradients = [[] for _ in params]
    outputs = output.view(n_agreement_envs, batch_size, -1)
    targets = target.view(n_agreement_envs, batch_size, -1)

    outputs = outputs.squeeze(-1)
    targets = targets.squeeze(-1)

    total_loss = 0.0
    for env_outputs, env_targets in zip(outputs, targets):
        env_loss = loss_fn(env_outputs, env_targets)
        total_loss += env_loss
        env_grads = torch.autograd.grad(env_loss, params, retain_graph=True)
        for grads, env_grad in zip(param_gradients, env_grads):
            grads.append(env_grad)
    mean_loss = total_loss / n_agreement_envs
    assert len(param_gradients) == len(params)
    assert len(param_gradients[0]) == n_agreement_envs

    masks = []
    avg_grads = []
    weights = []
    for param, grads in zip(params, param_gradients):
        assert len(grads) == n_agreement_envs
        grads = torch.stack(grads, dim=0)
        assert grads.shape == (n_agreement_envs,) + param.shape
        grad_signs = torch.sign(grads)
        mask = torch.mean(grad_signs, dim=0).abs() >= agreement_threshold
        mask = mask.to(torch.float32)
        assert mask.numel() == param.numel()
        avg_grad = torch.mean(grads, dim=0)
        assert mask.shape == avg_grad.shape

        if method == "and_mask":
            mask_t = mask.sum() / mask.numel()
            param.grad = mask * avg_grad
            if scale_grad_inverse_sparsity:
                param.grad *= 1.0 / (1e-10 + mask_t)
        elif method == "geom_mean":
            prod_grad = torch.sign(avg_grad) * torch.exp(
                torch.sum(torch.log(torch.abs(grads) + 1e-10), dim=0) / n_agreement_envs
            )
            mask_t = mask.sum() / mask.numel()
            param.grad = mask * prod_grad
            if scale_grad_inverse_sparsity:
                param.grad *= 1.0 / (1e-10 + mask_t)
        else:
            raise ValueError()

        weights.append(param.data)
        avg_grads.append(avg_grad)
        masks.append(mask)

    return mean_loss, masks


# This is like the WILDS get_train_loader but we can set drop_last for the DataLoaders ourselves.
def get_train_loader(
    loader,
    dataset,
    batch_size,
    uniform_over_groups=None,
    grouper=None,
    distinct_groups=True,
    n_groups_per_batch=None,
    **loader_kwargs,
):
    """
    Constructs and returns the data loader for training.
    Args:
        - loader (str): Loader type. 'standard' for standard loaders and 'group' for group loaders,
                        which first samples groups and then samples a fixed number of examples belonging
                        to each group.
        - dataset (WILDSDataset or WILDSSubset): Data
        - batch_size (int): Batch size
        - uniform_over_groups (None or bool): Whether to sample the groups uniformly or according
                                              to the natural data distribution.
                                              Setting to None applies the defaults for each type of loaders.
                                              For standard loaders, the default is False. For group loaders,
                                              the default is True.
        - grouper (Grouper): Grouper used for group loaders or for uniform_over_groups=True
        - distinct_groups (bool): Whether to sample distinct_groups within each minibatch for group loaders.
        - n_groups_per_batch (int): Number of groups to sample in each minibatch for group loaders.
        - loader_kwargs: kwargs passed into torch DataLoader initialization.
    Output:
        - data loader (DataLoader): Data loader.
    """
    if loader == "standard":
        if uniform_over_groups is None or not uniform_over_groups:
            return DataLoader(
                dataset,
                shuffle=True,  # Shuffle training dataset
                sampler=None,
                collate_fn=dataset.collate,
                batch_size=batch_size,
                **loader_kwargs,
            )
        else:
            assert grouper is not None
            groups, group_counts = grouper.metadata_to_group(
                dataset.metadata_array, return_counts=True
            )
            group_weights = 1 / group_counts
            weights = group_weights[groups]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
            return DataLoader(
                dataset,
                shuffle=False,  # The WeightedRandomSampler already shuffles
                sampler=sampler,
                collate_fn=dataset.collate,
                batch_size=batch_size,
                **loader_kwargs,
            )

    elif loader == "group":
        if uniform_over_groups is None:
            uniform_over_groups = True
        assert grouper is not None
        assert n_groups_per_batch is not None
        if n_groups_per_batch > grouper.n_groups:
            raise ValueError(
                f"n_groups_per_batch was set to {n_groups_per_batch} but there are only {grouper.n_groups} groups specified."
            )

        group_ids = grouper.metadata_to_group(dataset.metadata_array)
        batch_sampler = GroupSampler(
            group_ids=group_ids,
            batch_size=batch_size,
            n_groups_per_batch=n_groups_per_batch,
            uniform_over_groups=uniform_over_groups,
            distinct_groups=distinct_groups,
        )

        return DataLoader(
            dataset,
            shuffle=None,
            sampler=None,
            collate_fn=dataset.collate,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
