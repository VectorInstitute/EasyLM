import os
import math
from typing import Any, Mapping, Text, Tuple, Union, NamedTuple
from functools import partial
import re
import dataclasses
import random

import dill
import flax
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as PS
from jax.sharding import Mesh
from jax.experimental.pjit import with_sharding_constraint as _with_sharding_constraint
from jax.experimental.pjit import pjit
from jax.interpreters import pxla
import numpy as np
from absl import logging
from flax import jax_utils
from flax.training.train_state import TrainState
from flax.core import FrozenDict
import optax
from transformers import FlaxLogitsWarper


class JaxRNG(object):
    """ A convenient stateful Jax RNG wrapper. Can be used to wrap RNG inside
        pure function.
    """

    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    """ JIT traceable version of FlaxLogitsWarper that performs temperature scaling."""
    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, input_ids, scores, cur_len):
        return scores / jnp.clip(self.temperature, a_min=1e-8)


def make_shard_and_gather_fns(partition_specs, dtype_specs=None):
    """ Create pytree of sharding and gathering functions from pytree of
        partition specs.
    """
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)

    def make_to_dtype_fn(dtype_spec):
        def to_dtype(tensor):
            if dtype_specs in float_dtypes and getattr(tensor, 'dtype', None) in float_dtypes:
                # Convert all float tensors to the same dtype
                return tensor.astype(dtype_specs)
            elif hasattr(dtype_spec, 'dtype') and hasattr(tensor, 'dtype'):
                return tensor.astype(dtype_spec.dtype)
            return tensor
        return to_dtype

    def make_shard_fn(partition_spec, dtype_spec=None):
        jax_shard_function = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=None,
            out_shardings=partition_spec
        )
        def shard_fn(tensor):
            return jax_shard_function(tensor).block_until_ready()
        return shard_fn

    def make_gather_fn(partition_spec, dtype_spec=None):
        jax_gather_fn = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=partition_spec,
            out_shardings=None
        )
        def gather_fn(tensor):
            return jax.device_get(jax_gather_fn(tensor))
        return gather_fn

    if dtype_specs is None or dtype_specs in float_dtypes:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs)
    else:
        shard_fns = jax.tree_util.tree_map(
            make_shard_fn, partition_specs, dtype_specs
        )
        gather_fns = jax.tree_util.tree_map(
            make_gather_fn, partition_specs, dtype_specs
        )
    return shard_fns, gather_fns


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def get_jax_mesh(axis_dims, names):
    if ':' in axis_dims:
        dims = []
        dim_names = []
        for axis in axis_dims.split(','):
            name, dim = axis.split(':')
            assert name in names
            dims.append(int(dim))
            dim_names.append(name)
        assert(set(dim_names) == set(names))
    else:
        dims = [int(x) for x in axis_dims.split(',')]
        dim_names = names
    assert len(dims) == len(names)
    return Mesh(np.array(jax.devices()).reshape(dims), dim_names)


def names_in_current_mesh(*names):
    """ Check if current mesh axes contain these names. """
    mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
    return set(names) <= set(mesh_axis_names)


def get_names_from_parition_spec(partition_specs):
    """ Return axis names from partition specs. """
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_parition_spec(item))

    return list(names)


def with_sharding_constraint(x, partition_specs):
    """ A smarter version of with_sharding_constraint that only applies the
        constraint if the current mesh contains the axes in the partition specs.
    """
    axis_names = get_names_from_parition_spec(partition_specs)
    if names_in_current_mesh(*axis_names):
        x = _with_sharding_constraint(x, partition_specs)
    return x


def wrap_function_with_rng(rng):
    """ To be used as decorator, automatically bookkeep a RNG for the wrapped function. """
    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)
        return wrapped
    return wrap_function


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


def get_metrics(metrics, unreplicate=False, stack=False):
    if unreplicate:
        metrics = flax.jax_utils.unreplicate(metrics)
    metrics = jax.device_get(metrics)
    if stack:
        return jax.tree_map(lambda *args: np.stack(args), *metrics)
    else:
        return {key: float(val) for key, val in metrics.items()}


def mse_loss(val, target, valid=None):
    if valid is None:
        valid = jnp.ones((*target.shape[:2], 1))
    valid = valid.astype(jnp.float32)
    loss = jnp.mean(
        jnp.where(
            valid > 0.0,
            jnp.square(val - target),
            0.0
        )
    )
    return loss


def cross_entropy_loss(logits, labels, smoothing_factor=0.):
    num_classes = logits.shape[-1]
    if labels.dtype == jnp.int32 or labels.dtype == jnp.int64:
        labels = jax.nn.one_hot(labels, num_classes)
    if smoothing_factor > 0.:
        labels = labels * (1. - smoothing_factor) + smoothing_factor / num_classes
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(logp * labels, axis=-1))


def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)

    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == tokens,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy

def rrhf_loss(pos_logits, pos_tokens, pos_valid, neg_logits, neg_tokens, neg_valid):
    pos_loss, _ = cross_entropy_loss_and_accuracy(pos_logits, pos_tokens, pos_valid)
    neg_loss, _ = cross_entropy_loss_and_accuracy(neg_logits, neg_tokens, neg_valid)
    
    ranking_loss = jnp.maximum(pos_loss - neg_loss, jnp.array(0.0))
    ranking_correct = jnp.where(
        pos_loss < neg_loss,
        jnp.array(True),
        jnp.array(False)
    )
    ranking_accuracy = jnp.mean(ranking_correct)

    total_loss = ranking_loss + pos_loss
    return total_loss, ranking_accuracy

def dpo_loss(yw_logits, yw_logits_ref, yw_tokens, yw_valid, yl_logits, yl_logits_ref, yl_tokens, yl_valid, kl_scale):
    """
    Rewrite fractions in the original DPO formulation 
    (equation 7 in v1 of the paper) using the property of logarithm:
    log(a/b) = log(a) - log(b)

    yw: continuations in each pair that are preferred more.
    yl: continuations in each pair that are preferred less.
    """
    policy_yw_log_prob, policy_yw_acc = cross_entropy_loss_and_accuracy(yw_logits, yw_tokens, yw_valid)
    policy_yl_log_prob, policy_yl_acc = cross_entropy_loss_and_accuracy(yl_logits, yl_tokens, yl_valid)

    ref_yw_log_prob, ref_yw_acc = cross_entropy_loss_and_accuracy(yw_logits_ref, yw_tokens, yw_valid)
    ref_yl_log_prob, ref_yl_acc = cross_entropy_loss_and_accuracy(yl_logits_ref, yl_tokens, yl_valid)

    policy_log_ratios = policy_yw_log_prob - policy_yl_log_prob
    ref_log_ratios = ref_yw_log_prob - ref_yl_log_prob

    losses = -jax.nn.log_sigmoid(kl_scale * (policy_log_ratios - ref_log_ratios))
    rewards = kl_scale * (policy_log_ratios - ref_log_ratios)
    reward_modelling_accuracy = jnp.mean(policy_yw_log_prob > ref_yl_log_prob)

    return losses, (reward_modelling_accuracy, rewards)
    

def global_norm(tree):
    """ Return the global L2 norm of a pytree. """
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    flattened, _ = jax.flatten_util.ravel_pytree(squared)
    return jnp.sqrt(jnp.sum(flattened))


def average_metrics(metrics):
    return jax.tree_map(
        lambda *args: jnp.mean(jnp.stack(args)),
        *metrics
    )


def get_float_dtype_by_name(dtype):
    return {
        'bf16': jnp.bfloat16,
        'fp16': jnp.float16,
        'fp32': jnp.float32,
        'fp64': jnp.float64,
    }[dtype]


def float_tensor_to_dtype(tensor, dtype):
    if dtype is None or dtype == '':
        return tensor
    if isinstance(dtype, str):
        dtype = get_float_dtype_by_name(dtype)
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    if getattr(tensor, 'dtype', None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


def float_to_dtype(tree, dtype):
    return jax.tree_util.tree_map(
        partial(float_tensor_to_dtype, dtype=dtype), tree
    )


def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


def tree_path_to_string(path, sep=None):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)


def flatten_tree(xs, is_leaf=None, sep=None):
    flattened, _ = jax.tree_util.tree_flatten_with_path(xs, is_leaf=is_leaf)
    output = {}
    for key, val in flattened:
        output[tree_path_to_string(key, sep=sep)] = val
    return output


def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
        tree, *rest,
        is_leaf=is_leaf
    )


def match_partition_rules(rules, params):
    """ Returns a pytree of PartitionSpec according to rules. Supports handling
        Flax TrainState and Optax optimizer state.
    """
    def get_partition_spec(name, leaf):
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            """ Don't partition scalar values. """
            return PS()
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                return ps
        raise ValueError(f'Partition rule not found for param: {name}')
    return named_tree_map(get_partition_spec, params, sep='/')


def get_weight_decay_mask(exclusions):
    """ Return a weight decay mask function that computes the pytree masks
        according to the given exclusion rules.
    """
    def decay(name, _):
        for rule in exclusions:
            if re.search(rule, name) is not None:
                return False
        return True

    def weight_decay_mask(params):
        return named_tree_map(decay, params, sep='/')

    return weight_decay_mask


def tree_apply(fns, tree):
    """ Apply a pytree of functions to the pytree. """
    return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)

