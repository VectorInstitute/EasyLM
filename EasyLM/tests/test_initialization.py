import unittest

from os import environ
import logging

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit

import optax

from ..models.llama.llama_train import DPOTrainState
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.models.llama.llama_model import (
    LLaMAConfig,
    FlaxLLaMAForCausalLM,
    FlaxLLaMAForCausalLMModule,
)
from EasyLM.jax_utils import (
    JaxRNG,
    next_rng,
    match_partition_rules,
    make_shard_and_gather_fns,
    set_random_seed,
)

PARAMS_PATH = environ.get("TEST_PARAMS_PATH", "params::/data/models/open-llama-3b")
MODEL_VARIATION = environ.get("TEST_MODEL_VARIATION", "3b")
SEQ_LENGTH = int(environ.get("TEST_SEQ_LENGTH", 12))

logger = logging.getLogger(__name__)


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(0)
        optimizer = optax.adamw(1e-5)

        llama_config = LLaMAConfig.load_config(MODEL_VARIATION)
        cls.model = FlaxLLaMAForCausalLMModule(llama_config)  # type: ignore

        cls.checkpointer = StreamingCheckpointer(
            StreamingCheckpointer.get_default_config(),
            "/dev/shm/",
            enable=jax.process_index() == 0,
        )

        def create_trainstate_from_params(params, params_ref=None):
            return DPOTrainState.create(
                params=params, params_ref=params_ref, tx=optimizer, apply_fn=None
            )

        def init_fn(rng):
            rng_generator = JaxRNG(rng)
            params = cls.model.init(
                input_ids=jnp.zeros((4, SEQ_LENGTH), dtype=jnp.int32),
                position_ids=jnp.zeros((4, SEQ_LENGTH), dtype=jnp.int32),
                attention_mask=jnp.ones((4, SEQ_LENGTH), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
            params_ref = cls.model.init(
                input_ids=jnp.zeros((4, SEQ_LENGTH), dtype=jnp.int32),
                position_ids=jnp.zeros((4, SEQ_LENGTH), dtype=jnp.int32),
                attention_mask=jnp.ones((4, SEQ_LENGTH), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
            return DPOTrainState.create(
                params=params, params_ref=params_ref, tx=optimizer, apply_fn=None
            )

        train_state_shapes = jax.eval_shape(init_fn, next_rng())
        train_state_partition = match_partition_rules(
            LLaMAConfig.get_partition_rules(), train_state_shapes
        )

        sharded_create_trainstate_from_params = pjit(
            create_trainstate_from_params,
            in_shardings=(
                train_state_partition.params,
                train_state_partition.params_ref,
            ),  # type: ignore
            out_shardings=train_state_partition,
            donate_argnums=(0,),
        )
        shard_fns, gather_fns = make_shard_and_gather_fns(
            train_state_partition, train_state_shapes
        )
        mesh = LLaMAConfig.get_jax_mesh("1,1,-1")
        with mesh:
            _, restored_params = cls.checkpointer.load_trainstate_checkpoint(
                PARAMS_PATH, train_state_shapes, shard_fns
            )
            _, restored_params_ref = cls.checkpointer.load_trainstate_checkpoint(
                PARAMS_PATH, train_state_shapes, shard_fns
            )
            cls.example_train_state = sharded_create_trainstate_from_params(
                restored_params, restored_params_ref
            )
            del restored_params
            del restored_params_ref

    def setUp(self):
        return

    def test_train_state_shape(self):
        train_state = InitializationTests.example_train_state
        train_state_shape = jax.tree_util.tree_map(jnp.shape, train_state)
        print(train_state_shape)
