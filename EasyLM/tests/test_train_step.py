import logging
import unittest

import jax.numpy as jnp

from EasyLM.jax_utils import dpo_loss, rrhf_loss

from EasyLM.models.llama.llama_model import LLaMAConfig
from EasyLM.data import (
    DatasetFactory,
    HuggingfaceDataset,
    TextProcessor,
    ContrastiveDatasetFactory,
)

SEQ_LENGTH = 512

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class RMLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer_config = LLaMAConfig.get_tokenizer_config(
            {"vocab_file": "/data/models/llama/tokenizer.model"}
        )
        dataset_config = ContrastiveDatasetFactory.get_default_config(
            {
                "type": "huggingface",
                "huggingface_dataset": HuggingfaceDataset.get_default_config(
                    {
                        "path": "Dahoas/full-hh-rlhf",
                        "split": "train",
                        "streaming": False,
                        "seq_length": SEQ_LENGTH,
                        "batch_size": 2,
                    }
                ),
                "positive_text_processor": TextProcessor.get_default_config(
                    {"fields": "[prompt],chosen"}
                ),
                "negative_text_processor": TextProcessor.get_default_config(
                    {"fields": "[prompt],rejected"}
                ),
            }
        )

        tokenizer = LLaMAConfig.get_tokenizer(tokenizer_config)
        dataset = ContrastiveDatasetFactory.load_dataset(dataset_config, tokenizer)
        cls.batch, _ = next(iter(dataset))
        cls.vocab_size = tokenizer.vocab_size

    def test_rrhf_loss(self):
        batch = RMLossTests.batch
        vocab_size = RMLossTests.vocab_size
        pos_logits = jnp.zeros((1, SEQ_LENGTH - 1, vocab_size))
        neg_logits = jnp.zeros((1, SEQ_LENGTH - 1, vocab_size))
        example_loss, example_accuracy = rrhf_loss(
            pos_logits,
            batch["positive_tokens"][:, 1:],
            batch["positive_loss_masks"][:, 1:],
            neg_logits,
            batch["negative_tokens"][:, 1:],
            batch["negative_loss_masks"][:, 1:],
        )
        print(example_loss.shape, example_loss)
        print(example_accuracy.shape, example_accuracy)

    def test_dpo_loss(self):
        batch = RMLossTests.batch
        vocab_size = RMLossTests.vocab_size
        yw_logits = jnp.zeros((1, SEQ_LENGTH - 1, vocab_size)).at[..., 0, 0].set(1e3)
        print(yw_logits)
        yw_logits_ref = jnp.zeros((1, SEQ_LENGTH - 1, vocab_size))
        yl_logits = jnp.zeros((1, SEQ_LENGTH - 1, vocab_size))
        yl_logits_ref = jnp.zeros((1, SEQ_LENGTH - 1, vocab_size))
        logger.debug("positive_loss_masks: {}".format(batch["positive_loss_masks"][:, 1:]))
        logger.debug("positive_tokens: {}".format(jnp.ones_like(batch["positive_tokens"][:, 1:])))

        example_loss, example_rm_accuracy, example_reward = dpo_loss(
            yw_logits,
            yw_logits_ref,
            jnp.ones_like(batch["positive_tokens"][:, 1:]),
            jnp.ones_like(batch["positive_loss_masks"][:, 1:]),
            yl_logits,
            yl_logits_ref,
            jnp.zeros_like(batch["negative_tokens"][:, 1:]),
            jnp.ones_like(batch["negative_loss_masks"][:, 1:]),
            jnp.array(0.1)
        )
        logging.info("example_loss: {}".format(example_loss))
        logging.info("example_rm_accuracy: {}".format(example_rm_accuracy))
        logging.info("example_reward: {}".format(example_reward))
        logging.info("example_loss.shape: {}".format(example_loss.shape))
        logging.info("example_rm_accuracy.shape: {}".format(example_rm_accuracy.shape))
        logging.info("example_reward.shape: {}".format(example_reward.shape))