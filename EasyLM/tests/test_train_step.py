import unittest

import jax.numpy as jnp

from EasyLM.jax_utils import rrhf_loss

from EasyLM.models.llama.llama_model import LLaMAConfig
from EasyLM.data import (
    DatasetFactory,
    HuggingfaceDataset,
    TextProcessor,
    ContrastiveDatasetFactory,
)

SEQ_LENGTH = 512


class RRHFLossTestCases(unittest.TestCase):
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
        batch = RRHFLossTestCases.batch
        vocab_size = RRHFLossTestCases.vocab_size
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

