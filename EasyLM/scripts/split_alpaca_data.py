"""
Shuffle and split alpaca_data.json into train, validation, and test.
"""
import argparse
from os.path import exists
import json

import jax
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("input_json")
parser.add_argument("prng_seed", type=int)
parser.add_argument("output_json_train")
parser.add_argument("output_json_val")
parser.add_argument("output_json_test")

args = parser.parse_args()
input_json_path = args.input_json
output_json_path_train = args.output_json_train
output_json_path_val = args.output_json_val
output_json_path_test = args.output_json_test
prng_seed = int(args.prng_seed)  # JAX PRNG Key for reproducibility
# Complain if output file exists.
assert not exists(output_json_path_train)
assert not exists(output_json_path_val)
assert not exists(output_json_path_test)

prng_key_array = jax.random.PRNGKey(prng_seed)

with open(input_json_path, "r") as input_json_file:
    examples = json.load(input_json_file)

assert isinstance(examples, list)
num_examples = len(examples)
indices = np.arange(num_examples)
shuffled_indices = jax.random.permutation(prng_key_array, indices)

output_examples = []
for index in shuffled_indices:
    output_examples.append(examples[index])

num_train_examples = int(num_examples * 0.9)
num_validation_examples = int(num_examples * 0.05)

# Pretty-print JSON.
with open(output_json_path_train, "w") as output_json_file:
    index_a = 0
    index_b = num_train_examples
    json.dump(output_examples[index_a:index_b], output_json_file, indent=2)

with open(output_json_path_val, "w") as output_json_file:
    index_a = num_train_examples
    index_b = num_train_examples + num_validation_examples
    json.dump(output_examples[index_a:index_b], output_json_file, indent=2)

with open(output_json_path_test, "w") as output_json_file:
    index_a = num_train_examples + num_validation_examples
    json.dump(output_examples[index_a:], output_json_file, indent=2)
