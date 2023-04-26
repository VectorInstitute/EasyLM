"""
Shuffle and split alpaca_data.json into train, validation, and test.
Save the output in JSONLines format. Each line will be a valid JSON dictionary with 
no indent.
"""
import argparse
from os.path import exists
import json

import jax
import numpy as np
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("input_json")
parser.add_argument("prng_seed", type=int)
parser.add_argument("output_jsonl_train")
parser.add_argument("output_jsonl_val")
parser.add_argument("output_jsonl_test")

args = parser.parse_args()
input_json_path = args.input_json
output_jsonl_path_train = args.output_jsonl_train
output_jsonl_path_val = args.output_jsonl_val
output_jsonl_path_test = args.output_jsonl_test
prng_seed = int(args.prng_seed)  # JAX PRNG Key for reproducibility
# Complain if output file exists.
assert not exists(output_jsonl_path_train)
assert not exists(output_jsonl_path_val)
assert not exists(output_jsonl_path_test)

prng_key_array = jax.random.PRNGKey(prng_seed)

with open(input_json_path, "r") as input_json_file:
    examples = json.load(input_json_file)

assert isinstance(examples, list)
num_examples = len(examples)
indices = np.arange(num_examples)
shuffled_indices = jax.random.permutation(prng_key_array, indices)

# Convert examples into a list of JSON-serialized strings.
example_json_list = [(json.dumps(example) + "\n") for example in tqdm(examples)]

output_examples = []
for index in tqdm(shuffled_indices):
    output_examples.append(example_json_list[index])

num_train_examples = int(num_examples * 0.9)
num_validation_examples = int(num_examples * 0.05)

# Pretty-print JSON.
with open(output_jsonl_path_train, "w") as output_jsonl_file:
    index_a = 0
    index_b = num_train_examples
    output_jsonl_file.writelines(output_examples[index_a:index_b])

with open(output_jsonl_path_val, "w") as output_jsonl_file:
    index_a = num_train_examples
    index_b = num_train_examples + num_validation_examples
    output_jsonl_file.writelines(output_examples[index_a:index_b])

with open(output_jsonl_path_test, "w") as output_jsonl_file:
    index_a = num_train_examples + num_validation_examples
    output_jsonl_file.writelines(output_examples[index_a:])
