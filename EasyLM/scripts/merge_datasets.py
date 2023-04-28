import argparse
from os.path import exists

from datasets import concatenate_datasets, DatasetDict, load_dataset
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", required=True)
parser.add_argument("input_dataset_names", nargs="+")
args = parser.parse_args()

output_path = args.output_path
input_dataset_names = args.input_dataset_names
assert not exists(output_path)

output_dataset_dict = DatasetDict()
input_dataset_dicts = []
for dataset_name in tqdm(input_dataset_names):
    dataset_dict = load_dataset(dataset_name)
    if tuple(dataset_dict.keys()) == ("train",):  # no train-val-test split
        _split = dataset_dict["train"].train_test_split(test_size=0.3)
        _split_test_val = _split["test"].train_test_split(test_size=0.5)

        train_split = _split["train"]
        validation_split = _split_test_val["train"]
        test_split = _split_test_val["test"]

        dataset_dict = DatasetDict(
            {
                "train": train_split,
                "validation": validation_split,
                "test_split": test_split,
            }
        )

    input_dataset_dicts.append(dataset_dict)

split_names = set()
for dataset_dict in input_dataset_dicts:
    for split_name in dataset_dict.keys():
        split_names.add(split_name)

for split_name in split_names:
    split_combined = []
    for dataset_dict in input_dataset_dicts:
        split = dataset_dict.get(split_name)

        if split is not None:
            split_combined.append(split)

    output_dataset_dict[split_name] = concatenate_datasets(split_combined)

output_dataset_dict.save_to_disk(output_path)
print("Saved to disk:", output_path)
print(output_dataset_dict)
