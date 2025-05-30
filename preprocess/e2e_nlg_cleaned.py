# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class E2E_NLG(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "e2e_nlg_cleaned"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            input_text = datapoint["meaning_representation"]
            lines.append((input_text, datapoint["human_reference"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('e2e_nlg_cleaned', trust_remote_code=True)

def main():
    dataset = E2E_NLG()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed)

if __name__ == "__main__":
    main()