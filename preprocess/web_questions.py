# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class WebQuestions(FewshotGymTextToTextDataset):
    
    def __init__(self):
        self.hf_identifier = "web_questions"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        lines = self.map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)
        
        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        np.random.seed(42)
        for datapoint in hf_dataset[split_name]:
            lines.append((datapoint["question"], "\t".join(datapoint["answers"])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("web_questions", trust_remote_code=True)

def main():
    dataset = WebQuestions()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed)

if __name__ == "__main__":
    main()