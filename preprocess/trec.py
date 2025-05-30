# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class TREC(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "trec"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"DESC",
            1:"ENTY",
            2:"ABBR",
            3:"HUM",
            4:"NUM",
            5:"LOC",
        }

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
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["text"], self.label[datapoint["label-coarse"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('trec', trust_remote_code=True)

def main():
    dataset = TREC()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed)

if __name__ == "__main__":
    main()