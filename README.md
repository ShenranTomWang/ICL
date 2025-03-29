# ICL
Here are the instructions for getting started with this repository

## Data Preperation
Data are scraped from `datasets`. To download data, first make sure that the version of `datasets` is 1.4.0 (which, if you installed dependencies in `requirements.txt`, is not). then run
```shell
cd preprocess
python _build_gym.py --build --n_proc=<number of processes to run> --do_test --test_k {4|8|32}
```
After scraping the data, you can convert the version of `datasets` back to your version.

## Generating Ablations
To generate `random` or `#%_correct` data, run the following:
```shell
python create_data.py --variant {random|0_correct|25_correct|50_correct|75_correct} --dataset {dataset}
```
For a `task` defined in [config](./config/), you must create datasets required by this task one by one. After creating data you will see datasets like `dataset_{variant}` in your data folder, and you need to manually add `{task}_{variant}` config in [config](./config/). I have added `analysis_classification_random` so you don't need to worry about this one.

## Running Experiments
To run experiments, follow the script outlined in `test_custom_analysis_classification.sh`. This runs the experiment on `analysis_classification` task which contains 5 datasets of different purposes, all classification. to run other tasks, you can define your task in [config](./config/). The script uses `k = 4` samples of demonstrations, but you can change them accordingly