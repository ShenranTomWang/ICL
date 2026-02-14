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
To generate `random` or `#%_correct` data by dataset, run the following:
```shell
python create_data_custom.py --variant {random|0_correct|25_correct|50_correct|75_correct} --dataset {dataset}
```
Alternatively, if you want to generate a variant of all datasets in a `config.json` file under [config](./config/), run:
```shell
python create_data_custom.py --variant {random|0_correct|25_correct|50_correct|75_correct} --task {config_name}
```
This will create the corresponding datasets, and a new `config.json` file under [config](./config/).

## Running Experiments
1. Behaviour experiments (Section 4): this [script](test_fv_og.sh) runs on the parametric knowledge retrieval datasets.
2. Extracting FVs for all heads (Section 5.1): this [script](extract_fv_steer.sh) runs on the parametric knowledge retrieval datasets.
3. Identifying head importance via AIE (Section 5.1): this [script](function_vectors_original.sh) runs on the parametric knowledge retrieval datasets.
4. Steering top FV heads (Section 5.3): this [script](test_fv_og_steer.sh) runs on the parametric knowledge retrieval datasets.
5. Ablating top FV heads (Section 5.3): this [script](test_fv_og_removal_ablation.sh) runs on the parametric knowledge retrieval datasets.
