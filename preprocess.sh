cd preprocess
python ./_build_gym.py --build --n_proc 40 --do_test --task "ALL" --train_k 16 --valid_k 16
python ./_build_gym.py --build --n_proc 40 --do_test --task "ALL" --train_k -1 --valid_k 100
python ./_build_gym.py --build --n_proc 8 --do_test --task "FV" --train_k -1 --valid_k 100
python ./_build_gym.py --build --n_proc 18 --do_test --task "FV_OG" --train_k -1 --valid_k 100