import random

import numpy as np
import argparse
import os
import pickle


import pandas as pd
import torch

from ts_benchmark.utils.random_utils import fix_random_seed
import time
import datetime

from ts2vec import TS2Vec
import ts2vec_model.datautils as datautils
from ts2vec_model.utils import init_dl_program, name_with_datetime

from sklearn.preprocessing import StandardScaler


def save_checkpoint_callback(save_every=1, unit="epoch"):
    assert unit in ("epoch", "iter")

    def callback(model, loss):
        n = model.n_epochs if unit == "epoch" else model.n_iters
        if n % save_every == 0:
            model.save(f"{run_dir}/model_{n}.pkl")

    return callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_name",
        help="The folder name used to save model, output and evaluation metrics. This can be set to any word",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="The gpu no. used for training and inference (defaults to 0)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="The batch size (defaults to 8)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="The learning rate (defaults to 0.001)"
    )
    parser.add_argument(
        "--repr_dims",
        type=int,
        default=320,
        help="The representation dimension (defaults to 320)",
    )
    parser.add_argument("--sample_num", type=int, default=80, help="samples num")

    parser.add_argument(
        "--max-train-length",
        type=int,
        default=3000,
        help="For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)",
    )
    parser.add_argument(
        "--iters", type=int, default=None, help="The number of iterations"
    )
    parser.add_argument("--epochs", type=int, default=30, help="The number of epochs")
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save the checkpoint every <save_every> iterations/epochs",
    )
    parser.add_argument("--seed", type=int, default=301, help="The random seed")
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="The maximum allowed number of threads used by this process",
    )
    parser.add_argument("--ratio", nargs="+", type=float, default=[0.6, 0.2, 0.2])

    args = parser.parse_args()

    fix_random_seed()
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    dataset_algorithm = np.load(
        "../../single_forecast_result/dataset_algorithm.npy", allow_pickle=True
    )
    dataset_path = "../../single_forecast_result/chosen_datasets"
    dataset_list = []

    for dataset, algorithm in dataset_algorithm:
        data = pd.read_csv(os.path.join(dataset_path, dataset))
        df = data[["data"]][-args.sample_num :]
        dataset_list.append(df.values)

    dataset = np.concatenate(dataset_list, axis=1)

    print("Loading data... ", end="")

    data, train_slice, valid_slice, test_slice, scaler = datautils.process_data(
        dataset, args.ratio
    )
    train_data = data[:, train_slice]

    print("done")

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
    )

    if args.save_every is not None:
        unit = "epoch" if args.epochs is not None else "iter"
        config[f"after_{unit}_callback"] = save_checkpoint_callback(
            args.save_every, unit
        )

    run_dir = "./ts2vec_model/training/" + args.run_name
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = TS2Vec(input_dims=train_data.shape[-1], device=device, **config)
    if os.path.exists(f"{run_dir}/model.pkl"):
        model.load(f"{run_dir}/model.pkl")
    else:
        print("Training...")
        loss_log = model.fit(
            train_data, n_epochs=args.epochs, n_iters=args.iters, verbose=True
        )
        model.save(f"{run_dir}/model.pkl")

        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    print("Encoding...")

    all_repr = model.encode(
        data, causal=True, sliding_length=1, sliding_padding=2, batch_size=256
    )

    data_alg = []

    for id, (dataset, algorithm) in enumerate(list(dataset_algorithm)):
        data = np.squeeze(all_repr[id, :, :])
        data_alg.append([data,algorithm])

    print("Finished.")
    with open("data.pkl", "wb") as f:
        pickle.dump(data_alg, f)

    with open("data.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    print(loaded_data)
