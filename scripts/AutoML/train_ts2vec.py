
import numpy as np
import argparse
import os

import pandas as pd
import pickle
from ts_benchmark.utils.random_utils import fix_random_seed
import time
import datetime

from ts2vec import TS2Vec
from ts2vec_model.utils import init_dl_program

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
    parser.add_argument("--sample_num", type=int, default=48, help="samples num")

    parser.add_argument(
        "--max-train-length",
        type=int,
        default=3000,
        help="For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)",
    )
    parser.add_argument(
        "--iters", type=int, default=None, help="The number of iterations"
    )
    parser.add_argument("--epochs", type=int, default=50, help="The number of epochs")
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
    parser.add_argument(
        "--max-pred-len",
        type=int,
        default=24,
        help="The maximum pred-len in univariate forecasting",
    )
    parser.add_argument("--exp_id", type=int, default=2, help="The experiment id")
    args = parser.parse_args()

    fix_random_seed()
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    dataset_algorithm = np.load(
        "single_forecast_result/dataset_algorithm.npy", allow_pickle=True
    )

    # with open("../../single_forecast_result/dataset_algorithm.pkl", "rb") as f:
    #     dataset_algorithm = pickle.load(f)
    dataset_path = "single_forecast_result/chosen_datasets"
    dataset_list = []

    for dataset, algorithm in dataset_algorithm:
        data = pd.read_csv(os.path.join(dataset_path, dataset))
        df = data[["data"]][-(args.sample_num + args.max_pred_len) : -args.max_pred_len]
        dataset_list.append(df.values)

    dataset = np.concatenate(dataset_list, axis=1)

    print("Loading data... ", end="")
    scaler = StandardScaler().fit(dataset)
    data = scaler.transform(dataset)

    if data.ndim == 2:
        data = np.expand_dims(data, 0)
    elif data.ndim == 1:
        data = np.expand_dims(data, 0)
        data = np.expand_dims(data, -1)

    data = np.transpose(data, (2, 1, 0))

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

    model = TS2Vec(input_dims=data.shape[-1], device=device, **config)
    if os.path.exists(f"{run_dir}/model_{args.exp_id}.pkl"):
        model.load(f"{run_dir}/model_{args.exp_id}.pkl")
    else:
        print("Training...")
        loss_log = model.fit(
            data, n_epochs=args.epochs, n_iters=args.iters, verbose=True
        )
        model.save(f"{run_dir}/model_{args.exp_id}.pkl")

        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    print("Encoding...")

    all_repr = model.encode(
        data, causal=True, sliding_length=1, sliding_padding=2, batch_size=256
    )

    data_alg = []

    for id, (dataset, algorithm) in enumerate(list(dataset_algorithm)):
        data = np.squeeze(all_repr[id, :, :])
        data_alg.append([data, algorithm])

    print("Finished.")
    with open(f"single_forecast_result/data_{args.exp_id}.pkl", "wb") as f:
        pickle.dump(data_alg, f)

    with open(f"single_forecast_result/data_{args.exp_id}.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    print(loaded_data)
