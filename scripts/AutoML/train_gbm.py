import torch
import lightgbm as lgb
import pickle
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from ts_benchmark.utils.random_utils import fix_random_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Args for zero-cost AutoML")

parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--exp_id", type=int, default=1)

args = parser.parse_args()
fix_random_seed()

torch.set_num_threads(3)

if __name__ == "__main__":
    with open("data.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    length = len(loaded_data)
    train_data = loaded_data[: int(0.8 * length)]
    valid_data = loaded_data[int(0.8 * length) :]
    train_datasets, train_vectors = zip(*train_data)
    train_features = np.array(
        [np.reshape(train_datasets[i], -1) for i in range(len(train_datasets))]
    )
    train_vectors = list(train_vectors)
    train_y_series = np.array(np.argmax(train_vectors, axis=1))
    train_set = lgb.Dataset(train_features, label=train_y_series)

    valid_datasets, valid_vectors = zip(*valid_data)
    valid_features = np.array(
        [np.reshape(valid_datasets[i], -1) for i in range(len(valid_datasets))]
    )
    valid_vectors = list(valid_vectors)
    valid_y_series = np.array(np.argmax(valid_vectors, axis=1))

    valid_set = lgb.Dataset(valid_features, label=valid_y_series)

    params = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": 31,
        "metric": "multi_error",
        "min_data_in_leaf": 100,
        "learning_rate": 0.06,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.4,
        "lambda_l2": 0.5,
        "min_gain_to_split": 0.2,
    }

    gbm = lgb.train(params, train_set, num_boost_round=100, valid_sets=[valid_set])

    pred = gbm.predict(valid_features, num_iteration=gbm.best_iteration)

    pred_list = [list(x).index(max(x)) for x in pred]

    print(f"pred_accuracy on valid set:{accuracy_score(valid_y_series, pred_list)}")

    pred = gbm.predict(train_features, num_iteration=gbm.best_iteration)

    pred_list = [list(x).index(max(x)) for x in pred]

    print(f"pred_accuracy on train set:{accuracy_score(train_y_series, pred_list)}")
