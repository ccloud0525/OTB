import os.path

import torch
import lightgbm as lgb
import pickle
import numpy as np
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from ts_benchmark.utils.random_utils import fix_random_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Args for zero-cost AutoML")

parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--exp_id", type=int, default=2)
parser.add_argument("--mode", type=str, default="normal")

args = parser.parse_args()
fix_random_seed()

torch.set_num_threads(3)




if __name__ == "__main__":
    with open(f"single_forecast_result/data_{args.exp_id}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
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
    if not os.path.exists("ckpt"):
        os.mkdir("ckpt")

    if args.mode == "normal":
        with open(f"ckpt/gbm_normal_{args.exp_id}.log", "w") as f:
            print(f"start training...", file=f)
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

        gbm = lgb.train(params, train_set, num_boost_round=100, valid_sets=[valid_set])

        pred = gbm.predict(train_features, num_iteration=gbm.best_iteration)
        pred_list = [list(x).index(max(x)) for x in pred]
        with open(f"ckpt/gbm_normal_{args.exp_id}.log", "a") as f:
            print(
                f"pred_accuracy on train set:{accuracy_score(train_y_series, pred_list)}",
                file=f,
            )

        def get_index(lst, k):
            np_lst = np.array(lst)
            indices = np.argsort(-np_lst)
            return indices[0:k]

        def test_k(pred, k):
            pred_list = [get_index(list(x), k) for x in pred]

            cnt = 0
            for i in range(len(valid_y_series)):
                if valid_y_series[i] in pred_list[i]:
                    cnt += 1

            accuracy = cnt / len(valid_y_series)
            return accuracy

        pred = gbm.predict(valid_features, num_iteration=gbm.best_iteration)
        # for k in range(3, 30):
        #     print(f"pred_accuracy on valid set with k={k}:{test_k(pred, k)}")
        with open(f"ckpt/gbm_normal_{args.exp_id}.log", "a") as f:
            print(f"accuracy on valid set:{test_k(pred, args.top_k)}", file=f)

        gbm.save_model(f"ckpt/gbm_{args.exp_id}.txt")

    elif args.mode == "k-fold":
        with open(f"ckpt/gbm_k-fold_{args.exp_id}.log", "w") as f:
            print(f"start k-fold training...", file=f)

        datasets, vectors = zip(*loaded_data)
        X = np.array([np.reshape(datasets[i], -1) for i in range(len(datasets))])
        vectors = list(vectors)
        y = np.array(np.argmax(vectors, axis=1))
        num_folds = 5  # 假设使用5折交叉验证
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        best_model = None
        best_acc = 0
        best_fold = 0
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

            # 训练模型

            gbm = lgb.train(
                params, train_data, num_boost_round=100, valid_sets=[valid_data]
            )

            # 看看训练集拟合效果
            pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
            pred_list = [list(x).index(max(x)) for x in pred]
            with open(f"ckpt/gbm_k-fold_{args.exp_id}.log", "a") as f:
                print(
                    f"pred_accuracy on train set:{accuracy_score(y_train, pred_list)}",
                    file=f,
                )

            # 在验证集上进行预测
            y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

            def get_index(lst, k):
                np_lst = np.array(lst)
                indices = np.argsort(-np_lst)
                return indices[0:k]

            def test_k(pred, actual, k):
                pred_list = [get_index(list(x), k) for x in pred]

                cnt = 0
                for i in range(len(actual)):
                    if actual[i] in pred_list[i]:
                        cnt += 1

                accuracy = cnt / len(actual)
                return accuracy

            # 评估模型性能
            acc = test_k(y_pred, y_valid, args.top_k)
            with open(f"ckpt/gbm_k-fold_{args.exp_id}.log", "a") as f:
                print(f"Fold {fold + 1}, ACC on valid set: {acc}", file=f)

            gbm.save_model(f"ckpt/gbm_{args.exp_id}_{fold + 1}.txt")

            naive_seasonal_test_label = []
            naive_seasonal_test_pred = []

            for id, label in enumerate(y_valid):
                if label == 28:
                    naive_seasonal_test_label.append(label)
                    naive_seasonal_test_pred.append(y_pred[id])

            naive_seasonal_acc = test_k(
                naive_seasonal_test_pred, naive_seasonal_test_label, args.top_k
            )

            nhits_test_label = []
            nhits_test_pred = []

            for id, label in enumerate(y_valid):
                if label == 12:
                    nhits_test_label.append(label)
                    nhits_test_pred.append(y_pred[id])

            nhits_acc = test_k(nhits_test_pred, nhits_test_label, args.top_k)

            with open(f"ckpt/gbm_k-fold_{args.exp_id}.log", "a") as f:
                print(
                    f"Fold {fold + 1},NaiveSeasonal ACC on valid set: {naive_seasonal_acc} of {len(naive_seasonal_test_label)} datasets",
                    file=f,
                )
                print(
                    f"Fold {fold + 1},Nhits ACC on valid set: {nhits_acc} of {len(nhits_test_label)} datasets",
                    file=f,
                )

        with open(f"ckpt/gbm_k-fold_{args.exp_id}.log", "a") as f:
            print(f"Best Fold: {best_fold}, Best ACC on valid set: {best_acc}", file=f)
