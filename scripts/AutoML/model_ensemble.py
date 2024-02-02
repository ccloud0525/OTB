import lightgbm as lgb
import pandas as pd
from .ts2vec import TS2Vec
import numpy as np
from .ALGORITHMS import ALGORITHMS
import os
from ts_benchmark.models.get_model import get_model
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from ts_benchmark.baselines.utils import train_val_split
from torch.utils.data import DataLoader, Dataset
from ts_benchmark.baselines.time_series_library.utils.tools import EarlyStopping
import torch.nn.functional as F
import random


def set_seed(seed):
    """
    Fix all seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class EnsembleDataset(Dataset):
    def __init__(self, X, Y):
        # 初始化数据集，这里我们假设数据是CSV文件，您需要根据您的数据集进行调整
        super(EnsembleDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        # 获取数据样本，这里我们假设每个样本都是一个字典，您需要根据您的数据集进行调整

        return self.X[index], self.Y[index]

    def __len__(self):
        # 返回数据集的大小
        return len(self.X)


class EnsembleModelAdapter:
    def __init__(
        self,
        recommend_model_hyper_params,
        dataset,
        sample_len=24,
        top_k=5,
        ensemble="learn",
        batch_size=8,
        lr=0.001,
        epochs=10,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.EnsembleModel = EnsembleModel(
            recommend_model_hyper_params,
            dataset,
            self.device,
            sample_len,
            top_k,
        )
        self.ensemble = ensemble
        self.criterion = nn.MSELoss()
        self.recommend_model_hyper_params = recommend_model_hyper_params
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

    def _data_process(self, train: pd.DataFrame, ratio: float):
        horizon_len = self.recommend_model_hyper_params["input_chunk_length"]
        pred_len = self.recommend_model_hyper_params["output_chunk_length"]
        length = int(len(train) * (1 - ratio))
        if length < horizon_len + pred_len:
            # if len(train) < 2 * (horizon_len + pred_len):
            #     raise ValueError("数据集过短，无法进行训练")
            # else:
            #     border = len(train) - horizon_len - pred_len
            #     train_data, val_data = split_before(train, border)
            raise ValueError("数据集过短，无法进行训练")

        else:
            train_data, val_data = train_val_split(train, ratio, None)

        val_x = torch.FloatTensor(
            self.EnsembleModel.inner_forecast_back(horizon_len, pred_len, val_data)
        )
        val_y = [
            torch.FloatTensor(
                val_data.iloc[i + horizon_len : i + horizon_len + pred_len].values
            )
            for i in range(0, val_data.shape[0] - pred_len - horizon_len + 1)
        ]
        self.val_dataset = EnsembleDataset(val_x, val_y)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)

        train_x = torch.FloatTensor(
            self.EnsembleModel.inner_forecast_back(horizon_len, pred_len, train_data)
        )
        train_y = [
            torch.FloatTensor(
                train_data.iloc[i + horizon_len : i + horizon_len + pred_len].values
            )
            for i in range(0, train_data.shape[0] - pred_len - horizon_len + 1)
        ]

        self.train_dataset = EnsembleDataset(train_x, train_y)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def learn_ensemble_weight(self, train: pd.DataFrame, ratio: float):
        if self.ensemble == "mean":
            return
        elif self.ensemble != "learn":
            raise ValueError("ensemble type error")

        self.optimizer = torch.optim.Adam([self.EnsembleModel.weight], lr=self.lr)

        try:
            self._data_process(train, ratio)
        except Exception as e:
            import traceback as tb

            tb.print_exc()
            print(str(e))

            return

        self.EnsembleModel.train()
        self.early_stop = EarlyStopping(patience=min(self.epochs // 2, 3))

        for epoch in range(self.epochs):
            valid_loss = self._validate()
            self.early_stop(valid_loss, self.EnsembleModel)

            for x, y in self.train_loader:
                self.optimizer.zero_grad()
                y_pred = self.EnsembleModel(x)
                y = y.to(self.device)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()

            if self.early_stop.early_stop:
                break

        self.EnsembleModel.load_state_dict(self.early_stop.check_point)

    def _validate(self):
        self.EnsembleModel.eval()
        total_loss = []
        with torch.no_grad():
            for x, y in self.val_loader:
                y_pred = self.EnsembleModel(x)
                y = y.to(self.device)
                loss = self.criterion(y_pred, y)
                total_loss.append(loss.item())

            return np.mean(total_loss)

    def forecast_fit(self, train: pd.DataFrame, ratio: float):
        self.EnsembleModel.forecast_fit(train, ratio)

    def forecast(self, pred_len: int, train: pd.DataFrame):
        return self.EnsembleModel.weighted_forecast(pred_len, train)


class EnsembleModel(nn.Module):
    def __init__(
        self, recommend_model_hyper_params, dataset, device, sample_len=24, top_k=5
    ):
        super().__init__()
        self.top_k = top_k
        self.dataset = dataset
        self.sample_len = sample_len
        self.recommend_model_hyper_params = recommend_model_hyper_params
        self.trained_models = []
        self.device = device
        self._compile()
        self._parse()

    def _compile(self):
        device = self.device
        train_data = self.dataset.reset_index(drop=True)
        np_data = train_data.values

        scaler = StandardScaler().fit(np_data)
        train_data = scaler.transform(np_data)
        train_data = np.transpose(train_data, (1, 0))

        if train_data.ndim != 3:
            train_data = np.expand_dims(train_data, axis=-1)

        config = dict(
            batch_size=8,
            lr=0.001,
            output_dims=320,
            max_train_length=3000,
        )

        sample_data = train_data[:, -self.sample_len :, ...]
        model = TS2Vec(input_dims=sample_data.shape[-1], device=device, **config)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model.load(os.path.join(current_dir, "ts2vec_model/training/0118/model.pkl"))

        repr = model.encode(
            sample_data,
            causal=True,
            sliding_length=1,
            sliding_padding=2,
            batch_size=256,
        )

        gbm = lgb.Booster(model_file=os.path.join(current_dir, f"ckpt/best_gbm_1.txt"))
        test_feature = np.reshape(repr, -1)
        pred = np.reshape(gbm.predict([test_feature]), -1)

        def get_index(lst, k):
            np_lst = np.array(lst)
            indices = np.argsort(-np_lst)
            return indices[0:k]

        indices = get_index(pred, self.top_k)
        self.model_name_lst = [ALGORITHMS[index] for index in indices]

    def _parse(self):
        model_config = {"models": []}
        adapter_lst = []
        new_model_name_lst = []

        for model_name in self.model_name_lst:
            if "darts" in model_name:
                adapter = None
                model_name = "ts_benchmark.baselines.darts_models_single." + model_name
            else:
                adapter = "transformer_adapter_single"
                model_name = (
                    "ts_benchmark.baselines.time_series_library."
                    + model_name
                    + "."
                    + model_name
                )
            adapter_lst.append(adapter)

            new_model_name_lst.append(model_name)

        for adapter, model_name, model_hyper_params in zip(
            adapter_lst, new_model_name_lst, new_model_name_lst
        ):
            model_config["models"].append(
                {
                    "adapter": adapter if adapter is not None else None,
                    "model_name": model_name,
                    "model_hyper_params": {},
                }
            )
            model_config[
                "recommend_model_hyper_params"
            ] = self.recommend_model_hyper_params

        self.model_factory_lst = get_model(model_config)

    def __repr__(self) -> str:
        """
        返回模型名称的字符串表示。
        """
        return f"EnsembleModel_{self.top_k}"

    def forecast_fit(self, train: pd.DataFrame, ratio: float):
        # model_factory_lst = [self.model_factory_lst[-2], self.model_factory_lst[-1]]
        # rf = self.model_factory_lst[-2]
        # nhits = self.model_factory_lst[-1]
        # self.model_factory_lst[-1] = nhits
        # self.model_factory_lst[-2] = rf
        for model_factory in self.model_factory_lst:
            set_seed(2021)
            try:
                model = model_factory()
                if hasattr(model, "forecast_fit"):
                    model.forecast_fit(train, ratio)  # 在训练数据上拟合模型
                else:
                    model.fit(train, ratio)  # 在训练数据上拟合模型

                self.trained_models.append(model)
            except:
                continue

        self.weight = nn.Parameter(
            torch.rand(
                1,
                len(self.trained_models),
            ),
            requires_grad=False,
        ).to(self.device)
        self.weight.requires_grad = True
        nn.init.constant_(self.weight, 1 / len(self.trained_models))

    def forward(self, X: torch.Tensor):
        X = X.to(self.device)
        weighted_predict = torch.einsum(
            "ij,njkl->nikl", F.softmax(self.weight), X
        ).squeeze(1)
        return weighted_predict

    def forecast(self, pred_len: int, train: pd.DataFrame):
        predict_list = []
        for model in self.trained_models:
            try:
                temp = model.forecast(pred_len, train)

            except Exception as e:
                import traceback as tb

                tb.print_exc()
                print(f"{repr(model)}:{str(e)}")
                continue

            if np.any(np.isnan(temp)):
                continue
            predict_list.append(temp)  # 预测未来数据
            # print(f"{repr(model)}")
            # print(temp)

        return np.array(predict_list)

    def weighted_forecast(self, pred_len: int, train: pd.DataFrame):
        predict_list = []
        for model in self.trained_models:
            try:
                temp = model.forecast(pred_len, train)

            except Exception as e:
                import traceback as tb

                tb.print_exc()
                print(f"{repr(model)}:{str(e)}")
                continue

            if np.any(np.isnan(temp)):
                continue
            predict_list.append(temp)  # 预测未来数据
            print(f"{repr(model)}")
            print(temp)
        predict = torch.FloatTensor(np.array(predict_list)).to(self.device)
        weight = F.softmax(self.weight)
        weighted_predict = torch.einsum("ij,jkl->ikl", weight, predict).squeeze(0)

        predict = weighted_predict.detach().cpu().numpy()
        return predict

    def inner_forecast_back(self, horizon_len: int, pred_len: int, data: pd.DataFrame):
        predict_list = []
        for model in self.trained_models:
            try:
                temp = model.inner_forecast_back(horizon_len, pred_len, data)

            except Exception as e:
                import traceback as tb

                tb.print_exc()
                print(f"{repr(model)}:{str(e)}")
                continue

            if np.any(np.isnan(temp)):
                continue
            predict_list.append(temp)  # 预测未来数据
            # print(f"{repr(model)}")
            # print(temp)

        predict = np.array(predict_list)

        predict = np.transpose(predict, (1, 0, 2, 3))
        return predict
