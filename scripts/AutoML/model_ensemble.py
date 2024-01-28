import lightgbm as lgb
import pandas as pd
from .ts2vec_model.datautils import process_data
from .ts2vec import TS2Vec
from .ts2vec_model.utils import init_dl_program
import numpy as np
from .ALGORITHMS import ALGORITHMS
import os
from ts_benchmark.models.get_model import get_model


class EnsembleModel:
    def __init__(self, raw_model_factory, dataset, pred_len=48, sample_len=24, top_k=5):
        self.top_k = top_k
        self.dataset = dataset
        self.pred_len = pred_len
        self.sample_len = sample_len
        self.raw_model_factory = raw_model_factory
        self.trained_models = []

        self.__compile()
        self.__parse()

    def __compile(self):
        device = init_dl_program(0, seed=301, max_threads=None)
        data = self.dataset.reset_index(drop=True)
        np_data = data.values
        data, train_slice, valid_slice, test_slice, scaler = process_data(
            np_data, [0.6, 0.2, 0.2]
        )

        config = dict(
            batch_size=8,
            lr=0.001,
            output_dims=320,
            max_train_length=3000,
        )
        sample_data = data[:, -(self.sample_len + self.pred_len) : -self.pred_len, :]
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

    def __parse(self):
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
            ] = self.raw_model_factory.model_hyper_params

        self.model_factory_lst = get_model(model_config)

    def __repr__(self) -> str:
        """
        返回模型名称的字符串表示。
        """
        return f"EnsembleModel_{self.top_k}"

    def forecast_fit(self, train: pd.DataFrame, ratio: float):
        for model_factory in self.model_factory_lst:
            try:
                model = model_factory()
                if hasattr(model, "forecast_fit"):
                    model.forecast_fit(train, ratio)  # 在训练数据上拟合模型
                else:
                    model.fit(train, ratio)  # 在训练数据上拟合模型

                self.trained_models.append(model)
            except:
                continue

    def forecast(self, pred_len: int, train: pd.DataFrame):
        variable_num = train.shape[-1]
        predict = np.zeros((pred_len, variable_num))
        actual_num = 0
        for model in self.trained_models:
            try:
                temp = model.forecast(pred_len, train)
            except:
                continue

            if np.any(np.isnan(temp)):
                continue
            predict += temp  # 预测未来数据
            actual_num += 1

        predict /= actual_num
        return predict
