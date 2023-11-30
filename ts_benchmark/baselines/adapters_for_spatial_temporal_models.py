import os.path

import torch
import torch.nn as nn

from .time_series_library.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.utils.data_processing import split_before
from typing import Type, Dict
from torch import optim
import numpy as np
import pandas as pd
from ts_benchmark.baselines.utils import data_provider, train_val_split

from ..common.constant import ROOT_PATH

DEFAULT_SPATIAL_TEMPORAL_BASED_HYPER_PARAMS = {
    "batch_size": 64,
    "lr": 0.003,
    "num_epochs": 10,
    "loss": "L1",
    "patience": 3,
    "num_node": 7,
    "input_dim": 1,
    "output_dim": 1,
    "seq_len": 12,
    "pred_len": 12,
    "label_len": 6,
    "hidden_dim": 64,
    "num_layers": 2,
    "default_graph": True,
    "embed_dim": 2,
    "cheb_k": 3,
    "freq": "h",
    "num_workers": 5,
    "lradj": "type1",
}


def spatial_temporal_dim_unifrom(sample: torch.Tensor):
    return sample.unsqueeze(-1)


class StandardScaler:
    def __init__(self):
        pass

    def fit(self, train_data: np.ndarray):
        self.mean = train_data.mean()
        self.std = train_data.std()

    def transform(self, data: np.ndarray):
        return (data - self.mean) / self.std

    def inverse_transformer(self, data: np.ndarray):
        return data * self.std + self.mean


class SpatialTemporalConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_SPATIAL_TEMPORAL_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class SpatialTemporalAdapter:
    def __init__(self, model_name, model_class, **kwargs):
        super(SpatialTemporalAdapter, self).__init__()
        self.config = SpatialTemporalConfig(**kwargs)
        self.model_name = model_name
        self.model_class = model_class
        self.scaler = StandardScaler()

    @staticmethod
    def required_hyper_params() -> dict:
        """
        返回 TimeSeriesLSTM 所需的超参数。

        :return: 一个空字典，表示 TimeSeriesLSTM 不需要额外的超参数。
        """
        return {}

    def __repr__(self) -> str:
        """
        返回模型名称的字符串表示。
        """
        return self.model_name

    def padding_data_for_forecast(self, test):
        time_column_data = test.index
        data_colums = test.columns
        start = time_column_data[-1]
        # padding_zero = [0] * (self.config.pred_len + 1)
        date = pd.date_range(
            start=start, periods=self.config.pred_len + 1, freq=self.config.freq.upper()
        )
        df = pd.DataFrame(columns=data_colums)

        df.iloc[: self.config.pred_len + 1, :] = 0

        df["date"] = date
        # df = pd.DataFrame({"date": date, "col_1": padding_zero})
        df = df.set_index("date")
        new_df = df.iloc[1:]
        test = pd.concat([test, new_df])
        return test

    def validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for input, target, input_mark, target_mark in valid_data_loader:
            input, target = (
                spatial_temporal_dim_unifrom(input).to(device),
                spatial_temporal_dim_unifrom(target).to(device),
            )

            output = self.model(input)

            target = target[:, -config.pred_len :, ...]
            output = output[:, -config.pred_len :, ...]

            loss = criterion(output, target).detach().cpu().numpy()
            total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def forecast_fit(self, train_data: pd.DataFrame, ratio):
        """
        训练模型。

        :param train_data: 用于训练的时间序列数据。
        """
        self.model = self.model_class(self.config)
        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config

        dataset_info = str(train_data.shape) + str(train_data.iloc[0, 0])

        train_data_value, valid_data_value = train_val_split(
            train_data, ratio, config.seq_len
        )
        self.scaler.fit(train_data_value.values)

        train_data = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data_value.values),
            columns=valid_data_value.columns,
            index=valid_data_value.index,
        )

        valid_dataset, valid_data_loader = data_provider(
            valid_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        train_dataset, train_data_loader = data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Define the loss function and optimizer
        if config.loss == "L1":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = EarlyStopping(patience=config.patience)

        self.model.to(device)
        # 计算可学习参数的总数
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total trainable parameters: {total_params}")

        saved_str = f"{dataset_info}; {self.model_name} {str(vars(self.config))} Total trainable parameters: {total_params}\n"
        save_log_path = os.path.join(ROOT_PATH, "result/middle_result.txt")
        with open(save_log_path, "a") as file:
            file.write(saved_str)

        # print(self.model.state_dict())

        for epoch in range(config.num_epochs):
            self.model.train()
            # for input, target, input_mark, target_mark in train_data_loader:
            for i, (input, target, input_mark, target_mark) in enumerate(
                train_data_loader
            ):
                optimizer.zero_grad()
                input, target = (
                    spatial_temporal_dim_unifrom(input).to(device),
                    spatial_temporal_dim_unifrom(target).to(device),
                )

                output = self.model(input)

                target = target[:, -config.pred_len :, ...]
                output = output[:, -config.pred_len :, ...]
                loss = criterion(output, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                optimizer.step()
            valid_loss = self.validate(valid_data_loader, criterion)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break

            adjust_learning_rate(optimizer, epoch + 1, config)

    def forecast(self, pred_len: int, train: pd.DataFrame) -> np.ndarray:
        """
        进行预测。

        :param pred_len: 预测的长度。
        :param testdata: 用于预测的时间序列数据。
        :return: 预测结果的数组。
        """

        self.model.load_state_dict(self.early_stopping.check_point)

        train = pd.DataFrame(
            self.scaler.transform(train.values),
            columns=train.columns,
            index=train.index,
        )

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        train, test = split_before(train, len(train) - config.seq_len)

        # 生成transformer类方法需要的额外时间戳mark
        test = self.padding_data_for_forecast(test)

        test_data_set, test_data_loader = data_provider(
            test, config, timeenc=1, batch_size=1, shuffle=False, drop_last=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            answer = None
            while answer is None or answer.shape[0] < pred_len:
                for input, target, input_mark, target_mark in test_data_loader:
                    input, target = (
                        spatial_temporal_dim_unifrom(input).to(device),
                        spatial_temporal_dim_unifrom(target).to(device),
                    )

                    output = self.model(input).squeeze(-1)

                column_num = output.shape[-1]
                temp = output.cpu().numpy().reshape(-1, column_num)[-config.pred_len :]

                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= pred_len:
                    answer[-pred_len:] = self.scaler.inverse_transform(
                        answer[-pred_len:]
                    )

                    return answer[-pred_len:]

                output = output.cpu().numpy()[:, -config.pred_len :, ...]
                for i in range(config.pred_len):
                    test.iloc[i + config.seq_len] = output[0, i, ...]

                test = test.iloc[config.pred_len :]
                test = self.padding_data_for_forecast(test)

                test_data_set, test_data_loader = data_provider(
                    test,
                    config,
                    timeenc=1,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )


def generate_model_factory(
    model_name: str, model_class: type, required_args: dict
) -> Dict:
    """
    生成模型工厂信息，用于创建 TransformerAdapter 模型适配器。

    :param model_name: 模型名称。
    :param model_class: TsingHua Timeseries library 模型类。
    :param required_args: 模型初始化所需参数。
    :param allow_fit_on_eval: 是否允许在预测阶段拟合模型。
    :return: 包含模型工厂和所需参数的字典。
    """

    def model_factory(**kwargs) -> SpatialTemporalAdapter:
        """
        模型工厂，用于创建 TransformerAdapter模型适配器对象。

        :param kwargs: 模型初始化参数。
        :return:  模型适配器对象。
        """
        return SpatialTemporalAdapter(model_name, model_class, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }


def spatial_temporal_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        required_args={
            "seq_len": "input_chunk_length",
            "pred_len": "output_chunk_length",
        },
    )
