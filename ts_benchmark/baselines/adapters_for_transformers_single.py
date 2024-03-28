import os

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .time_series_library.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.utils.data_processing import split_before
from typing import Type, Dict
from torch import optim
import numpy as np
import pandas as pd
from ts_benchmark.baselines.utils import data_provider, train_val_split
from ..common.constant import ROOT_PATH

DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "top_k": 5,
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 64,
    "d_ff": 64,
    "embed": "timeF",
    "freq": "h",
    "lradj": "type1",
    "moving_avg": 25,
    "num_kernels": 6,
    "factor": 3,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 4,
    "stride": 2,
    "dropout": 0.1,
    "batch_size": 8,
    "lr": 0.001,
    "num_epochs": 35,
    "num_workers": 0,
    "loss": "MSE",
    "itr": 1,
    "distil": True,
    "patience": 5,
    "task_name": "short_term_forecast",
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "mem_dim": 32,
    "conv_kernel": [12, 16],
}


class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class TransformerAdapter_single:
    def __init__(self, model_name, model_class, **kwargs):
        self.config = TransformerConfig(**kwargs)
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

    def hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("不规则的时间间隔")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)

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
            input, target, input_mark, target_mark = (
                input.to(device),
                target.to(device),
                input_mark.to(device),
                target_mark.to(device),
            )
            # decoder input
            dec_input = torch.zeros_like(target[:, -config.pred_len :, :]).float()
            dec_input = (
                torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                .float()
                .to(device)
            )

            output = self.model(input, input_mark, dec_input, target_mark)

            target = target[:, -config.pred_len :, :]
            output = output[:, -config.pred_len :, :]
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
        self.hyper_param_tune(train_data)

        if len(train_data) < 1000:
            self.config.patience = 5
        elif len(train_data) >= 1000 and len(train_data) < 3000:
            self.config.patience = 4
        else:
            self.config.patience = 3

        self.model = self.model_class(self.config)
        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        #
        dataset_info = str(train_data.shape) + str(train_data.iloc[0, 0])
        #
        # self.scaler.fit(train_data.values)
        #
        # train_data_value = pd.DataFrame(self.scaler.transform(train_data.values), columns=train_data.columns,
        #                                       index=train_data.index)
        train_data_value = train_data

        train_dataset, train_data_loader = data_provider(
            train_data_value,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Define the loss function and optimizer
        # criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        criterion = nn.SmoothL1Loss()

        optimizer = optim.Adam(self.model.parameters(), lr=config.lr, foreach=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = EarlyStopping(patience=config.patience)

        self.model.to(device)
        # 计算可学习参数的总数
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total trainable parameters: {total_params}")
        # saved_str = f"{dataset_info}; {self.model_name} {str(vars(self.config))} Total trainable parameters: {total_params}\n"
        # save_log_path = os.path.join(ROOT_PATH, "result/univariate_forecast.txt")
        # with open(save_log_path, 'a') as file:
        #     file.write(saved_str)
        # print(self.model.state_dict())

        for epoch in range(config.num_epochs):
            self.model.train()
            # for input, target, input_mark, target_mark in train_data_loader:
            for i, (input, target, input_mark, target_mark) in enumerate(
                train_data_loader
            ):
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                # decoder input
                dec_input = torch.zeros_like(target[:, -config.pred_len :, :]).float()
                dec_input = (
                    torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                    .float()
                    .to(device)
                )

                output = self.model(input, input_mark, dec_input, target_mark)

                target = target[:, -config.pred_len :, :]
                output = output[:, -config.pred_len :, :]
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

            valid_loss = self.validate(train_data_loader, criterion)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break

    def forecast(self, pred_len: int, train: pd.DataFrame) -> np.ndarray:
        """
        进行预测。

        :param pred_len: 预测的长度。
        :param testdata: 用于预测的时间序列数据。
        :return: 预测结果的数组。
        """

        self.model.load_state_dict(self.early_stopping.check_point)

        # train = pd.DataFrame(self.scaler.transform(train.values), columns=train.columns,
        #                                       index=train.index)
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
                    input, target, input_mark, target_mark = (
                        input.to(device),
                        target.to(device),
                        input_mark.to(device),
                        target_mark.to(device),
                    )
                    dec_input = torch.zeros_like(
                        target[:, -config.pred_len :, :]
                    ).float()
                    dec_input = (
                        torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                        .float()
                        .to(device)
                    )
                    output = self.model(input, input_mark, dec_input, target_mark)

                column_num = output.shape[-1]
                temp = output.cpu().numpy().reshape(-1, column_num)[-config.pred_len :]

                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= pred_len:
                    # answer[-pred_len:] = self.scaler.inverse_transform(answer[-pred_len:])
                    return answer[-pred_len:]

                output = output.cpu().numpy()[:, -config.pred_len :, :]
                for i in range(config.pred_len):
                    test.iloc[i + config.seq_len] = output[0, i, :]

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

    def inner_forecast_back(
        self, horizon_len: int, pred_len: int, data: pd.DataFrame
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_set, data_loader = data_provider(
            data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        output = []
        for input, target, input_mark, target_mark in data_loader:
            input, target, input_mark, target_mark = (
                input.to(device),
                target.to(device),
                input_mark.to(device),
                target_mark.to(device),
            )
            dec_input = torch.zeros_like(target[:, -config.pred_len :, :]).float()
            dec_input = (
                torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                .float()
                .to(device)
            )
            output.append(self.model(input, input_mark, dec_input, target_mark))

        output = torch.concat(output, dim=0)
        output = output.detach().cpu().numpy()
        return output


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

    def model_factory(**kwargs) -> TransformerAdapter_single:
        """
        模型工厂，用于创建 TransformerAdapter模型适配器对象。

        :param kwargs: 模型初始化参数。
        :return:  模型适配器对象。
        """
        return TransformerAdapter_single(model_name, model_class, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }


def transformer_adapter_single(model_info: Type[object]) -> object:
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
