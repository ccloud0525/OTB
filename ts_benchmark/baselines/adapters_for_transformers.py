import torch
import torch.nn as nn
from .time_series_library.utils.tools import EarlyStopping
from ts_benchmark.utils.data_processing import split_before
from typing import Type, Dict
from torch import optim

import numpy as np
import pandas as pd
from ts_benchmark.baselines.utils import DataloaderForTransformer

DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "top_k": 1,
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 2,
    "d_model": 32,
    "d_ff": 32,
    "embed": "timeF",
    "freq": "h",
    "moving_avg": 25,
    "num_kernels": 6,
    "factor": 3,
    "n_heads": 8,
    "activation": "gelu",
    "output_attention": 0,
    "dropout": 0.1,
    "lr": 0.001,
    "num_epochs": 100,
    "task_name": "short_term_forecast",
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
}


class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class TransformerAdapter:
    def __init__(self, model_name, model_class, **kwargs):
        super(TransformerAdapter, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.model_name = model_name
        self.model_class = model_class
        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)

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
        start = time_column_data[-1]
        padding_zero = [0] * (self.config.pred_len + 1)
        date = pd.date_range(
            start=start, periods=self.config.pred_len + 1, freq=self.config.freq.upper()
        )
        df = pd.DataFrame({"date": date, "col_1": padding_zero})
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

    def forecast_fit(self, train_data: pd.DataFrame):
        """
        训练模型。

        :param train_data: 用于训练的时间序列数据。
        """
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("不规则的时间间隔")
        else:
            self.config.freq = freq[0].lower()

        self.model = self.model_class(self.config)

        config = self.config
        border = int((train_data.shape[0]) * 0.8)

        train_data, valid_data = split_before(train_data, border)
        valid_data_loader = DataloaderForTransformer(
            dataset=valid_data,
            batch_size=64,
            history_len=config.seq_len,
            prediction_len=config.pred_len,
            label_len=config.label_len,
            shuffle=False,
            timeenc=0,
            freq=self.config.freq,
        )
        # Create the data loader (dataloader)
        train_data_loader = DataloaderForTransformer(
            dataset=train_data,
            batch_size=64,
            history_len=config.seq_len,
            prediction_len=config.pred_len,
            label_len=config.label_len,
            shuffle=False,
            timeenc=0,
            freq=self.config.freq,
        )

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        early_stopping = EarlyStopping()
        self.model.to(device)

        for epoch in range(config.num_epochs):
            self.model.train()
            for input, target, input_mark, target_mark in train_data_loader:
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
            valid_loss = self.validate(valid_data_loader, criterion)
            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                self.model.load_state_dict(early_stopping.check_point)
                break

    def forecast(self, pred_len: int, train: pd.DataFrame) -> np.ndarray:
        """
        进行预测。

        :param pred_len: 预测的长度。
        :param testdata: 用于预测的时间序列数据。
        :return: 预测结果的数组。
        """
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        train, test = split_before(train, len(train) - config.seq_len)

        # 生成transformer类方法需要的额外时间戳mark
        test = self.padding_data_for_forecast(test)

        test_data_loader = DataloaderForTransformer(
            dataset=test,
            batch_size=1,
            history_len=config.seq_len,
            prediction_len=config.pred_len,
            label_len=config.label_len,
            shuffle=False,
            timeenc=0,
            freq=self.config.freq,
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
                    break

                temp = output.cpu().numpy().flatten()[-config.pred_len :]
                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= pred_len:
                    return answer[-pred_len:]

                output = output.cpu().numpy()[:, -config.pred_len :, :]
                for i in range(config.pred_len):
                    test["col_1"][i + config.seq_len] = output[0, i, :]
                test = test.iloc[config.pred_len :]
                test = self.padding_data_for_forecast(test)
                test_data_loader = DataloaderForTransformer(
                    dataset=test,
                    batch_size=1,
                    history_len=config.seq_len,
                    prediction_len=config.pred_len,
                    label_len=config.label_len,
                    shuffle=False,
                    timeenc=0,
                    freq=config.freq,
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

    def model_factory(**kwargs) -> TransformerAdapter:
        """
        模型工厂，用于创建 TransformerAdapter模型适配器对象。

        :param kwargs: 模型初始化参数。
        :return:  模型适配器对象。
        """
        return TransformerAdapter(model_name, model_class, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }


def transformer_adapter(model_info: Type[object]) -> object:
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
