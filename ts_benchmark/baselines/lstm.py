import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ts_benchmark.baselines.utils import SlidingWindowDataLoader
from ts_benchmark.utils.data_processing import split_before


class LSTMModel(nn.Module):
    """
    LSTM 模型类。

    该模型用于序列数据的预测和分析，利用 LSTM 神经网络进行建模和训练。
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, output_size: int
    ):
        """
        初始化 LSTM 模型。

        :param input_size: 输入数据的特征维度。
        :param hidden_size: LSTM 隐层的大小。
        :param num_layers: LSTM 层数。
        :param output_size: 输出数据的维度。
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        :param x: 输入的数据张量，维度为 (batch_size, sequence_length, input_size)。
        :return: 模型的输出张量，维度为 (batch_size, output_size)。
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class TimeSeriesLSTM:
    """
    TimeSeriesLSTM 类。

    该类封装了一个使用 LSTM 模型进行时间序列预测的过程。
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 12,
        seq_length: int = 12,
        lr: float = 0.001,
        num_epochs: int = 5,
    ):
        """
        初始化 TimeSeriesLSTM。

        :param input_size: 输入数据的特征维度。
        :param hidden_size: LSTM 隐层的大小。
        :param num_layers: LSTM 层数。
        :param output_size: 输出数据的维度。
        :param seq_length: 序列长度。
        :param lr: 学习率。
        :param num_epochs: 训练的轮次。
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = 1
        self.model_name = "lstm"
        self.num_layers = num_layers
        self.output_size = output_size
        self.seq_length = seq_length
        self.lr = lr
        self.num_epochs = num_epochs
        self.model = LSTMModel(
            self.input_size, self.hidden_size, self.num_layers, self.output_size
        )
        self.prediction_length = output_size

    @staticmethod
    def required_hyper_params() -> dict:
        """
        返回 TimeSeriesLSTM 所需的超参数。

        :return: 一个空字典，表示 TimeSeriesLSTM 不需要额外的超参数。
        """
        return {}

    def forecast_fit(self, train_data: pd.DataFrame):
        """
        训练模型。

        :param train_data: 用于训练的时间序列数据。
        """
        # Create the data loader (dataloader)
        train_data_loader = SlidingWindowDataLoader(
            train_data,
            batch_size=64,
            history_length=self.seq_length,
            prediction_length=self.prediction_length,
            shuffle=False,
        )

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(self.num_epochs):
            self.model.train()
            for batch_data, target in train_data_loader:
                batch_data, target = batch_data.to(device), target.to(device)
                output = self.model(batch_data)
                target = target[:, :, -1]
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def forecast(self, pred_len: int, testdata: pd.DataFrame) -> np.ndarray:
        """
        进行预测。

        :param pred_len: 预测的长度。
        :param testdata: 用于预测的时间序列数据。
        :return: 预测结果的数组。
        """
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        train, test = split_before(testdata, len(testdata) - self.seq_length)
        test_data_loader = SlidingWindowDataLoader(
            test,
            batch_size=1,
            history_length=self.seq_length,
            prediction_length=0,
            shuffle=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for batch_data, target in test_data_loader:
                batch_data, target = batch_data.to(device), target.to(device)
                output = self.model(batch_data).numpy().flatten()
        return output


    def __repr__(self) -> str:
        """
        返回模型名称的字符串表示。
        """
        return self.model_name
