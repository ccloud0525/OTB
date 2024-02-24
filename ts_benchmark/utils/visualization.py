# """
# * @author: 踏雪尋梅
# *
# * @create: 2024-02-22 19:10
# *
# * @description: 可视化ensemble model及其子模型对于单变量时间序列的预测效果
# """
import os.path

import matplotlib.pyplot as plt
import numpy as np
from ..evaluation.metrics.regression_metrics import smape


def Visualize_Ensemble_Model(
    train, test, pred, middle_result, weight_dict, series_name
):
    # 训练集和测试集数据
    train_data = np.array(train)  # 请用实际的训练集数据替换掉[...]
    test_data = np.array(test)  # 请用实际的测试集数据替换掉[...]
    pred_data = np.array(pred)

    pred_len = len(pred_data)
    model_names = list(middle_result.keys())
    # 设置颜色和标签
    colors = ["black", "red", "green", "blue", "orange", "purple"]  # 请根据实际情况调整颜色数量和选择
    labels = [
        "Train Data",
        "Test Data",
        *[
            f"{model_name}, weight: {np.around(weight_dict[model_name].item(), decimals=3)} , smape : {np.around(smape(actual=test_data, predicted=middle_result[model_name]), decimals=2)}%"
            for model_name in model_names
        ],
        f"EnsembleModel, smape : {np.around(smape(actual=test_data, predicted=pred_data), decimals=2)}%",
    ]

    # 绘制图表
    plt.figure(figsize=(15, 8))

    max_length = min(len(train_data), 7 * pred_len)
    x_range = range(max_length + pred_len)

    train_range = x_range[:-pred_len]
    test_range = x_range[-pred_len:]

    # 绘制折线图

    # 设置横轴为日期格式
    # 绘制训练集和测试集数据
    plt.plot(train_range, train_data[-max_length:, 0], color=colors[0], label=labels[0])
    plt.plot(test_range, test_data[:, 0], color=colors[1], label=labels[1])
    plt.plot(test_range, pred_data[:, 0], color=colors[-1], label=labels[-1])

    # 绘制模型预测结果
    for i, (model, result) in enumerate(middle_result.items()):
        plt.plot(test_range, result[:, 0], color=colors[i + 2], label=labels[i + 2])

    data_name, _ = os.path.splitext(series_name)

    # 设置图表标题和轴标签
    plt.title(f"Model Predictions on {data_name}")
    plt.xlabel("date")  # 请根据实际情况替换为正确的横轴标签
    plt.ylabel("value")  # 请根据实际情况替换为正确的纵轴标签

    # 显示图例
    plt.legend()
    current_script_path = os.path.abspath(__file__)
    dir_path = os.path.abspath(
        os.path.join(current_script_path, "..", "..", "..", "result", "pictures_2")
    )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, f"{data_name}.png"), dpi=400)
    plt.close()


# 如果需要展示，需要使用线程安全的代码
# import os.path
# import sys
# import threading
# from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.dates as mdates


# class WorkerThread(threading.Thread):
#     def __init__(self, plot_func, *args, **kwargs):
#         super().__init__()
#         self.plot_func = plot_func
#         self.args = args
#         self.kwargs = kwargs
#
#     def run(self):
#         self.plot_func(*self.args, **self.kwargs)
#
#
# def Visualize_Ensemble_Model(
#     train, test, pred, middle_result, weight_dict, series_name
# ):
#     # 创建Qt应用程序
#     app = QApplication(sys.argv)
#
#     # 在主线程中创建窗口
#     window = MyWindow(train, test, pred, middle_result, weight_dict, series_name)
#
#     # 显示窗口
#     window.show()
#
#     # 启动Qt事件循环
#     sys.exit(app.exec_())
#
#
# class MyWindow(QMainWindow):
#     def __init__(self, train, test, pred, middle_result, weight_dict, series_name):
#         super().__init__()
#
#         self.central_widget = QWidget(self)
#         self.setCentralWidget(self.central_widget)
#
#         layout = QVBoxLayout(self.central_widget)
#
#         self.canvas = FigureCanvas(plt.Figure())
#         layout.addWidget(self.canvas)
#         self.series_name = series_name
#
#         # 创建并启动一个线程来进行绘图操作
#         thread = WorkerThread(
#             self.plot_data, train, test, pred, middle_result, weight_dict
#         )
#         thread.start()
#
#     def plot_data(self, train, test, pred, middle_result, weight_dict):
#         # 训练集和测试集数据
#         train_data = np.array(train)
#         test_data = np.array(test)
#         pred_data = np.array(pred)
#
#         pred_len = len(pred_data)
#         model_names = list(middle_result.keys())
#         # 设置颜色和标签
#         colors = [
#             "black",
#             "red",
#             "green",
#             "blue",
#             "orange",
#             "purple",
#         ]  # 请根据实际情况调整颜色数量和选择
#         labels = [
#             "Train Data",
#             "Test Data",
#             *[
#                 f"{model_name}, weight: {np.around(weight_dict[model_name].item(), decimals=3)}"
#                 for model_name in model_names
#             ],
#             "EnsembleModel",
#         ]
#
#         # 绘制图表
#         ax = self.canvas.figure.add_subplot(111)
#
#         x_range = range(len(train_data) + pred_len)
#         train_range = x_range[:-pred_len]
#         test_range = x_range[-pred_len:]
#
#         # 绘制折线图
#
#         # 设置横轴为日期格式
#         # 绘制训练集和测试集数据
#         ax.plot(train_range, train_data[:, 0], color=colors[0], label=labels[0])
#         ax.plot(test_range, test_data[:, 0], color=colors[1], label=labels[1])
#         ax.plot(test_range, pred_data[:, 0], color=colors[-1], label=labels[-1])
#
#         # 绘制模型预测结果
#         for i, (model, result) in enumerate(middle_result.items()):
#             ax.plot(test_range, result[:, 0], color=colors[i + 2], label=labels[i + 2])
#
#         # 设置图表标题和轴标签
#         ax.set_title("Model Predictions")
#         ax.set_xlabel("date")  # 请根据实际情况替换为正确的横轴标签
#         ax.set_ylabel("value")  # 请根据实际情况替换为正确的纵轴标签
#         ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
#
#         # 显示图例
#         ax.legend()
#
#         current_script_path = os.path.abspath(__file__)
#         dir_path = os.path.abspath(
#             os.path.join(current_script_path, "..", "..", "..", "result", "pictures")
#         )
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#         data_name, _ = os.path.splitext(self.series_name)
#         self.canvas.figure.savefig(os.path.join(dir_path, f"{data_name}.png"))
#         plt.close(self.canvas.figure)  # 关闭图形以释放资源
