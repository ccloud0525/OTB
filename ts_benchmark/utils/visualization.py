# """
# * @author: 踏雪尋梅
# *
# * @create: 2024-02-22 19:10
# *
# * @description: 可视化ensemble model及其子模型对于单变量时间序列的预测效果
# """
import os.path
import torch
import matplotlib.pyplot as plt
import numpy as np
from ..evaluation.metrics.regression_metrics import msmape


def generate_colors(num_colors):
    colormap = plt.cm.get_cmap("tab20", num_colors)  # 选择一个预定义的颜色映射，例如 'tab20'
    colors = [colormap(i) for i in range(num_colors)]  # 从颜色映射中获取指定数量的颜色
    return colors


def Visualize_Ensemble_Model(
        train, test, pred, middle_result, weight_dict, series_name
):
    model_num = len(weight_dict)
    # 训练集和测试集数据
    train_data = np.array(train)  # 请用实际的训练集数据替换掉[...]
    test_data = np.array(test)  # 请用实际的测试集数据替换掉[...]
    pred_data = np.array(pred)

    pred_len = len(pred_data)
    model_names = list(middle_result.keys())

    def convert_to_float(value):
        if isinstance(value, torch.Tensor):
            return value.item()
        else:
            return float(value)

    # 设置颜色和标签
    colors = ["black", "red", "purple"] + generate_colors(model_num)
    labels = [
        "Train Data",
        "Test Data",
        f"EnsembleModel, msmape : {np.around(msmape(actual=test_data, predicted=pred_data), decimals=2)}%",
        *[
            f"{model_name}, weight: {np.around(convert_to_float(weight_dict[model_name]), decimals=3)} , msmape : {np.around(msmape(actual=test_data, predicted=middle_result[model_name]), decimals=2)}%"
            for model_name in model_names
        ],
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
    plt.plot(test_range, pred_data[:, 0], color=colors[2], label=labels[2])

    # 绘制模型预测结果
    for i, model_name in enumerate(model_names):
        plt.plot(test_range, middle_result[model_name][:, 0], color=colors[i + 3], label=labels[i + 3])


    data_name = series_name

    # 设置图表标题和轴标签
    plt.title(f"Model Predictions on {data_name}")
    plt.xlabel("date")  # 请根据实际情况替换为正确的横轴标签
    plt.ylabel("value")  # 请根据实际情况替换为正确的纵轴标签

    # 显示图例
    plt.legend()
    current_script_path = os.path.abspath(__file__)
    dir_path = os.path.abspath(
        os.path.join(current_script_path, "..", "..", "..", "result", "pictures_test")
    )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, f"{data_name}.png"), dpi=400)
    plt.close()
