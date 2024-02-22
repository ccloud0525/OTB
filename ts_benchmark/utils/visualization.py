"""
* @author: 踏雪尋梅
*
* @create: 2024-02-22 19:10
*
* @description: 可视化ensemble model及其子模型对于单变量时间序列的预测效果
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


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
            f"{model_name}, weight: {np.around(weight_dict[model_name].item(), decimals=3)}"
            for model_name in model_names
        ],
        "EnsembleModel",
    ]

    # 绘制图表
    plt.figure(figsize=(15, 8))

    x_range = range(len(train_data) + pred_len)
    train_range = x_range[:-pred_len]
    test_range = x_range[-pred_len:]

    # 绘制折线图

    # 设置横轴为日期格式
    # 绘制训练集和测试集数据
    plt.plot(train_range, train_data[:, 0], color=colors[0], label=labels[0])
    plt.plot(test_range, test_data[:, 0], color=colors[1], label=labels[1])
    plt.plot(test_range, pred_data[:, 0], color=colors[-1], label=labels[-1])

    # 绘制模型预测结果
    for i, (model, result) in enumerate(middle_result.items()):
        plt.plot(test_range, result[:, 0], color=colors[i + 2], label=labels[i + 2])

    # 设置图表标题和轴标签
    plt.title("Model Predictions")
    plt.xlabel("date")  # 请根据实际情况替换为正确的横轴标签
    plt.ylabel("value")  # 请根据实际情况替换为正确的纵轴标签
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()

    current_script_path = os.path.abspath(__file__)
    dir_path = os.path.abspath(
        os.path.join(current_script_path, "..", "..", "..", "result", "pictures")
    )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data_name, _ = os.path.splitext(series_name)
    plt.savefig(os.path.join(dir_path, f"{data_name}.png"))
