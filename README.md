# Open TimeSeries Benchmark（OTB）

**OTB是一个面向时间序列研究人员的开源库。**

**我们提供了一个整洁的代码库来端到端的评估时间序列模型在不同评估策略以及评价指标下与baseline算法性能的对比。**

## Quickstart

### Installation

Given a python environment (**note**: this project ßis fully tested under python 3.8), install the dependencies with the following command:

```
pip install -r requirements.txt
```

### Data preparation

Prepare Data. You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/1N9j8bZGvt_2RMarXCSe-ceKmCTkvRSkl?usp=drive_link).Then place the downloaded data under the folder `./dataset`. 

### Example Usage

#### Forecasting Example:

- **Define the model class or factory**
  - We demonstrated what functions need to be implemented for time series forecasting based on **fixed_forecast strategy** using the LSTM algorithm. You can find the complete code in ` ./ts_benchmark/baseline/lstm.py`.
  - The algorithm expects input data in the form of a `pd.DataFrame`, where the time column serves as the index.

```
class TimeSeriesLSTM:

    def forecast_fit(self, train_data: pd.DataFrame):
        """
        训练模型。

        :param train_data: 用于训练的时间序列数据。
        :type train_data: pd.DataFrame
        """
		pass

    def forecast(self, pred_len: int, testdata: pd.DataFrame) -> np.ndarray:
        """
        进行预测。

        :param pred_len: 预测的长度
        :type pred_len: int
        :param testdata: 用于预测的时间序列数据
        :type testdata: pd.DataFrame
        
        :return: 预测结果的数组
        :rtype: np.ndarray
        """
        
        return output

    def __repr__(self) -> str:
        """
        返回模型名称的字符串表示。
        """
        
        return "LSTM"

```

- **Run benchmark with fixed_forecaststrategy**

  ```
  python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-set-name "small_forecast" --model-name "lstm.TimeSeriesLSTM"
  ```

#### Anomaly Detection Example：

- **Define the model class or factory**
  - We demonstrated what functions need to be implemented for anomaly detection based on **fixed_detect_score strategy** using the LOF algorithm. You can find the complete code in ` ./ts_benchmark/baseline/lof.py`.
  - The algorithm expects input data in the form of a `pd.DataFrame`, where the time column serves as the index.


```
class LOF:

    def detect_fit(self, train_data: pd.DataFrame, train_label=None):
        """
        训练 LOF 模型。

        :param train_data: 训练数据
        :type train_data: pd.DataFrame
        :param train_label: 标签数据（可选）
        :type train_label: pd.DataFrame
        """
        
        pass

    def detect_score(self, data: pd.DataFrame) -> np.ndarray:
        """
        使用 LOF 模型计算异常得分

        :param data: 待计算得分的数据
        :type data: pd.DataFrame

        :return: 异常得分数组
        :rtype: np.ndarray
        """
        
        return score

    def __repr__(self) -> str:
        """
        返回模型名称的字符串表示。
        """
        
        return "LOF"

```

- **Run benchmark with fixed_detect_score strategy:**

  ```
  python ./scripts/run_benchmark.py --config-path "fixed_detect_score_config.json" --data-set-name "small_detect" --model-name "lof.LOF"
  ```



## User guide

### Data Format

The algorithm expects input data in the form of a `pd.DataFrame`, where the time column serves as the index. If the time values are integers of type `int`, they will be retained in this format. However, if the time values are in the standard timestamp format, they will be converted to a `pd.DatetimeIndex` type.

#### Example

- **The time values are integers of type `int`，they will be retained in this format.**

```
           col_1  col_2  col_3  col_4  ...  col_53  
date                                   ...                               
1       2.146646    0.0    0.0    0.0  ...     0.0    
2       2.146646    0.0    0.0    0.0  ...     0.0    
3       2.146646    0.0    0.0    0.0  ...     0.0     
4       2.151326    0.0    0.0    0.0  ...     0.0     
5       2.163807    0.0    0.0    0.0  ...     0.0    
...          ...    ...    ...    ...  ...     ...     
132042  0.499149    0.0    0.0    0.0  ...     0.0  
132043  0.501221    0.0    0.0    0.0  ...     0.0     
132044  0.501221    0.0    0.0    0.0  ...     0.0     
132045  0.501221    0.0    0.0    0.0  ...     0.0    
132046 -0.954212    0.0    0.0    0.0  ...     0.0     
```

- **The time values are in the standard timestamp format, they will be converted to a `pd.DatetimeIndex` type.**

```
                           col_1
date                            
2012-09-28 12:00:00  2074.503844
2012-09-29 12:00:00  3024.346943
2012-09-30 12:00:00  3088.428014
2012-10-01 12:00:00  3103.715163
2012-10-02 12:00:00  3123.547161
...                          ...
2016-05-03 12:00:00  9033.287169
2016-05-04 12:00:00  9055.950486
2016-05-05 12:00:00  9202.848984
2016-05-06 12:00:00  9180.724092
2016-05-07 12:00:00  9132.311537
```

### Folder Description

```
- baselines：存储baseline模型。包括第三方库模型，以及本仓库复现的模型

- common：存储一些常量，例如配置文件路径：CONFIG_PATH

- config：存储不同评估策略下的配置文件

- data_loader：存储数据抓取以及数据加载的文件

- evaluation：存储评估策略类、评价指标的实现、运行评估模型的文件

- models：存储根据用户输入的模型路径，返回模型工厂

- report：存储呈现要评估算法与baseline算法性能对比的文件

- utils：存储一些工具文件

- pipeline：存储整个benchmark pipeline连通的文件
```



### Steps to Evaluate Your Model

- **Define you model class or factory**
  - 对于不同的策略而言，用户实现的模型为了适配本benchmark，模型当中应该实现如下函数.
  - 对于所有策略而言，required_hyper_params函数 是可选的，__repr__ 函数是必须的.
  - 其他函数与策略的匹配关系如下表：
  
  |    strategy_name     | Strategic implications                                       | forecast_fit | detect_fit | forecast | detect_label | detect_score |
  | :------------------: | :----------------------------------------------------------- | :----------: | :--------: | :------: | :----------: | :----------: |
  |    fixed_forecast    | Fixed_forecast, with a total of n time points. If the defined prediction step size is f time points, then (n-f) time points are used as training data to predict future f time points. |      √       |            |    √     |              |              |
  |   rolling_forecast   | Rolling_forecast mirrors the cross-validation approach commonly utilized in machine learning. Here, the term 'origin' pertains to the training set within the time series, which is gradually expanded. In simpler terms, this technique enables the generation of multiple forecasts, each produced using an increasingly larger training set extracted from a single time series. |      √       |            |    √     |              |              |
  |  fixed_detect_label  | Fixed_detect_label refers to the user defined segmentation ratio of the training set test set, and the algorithm ultimately outputs anomaly labels. |              |     √      |          |      √       |              |
  |  fixed_detect_score  | Fixed_detect_score  refers to the user defined segmentation ratio of the training set test set, and the algorithm ultimately outputs abnormal scores. |              |     √      |          |              |      √       |
  | unfixed_detect_label | Unfixed_detect_label refers to the segmentation of the training set test set following the original data segmentation method, and the algorithm ultimately outputs abnormal labels. |              |     √      |          |      √       |              |
  | unfixed_detect_score | Unfixed_detect_score refers to the segmentation of the training set test set following the original data segmentation method, and the algorithm ultimately outputs abnormal scores. |              |     √      |          |              |      √       |
  |   all_detect_label   | All_detect_label refers to not dividing the training and testing sets, where all data is used as both the training and testing sets, and the algorithm ultimately outputs anomaly labels. |              |     √      |          |      √       |              |
  |   all_detect_score   | All_detect_score refers to not dividing the training and testing sets, where all data is used as both the training and testing sets, and the algorithm ultimately outputs anomaly scores. |              |     √      |          |              |      √       |
  - **函数原型如下：**

    - required_hyper_params 函数:

      ```
      """
      返回模型所需的超参数。
      该函数是可选的，且是静态的。
      
      :return: 一个字典，表示模型需要的超参数。
      :rtype: dict
      """
      ```
    
    - forecast_fit 函数训练模型
    
      ```
      """
      在时间序列数据上拟合模型。
      
      :param series: 时间序列数据。
      :type series: pd.DataFrame
      """
      ```
    
    - forecast 函数对模型进行预测
    
      ```
      """
      使用模型进行预测。
      
      :param pred_len: 预测长度。
      :type pred_len: int
      :param train: 用于拟合模型的训练数据。
      :type train: pd.DataFrame
      
      :return: 预测结果。
      :rtype: np.ndarray
      """
      ```
    
    - detect_label 函数对模型进行预测
    
      ```
      """
      使用模型进行异常检测并生成标签。
      
      :param train: 用于异常检测的训练数据。
      :type train: pd.DataFrame
      
      :return: 异常标签数组。
      :rtype: np.ndarray
      """
      ```
    
    - detect_score 函数对模型进行预测
    
      ```
      """
      使用模型计算异常得分。
      
      :param train: 用于计算得分的训练数据。
      :type train: pd.DataFrame
      
      :return: 异常得分数组。
      :rtype: np.ndarray
      """
      ```
    
    - __repr __函数模型名称的字符串表示
    
      ```
      """
      返回模型名称的字符串表示。
      
      :return: 返回模型名称的字符串表示。
      :rtype: str
      """
      ```
    
    

- **Configure your Configuration File**

  - modify the corresponding config under the folder `./ts_benchmark/config/`.

  - modify the contents in run_benchmark_demo.py.
  
  - **We strongly recommend using the pre-defined configurations in `./ts_benchmark_config/`. Create your own  configuration file only when you have a clear understanding of the configuration items.**

- **运行benchmark可参考如下格式：**

```
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-set-name "small_forecast" --adapter None "statistics_darts_model_adapter" --model-name "darts_models.darts_arima" "darts.models.forecasting.arima.ARIMA" --model-hyper-params "{\"p\":7}" "{}" 
```



### Introduction to Configuration Parameters

```
`./ts_benchmark/config/`.中包含如下配置文件

# 固定预测
./ts_benchmark/config/fixed_forecast_config.json
# 滚动预测
./ts_benchmark/config/rolling_forecast_config.json
# 固定方式切分数据集，算法输出异常标签
./ts_benchmark/config/fixed_detect_label_config.json
# 固定方式切分数据集，算法输出异常得分
./ts_benchmark/config/fixed_detect_score_config.json
# 按照数据集原始切分方式切分数据，算法输出异常标签
./ts_benchmark/config/unfixed_detect_label_config.json
# 按照数据集原始切分方式切分数据，算法输出异常得分
./ts_benchmark/config/unfixed_detect_score_config.json
```

**每一份配置文件中包含以下四部分内容：**

```
1."data_loader_config": 配置需要评估的数据集特征信息; type: dict
-----"data-set-name"：数据集尺寸大小以及是时序预测数据集还是时序异常检测数据集;type: str
------------"large_forecast":大规模时序预测数据集
------------"medium_forecast": 中等规模时序预测数据集
------------"small_forecast":小规模时序预测数据集
------------"large_detect":大规模时序异常检测数据集
------------"medium_detect": 中等规模时序异常检测数据集
------------"small_detect":小规模时序异常检测数据集
-----"feature_dict"：包含如下的键值; type: dict
------------"if_univariate"：true代表挑选单元数据集，false代表挑选多元数据集
------------"if_trend"：true代表数据集有趋势性，false代表数据集无趋势性，null代表不区分有无趋势性，默认为null
------------"has_timestamp"：true代表数据集有时间戳，false代表数据集无时间戳，null代表不区分有无时间戳，默认为null
------------"if_season"：true代表数据集有季节性，false代表数据集无季节性，null代表不区分有无季节性，默认为null
```

```
2."model_config": 配置需要评估的模型信息以及模型的推荐超参数; type: dict
-----"models"：模型信息，类型为：list[dict]；如下三个键值以及其对应value构成一个dict；
------------"adapter"：选定模型适配器，如不需要适配器传入空字符串
------------"model-name"：模型路径
------------"model-hyper-params"：输入给模型的参数，若参数名和"recommend_model_hyper_params"中参数名相同，则覆盖后者
-----"recommend_model_hyper_params"：有的算法必须输入一些参数，才能运行；如果用户没有输入这些参数，我们提供了默认参数，供这些算法自动获取对应参数值; type: dict
------------"input_chunk_length"：模型回看窗口长度
------------"output_chunk_length"：模型输出步长
```

```
3."model_eval_config": 配置需要的评估策略以及评价指标; type: dict
-----"metric_name": 评价指标名称; type：str；dict；list； 示例如下：
------------"all": 代表测评本仓库支持的所有评价指标
------------"mae"：测评 Mean Absolute Error
------------{"name": "mase", "seasonality": 10}
------------[{"name": "mase", "seasonality": 10},"mae",{"name": "mase", "seasonality": 2}]
-----"strategy_args":
------------"strategy_name"：评价策略名称，options：["fixed_forecast", "rolling_forecast", "fixed_detect_label", "fixed_detect_score", "unfixed_detect_label", "unfixed_detect_score","all_detect_label","all_detect_score"]
------------"pred_len":预测步长
------------"train_test_split"：划分训练集，测试集的比例
------------"stride"：滚动预测中跨越步长
------------"num_rollings"：滚动预测中最大滚动次数
```

**The various `strategy_name` within `model_eval_config` require different configuration parameters. The correspondence between `strategy_name` and stratege related parameters is as follows:**

| strategy_name      | pred_len | train_test_split | stride | num_rollings |
| ------------------ | -------- | ---------------- | ------ | ------------ |
| fixed_forecast     | √        |                  |        |              |
| rolling_forecast   | √        | √                | √      | √            |
| fixed_detect_label |          | √                |        |              |
| fixed_detect_score |          | √                |        |              |

- "unfixed_detect_label"、"unfixed_detect_score"、"all_detect_label"、"all_detect_score". These four strategies do not require configuration of stratege related parameters.

  

```
4."report_config": 配置如何呈现评估算法与baseline算法对比的性能结果; type: dict
-----"log_file_path"：需要对比性能的算法的log路径
-----"report_model"：选定要被进行比较的baseline算法； type：str，list； 示例如下
------------"all": 所有baseline算法
------------"single":不与baseline算法进行性能比较，只输出"log_file_path"中算法性能。
------------"darts_naivedrift": 指定的单个baseline算法
------------['darts_naivedrift', 'darts_statsforecastautoces']：指定的多个baseline算法
-----"report_metrics"：指定进行性能对比的指标，指标必须是"log_file_path"中已有的指标
-----"report_type"：options["mean","median"] 利用指标结果的平均值还是中值进行性能对比
-----"fill_type"：若出现null值，应该用什么值进行替换，若为"mean_value"则代表利用非null值的平均值替换null值
-----"threshold_value"：容忍null值个数对阈值，当null值数量不超过阈值比例，则利用"fill_type"替换null值；否则算法性能太差，应该报错。

```

​	
