## AutoML

### 1 Pipeline

![pipeline](AutoML.assets/pipeline.png)

- 在生成Label的过程中，调用AutoML/single_forecast_result/data_process.py生成数据 dataset_algorithm.npy，表明每一条单变量数据集对应的最好算法（one-hot label）
- 处理dataset，调用AutoML/train_ts2vec.py，使用TS2Vec，预训练一个可以提取时间序列元特征的Encoder，同时提取训练集所有数据集的特征，存储为data.pkl
- 训练基于LightGBM的classifier，保存最优点的参数至ckpt
- 封装EnsembleModel类实现集成模型的快速搭建和实例化
  

### 2 启用方式

#### 2.1 结合OTB启用

```shell
# 指定model-name为ensemble即可
python run_benchmark.py --config-path "fixed_forecast_config.json" --data-set-name "small_forecast" --model-name "ensemble" --adapter "transformer_adapter" --eval-backend sequential --gpus 1 --num-workers 1 --timeout 60000
```

#### 2.2 单独使用

```python
# example
from scripts.AutoML.model_ensemble import EnsembleModel
data = DataPool().get_series(series_name)
train_length = len(data) - self.pred_len
if train_length <= 0:
    raise ValueError("The prediction step exceeds the data length")
train, test = split_before(data, train_length)  # 分割训练和测试数据
            
pred_len = 48
model = EnsembleModel(
                    recommend_model_hyper_params={"input_chunk_length":48,"output_chunk_length:48"},
  									# 需要输入长度和输出长度作为超参，这是内部模型所需参数的一个超集
                    dataset=data,
 										# 传入数据集，内部会自动用预训练好的TS2vec提取数据集特征，并且使用预训练好的classifier选择模型
                    top_k=5, 
  									# ensemble的模型个数
                )

model.forecast_fit(train, ratio=0.875) # ratio是指对于train，有些模型需要划分valid的比例
predict = model.forecast(pred_len, train)
```
