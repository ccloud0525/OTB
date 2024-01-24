import lightgbm as lgb
import pandas as pd
from .ts2vec_model.datautils import process_data
from .ts2vec import TS2Vec
from .ts2vec_model.utils import init_dl_program
import numpy as np
from .ALGORITHMS import ALGORITHMS
import os


def model_ensemble(data, k, pred_len, sample_len):
    device = init_dl_program(0, seed=301, max_threads=None)
    data = data.reset_index(drop=True)
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
    sample_data = data[:, -(sample_len + pred_len) : -pred_len, :]
    model = TS2Vec(input_dims=sample_data.shape[-1], device=device, **config)
    model.load("AutoML/ts2vec_model/training/0118/model.pkl")

    repr = model.encode(
        sample_data, causal=True, sliding_length=1, sliding_padding=2, batch_size=256
    )

    gbm = lgb.Booster(model_file=f"AutoML/ckpt/best_gbm_1.txt")
    test_feature = np.reshape(repr, -1)
    pred = np.reshape(gbm.predict([test_feature]), -1)

    def get_index(lst, k):
        np_lst = np.array(lst)
        indices = np.argsort(-np_lst)
        return indices[0:k]

    indices = get_index(pred, k)
    model_name_lst = [ALGORITHMS[index] for index in indices]
    return model_name_lst
