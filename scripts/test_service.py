'''
* @author: 踏雪尋梅
*
* @create: 2024-03-18 21:42
*
* @description: 
'''
import asyncio
import pandas as pd

# 运行异步任务
import json
from unittest import TestCase, IsolatedAsyncioTestCase
from scripts.service import forecast_service
import pandas as pd


class Test(IsolatedAsyncioTestCase):
    async def test_forecast_service(self):
        input_file_path = "m4_hourly_dataset_385.csv"
        model_name = "time_series_library.Triformer.Triformer"
        config_path = "fixed_forecast_config_yearly.json"
        strategy_args = {
            "pred_len": 24
        }
        model_hyper_params = {
            "d_model": 32,
            "d_ff": 64,
            # "seq_len": 96,
            # "pred_len": 96
        }
        adapter = "transformer_adapter_single"

        res = await forecast_service(input_file_path, model_name, config_path, strategy_args, model_hyper_params,
                                     adapter)
        print(res)
        self.assertIsNotNone(res)

    async def test_forecast_service_auto(self):
        input_file_path = "m4_hourly_dataset_385.csv"
        model_name = "ensemble"
        config_path = "fixed_forecast_config_yearly.json"
        strategy_args = {
            "pred_len": 24
        }
        model_hyper_params = {
            "d_model": 32,
            "d_ff": 64,
            # "seq_len": 96,
            # "pred_len": 96
        }

        adapter = "transformer_adapter_single"

        res = await forecast_service(input_file_path, model_name, config_path, strategy_args, model_hyper_params,
                                     adapter)
        print(res)
        self.assertIsNotNone(res)
        return res

    async def test_forecast_service_by_model_path(self, model_path):
        input_file_path = "m4_hourly_dataset_385.csv"
        config_paht = "fixed_forecast_config_yearly.json"
        strategy_args = {
            "pred_len": 24
        }
        model_hyper_params = {
            "d_model": 32,
            "d_ff": 64,
            # "seq_len": 96,
            # "pred_len": 96
        }
        adapter = "transformer_adapter_single"

        print(f'testing model: {model_path}')
        res = await forecast_service(input_file_path, model_path, config_paht, strategy_args, model_hyper_params,
                                     adapter)
        print(res)
        self.assertIsNotNone(res)
        return res

    def test_specify_model(self):
        model = ''
        self.test_forecast_service_by_model_path(model)

    def test_all_models(self):
        for model_path in model_path_list:
            asyncio.run(self.test_forecast_service_by_model_path(model_path))

    def test(self):
        asyncio.run(self.test_forecast_service_auto())
        asyncio.run(self.test_forecast_service())


model_name_path_pair = """
Triformer -> "time_series_library.Triformer.Triformer"
PatchTST -> "time_series_library.PatchTST.PatchTST"
Nonstationary_Transformer -> "time_series_library.Nonstationary_Transformer.Nonstationary_Transformer"
Informer -> "time_series_library.Informer.Informer"
TimesNet -> "time_series_library.TimesNet.TimesNet"
FEDformer -> "time_series_library.FEDformer.FEDformer"
NLinear -> "time_series_library.NLinear.NLinear"
Linear -> "time_series_library.Linear.Linear"
DLinear -> "time_series_library.DLinear.DLinear"
FiLM -> "time_series_library.FiLM.FiLM"
MICN -> "time_series_library.MICN.MICN"
Crossformer -> "time_series_library.Crossformer.Crossformer"
TCN -> "darts_models_single.darts_tcnmodel"
Nbeats -> "darts_models_single.darts_nbeatsmodel"
Nhits -> "darts_models_single.darts_nhitsmodel"
RNN -> "darts_models_single.darts_blockrnnmodel"
"""

model_path_list = ['time_series_library.Triformer.Triformer',
                   'time_series_library.PatchTST.PatchTST',
                   'time_series_library.Nonstationary_Transformer.Nonstationary_Transformer',
                   'time_series_library.Informer.Informer',
                   'time_series_library.TimesNet.TimesNet',
                   'time_series_library.FEDformer.FEDformer',
                   'time_series_library.NLinear.NLinear',
                   'time_series_library.Linear.Linear',
                   'time_series_library.DLinear.DLinear',
                   'time_series_library.FiLM.FiLM',
                   'time_series_library.MICN.MICN',
                   'time_series_library.Crossformer.Crossformer',
                   'dart_models_single.darts_tcnmodel',
                   'dart_models_single.darts_nbeatsmodel',
                   'dart_models_single.darts_nhitsmodel',
                   'dart_models_single.darts_blockrnnmodel'
                   ]
