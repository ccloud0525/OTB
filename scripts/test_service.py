'''
* @author: 踏雪尋梅
*
* @create: 2024-03-18 21:42
*
* @description: 
'''
import json
from unittest import TestCase
from scripts.service import forecast_service
import pandas as pd


class Test(TestCase):
    def test_forecast_service(self):
        input_file_path = "m4_daily_dataset_2486.csv"
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

        res = forecast_service(input_file_path, model_name, config_path, strategy_args, model_hyper_params,adapter)
        print(res)
        self.assertIsNotNone(res)

    def test_forecast_service_auto(self):
        input_file_path = "m4_daily_dataset_2486.csv"
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
    
        res = forecast_service(input_file_path, model_name, config_path, strategy_args, model_hyper_params,adapter)
        print(res)
        self.assertIsNotNone(res)
