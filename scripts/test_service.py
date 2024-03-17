from unittest import TestCase
from scripts.service import forecast_service
import pandas as pd


class Test(TestCase):
    def test_forecast_service(self):
        input_file_path = "test/m1_quarterly_dataset_1.csv"
        model_name = "time_series_library.Triformer.Triformer"
        config_path = "rolling_forecast_config.json"
        strategy_args = {
            "pred_len": 24
        }
        model_hyper_params = {
            "d_model": 32,
            "d_ff": 64,
            "seq_len": 96,
            "pred_len": 96
        }
        res = forecast_service(input_file_path, model_name, config_path, strategy_args, model_hyper_params)
        self.assertIsNotNone(res)
