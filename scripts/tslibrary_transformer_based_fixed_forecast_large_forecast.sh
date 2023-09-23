#!/bin/bash

 python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-set-name "large_forecast"  --model-name "time_series_library.TimesNet.TimesNet" --adapter "transformer_adapter"
