python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-set-name "small_forecast" --typical-data-name-list "finance_33.csv" --model-name "time_series_library.PatchTST.PatchTST" --model-hyper-params "{\"d_model\":32, \"d_ff\":64}" --metric-name \"mae\" \"mse\" \"rmse\" \"mape\" \"smape\" "{\"name\": \"mase\", \"seasonality\": 10}" \"wape\" \"msmape\"  --adapter "transformer_adapter_single" --gpus 0 --num-workers 1 --timeout 600