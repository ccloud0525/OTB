python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-set-name "small_forecast"
--typical-data-name-list "ETTh1.csv" "ETTh2.csv" "ETTm1.csv" "ETTm2.csv" "traffic_hourly_dataset.csv" "electricity_hourly_dataset.csv" "weather.csv" --strategy-args "{\"pred_len\":96}"  --model-name   "darts_models.darts_tidemodel"
--model-hyper-params "{\"n_epochs\":10,\"input_chunk_length\":96,\"output_chunk_length\":96}"
--gpus 0  --num-workers 1 --timeout 6000 --saved-path "tide"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-set-name "small_forecast"
--typical-data-name-list "ETTh1.csv" "ETTh2.csv" "ETTm1.csv" "ETTm2.csv" "traffic_hourly_dataset.csv" "electricity_hourly_dataset.csv" "weather.csv" --strategy-args "{\"pred_len\":192}"  --model-name   "darts_models.darts_tidemodel"
--model-hyper-params "{\"n_epochs\":10,\"input_chunk_length\":96,\"output_chunk_length\":192}"
--gpus 0  --num-workers 1 --timeout 6000 --saved-path "tide"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-set-name "small_forecast"
--typical-data-name-list "ETTh1.csv" "ETTh2.csv" "ETTm1.csv" "ETTm2.csv" "traffic_hourly_dataset.csv" "electricity_hourly_dataset.csv" "weather.csv" --strategy-args "{\"pred_len\":336}"  --model-name   "darts_models.darts_tidemodel"
--model-hyper-params "{\"n_epochs\":10,\"input_chunk_length\":96,\"output_chunk_length\":336}"
--gpus 0  --num-workers 1 --timeout 6000 --saved-path "tide"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-set-name "small_forecast"
--typical-data-name-list "ETTh1.csv" "ETTh2.csv" "ETTm1.csv" "ETTm2.csv" "traffic_hourly_dataset.csv" "electricity_hourly_dataset.csv" "weather.csv" --strategy-args "{\"pred_len\":720}"  --model-name   "darts_models.darts_tidemodel"
--model-hyper-params "{\"n_epochs\":10,\"input_chunk_length\":96,\"output_chunk_length\":720}"
--gpus 0  --num-workers 1 --timeout 6000 --saved-path "tide"