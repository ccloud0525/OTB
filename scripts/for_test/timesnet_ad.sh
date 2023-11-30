python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --typical-data-name-list "MSL.csv" --data-set-name "small_detect" --model-name "time_series_library.TimesNet.TimesNet" --adapter "transformer_adapter1" --model-hyper-params "{\"batch_size\":128, \"seq_len\":100,\"d_model\":8, \"d_ff\":16, \"e_layers\":1, \"num_epochs\":1, \"pred_len\":0}" --report-method csv --gpus 1 --num-workers 1 --saved-path "timesnet_ad"
python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --typical-data-name-list "MSL.csv" --data-set-name "small_detect" --model-name "time_series_library.TimesNet.TimesNet" --adapter "transformer_adapter1" --model-hyper-params "{\"batch_size\":128, \"seq_len\":100,\"d_model\":8, \"d_ff\":16, \"e_layers\":1, \"num_epochs\":1, \"pred_len\":0}" --report-method csv --gpus 1 --num-workers 1 --saved-path "timesnet_ad"


python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json"  --data-set-name "large_detect" --model-name "time_series_library.TimesNet.TimesNet" --adapter "transformer_adapter1" --model-hyper-params "{\"batch_size\":128, \"seq_len\":100,\"d_model\":64, \"d_ff\":64, \"e_layers\":1, \"num_epochs\":3, \"pred_len\":0}" --report-method csv --gpus 1 --num-workers 8 --timeout 60000 --eval-backend ray --saved-path "univariate_timesnet_ad"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json"  --data-set-name "large_detect" --model-name "time_series_library.TimesNet.TimesNet" --adapter "transformer_adapter1" --model-hyper-params "{\"batch_size\":128, \"seq_len\":100,\"d_model\":64, \"d_ff\":64, \"e_layers\":1, \"num_epochs\":3, \"pred_len\":0}" --report-method csv --gpus 2 --num-workers 8 --timeout 60000 --eval-backend ray --saved-path "univariate_timesnet_ad_score"