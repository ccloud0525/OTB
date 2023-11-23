python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --typical-data-name-list "PSM.csv" "MSL.csv" "SMAP.csv"  --data-set-name "small_detect" --model-name "Anomaly_trans.AnomalyTransformer.AnomalyTransformer" --model-hyper-params "{\"batch_size\":128, \"num_epochs\":3}" --adapter "anomaly_trans_adapter"  --report-method csv --gpus 7 --num-workers 1 --saved-path "test_yuzhi"



python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --typical-data-name-list "PSM.csv" "MSL.csv" "SMAP.csv" --data-set-name "small_detect" --model-name "time_series_library.TimesNet.TimesNet" --adapter "transformer_adapter1" --model-hyper-params "{\"batch_size\":128, \"seq_len\":100,\"d_model\":8, \"d_ff\":16, \"e_layers\":1, \"num_epochs\":3, \"pred_len\":0}" --report-method csv --gpus 7 --num-workers 1 --saved-path "test_yuzhi"



python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --typical-data-name-list "PSM.csv" "MSL.csv" "SMAP.csv" --data-set-name "small_detect" --model-name "DCdetector.DCdetector.DCdetector" --adapter "dcdetector_adapter" --model-hyper-params "{\"batch_size\":128, \"win_size\":105, \"patch_size\":[3,5,7], \"num_epochs\":3}" --report-method csv --gpus 7 --num-workers 1 --saved-path "test_yuzhi"