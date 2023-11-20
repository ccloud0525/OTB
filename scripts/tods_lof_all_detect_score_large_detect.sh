#!/bin/bash

python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --data-set-name "small_detect" --typical-data-name-list "S3-ADL4.test.csv@79.csv"  --model-name "tods_models.tods_lofski" --report-method csv --gpus 7  --num-workers 1 --saved-path "test"