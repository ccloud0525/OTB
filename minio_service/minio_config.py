import minio
import os
from datetime import datetime, timedelta

endpoint_URL = '113.31.110.212:9004'

access_key = 'minioaivisadmin'
secret_key = 'minioaivisadmin'

temp_input_files_bucket_name = "aivis-temp-files-input"
temp_raw_input_files_bucket_name = "aivis-temp-raw-files-input"

temp_task_output_bucket_name = "aivis-temp-task-output"

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

minio_client = minio.Minio(endpoint_URL, access_key, secret_key, secure=False)

legal_input_file_columns = ['date', 'data', 'cols']

EXPIRE_DAYS = 7
EXPIRE_TIMEDELTA = timedelta(days=EXPIRE_DAYS)
