from minio_service.minio_config import minio_client
import pandas as pd
import os
from io import BytesIO
from minio.error import S3Error

from typing import Union


def upload_dataframe_to_minio(df: pd.DataFrame, bucket_name: str, object_name: str):
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    try:
        minio_client.put_object(bucket_name=bucket_name, object_name=object_name, data=csv_buffer,
                                length=csv_buffer.getbuffer().nbytes)
    except S3Error as e:
        print(f'error while uploading {object_name} to {bucket_name} bucket: {e}')


def upload_bytes_file_to_minio(file: bytes, bucket_name: str, object_name: str):
    """
    for uploading raw csv file
    :param file:
    :param bucket_name:
    :param object_name:
    :return:
    """
    try:
        minio_client.put_object(bucket_name=bucket_name, object_name=object_name, data=BytesIO(file),
                                length=len(file))
    except S3Error as e:
        print(f'error while uploading {object_name} to {bucket_name} bucket: {e}')


def download_dataframe_from_minio(bucket_name: str, object_name: str) -> Union[pd.DataFrame, None]:
    try:
        response = minio_client.get_object(bucket_name=bucket_name, object_name=object_name)
        df = pd.read_csv(response)
        return df
    except S3Error as e:
        print(f'error while downloading {object_name} from {bucket_name} bucket: {e}')


def delete_file_from_minio(bucket_name: str, object_name: str):
    try:
        minio_client.remove_object(bucket_name=bucket_name, object_name=object_name)
    except S3Error as e:
        print(f'error while deleting {object_name} from {bucket_name} bucket: {e}')
