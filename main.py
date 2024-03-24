from io import BytesIO
from typing import Union, List

from aio_pika import connect, IncomingMessage, connect_robust
from aio_pika.abc import AbstractIncomingMessage
import asyncio
from fastapi import FastAPI

from datetime import datetime

import logging
import json
import pandas as pd
import schemas
import os

from service.sql_utils import get_db_no_depends

import pickle

from scripts.service import forecast_service
from minio_service.minio_files_interface import download_dataframe_from_minio, upload_bytes_file_to_minio
from minio_service.minio_config import temp_task_output_bucket_name

from service.temp_task_service.temp_task_db_service import complete_temp_task


class RabbitMQManager:
    def __init__(self):
        self.connection = None
        self.channel = None


rabbitmq_manager = RabbitMQManager()


async def main():
    connection = await connect_robust('amqp://guest:guest@113.31.110.212:5672')
    assert connection is not None
    print('connected to rabbitmq')

    channel = await connection.channel()
    assert channel is not None
    print('created channel')

    await channel.set_qos(prefetch_count=2)
    task_queue = await channel.declare_queue('task_queue', durable=True)

    print(f'awaiting tasks from queue: task_queue')

    await task_queue.consume(on_message)

    await asyncio.Future()


callback = None


async def fake_execute_task(task):
    for i in range(30):
        print(f'executing task: {task}')
        await asyncio.sleep(1)
    print(f'fake task execution completed: {task}')


async def forecast_wrap(task_request: schemas.TrainStartRequest):
    input_file_path = 'm4_daily_dataset_2486.csv'
    input_df: pd.DataFrame = download_dataframe_from_minio(bucket_name=task_request.bucket_name,
                                                           object_name=task_request.object_name)
    print(f'type of input_df: {type(input_df)}')
    print(f'forecast_wrap: task_request: {task_request}')

    csv_path = os.path.join('dataset', task_request.object_name)
    input_df.to_csv(csv_path, index=False)

    res = await forecast_service(
        input_file_path=task_request.object_name,
        model=task_request.model_name,
        config_path=task_request.config_path,
        strategy_args=task_request.strategy_args,
        model_hyper_params=task_request.model_hyper_params,
        adapter=task_request.adapter
    )

    if os.path.isfile(csv_path):
        os.remove(csv_path)
        print(f'file removed: {csv_path}')

    print(res)
    print(f'type of res: {type(res)}')
    return res


execute_task = forecast_wrap


async def on_message(message: AbstractIncomingMessage):
    async with message.process():
        task = message.body.decode()
        print(f'received task: {task}')
        task_request: dict = json.loads(task)
        task_request: schemas.TrainStartRequest = schemas.TrainStartRequest(**task_request)

        # execute task with dependency injection function
        result: Union[pd.DataFrame, List] = await execute_task(task_request)
        res_bytes = pickle.dumps(result)
        object_name = f'task_id_{task_request.task_id}.pkl'
        upload_bytes_file_to_minio(res_bytes, temp_task_output_bucket_name, object_name)
        print(f'saved result to minio: {object_name}')

        with get_db_no_depends() as db:
            complete_temp_task(task_id=task_request.task_id, end_time=datetime.now(),
                               status='success', bucket_name=temp_task_output_bucket_name,
                               object_name=object_name, db=db)
            print(f'db updated for task_id: {task_request.task_id}')


async def publish_result(result):
    channel = rabbitmq_manager.channel


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Key board interrupt detected, exiting...')
    except Exception as e:
        print(e)
