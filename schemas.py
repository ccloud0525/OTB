from typing import Union

from pydantic import BaseModel
from datetime import datetime


class TrainStartRequest(BaseModel):
    """
    send to train service
    """
    task_id: int

    bucket_name: str
    object_name: str

    model_id: int
    model_name: str

    config_path: str
    strategy_args: dict
    model_hyper_params: dict
    adapter: str


class TempTaskBase(BaseModel):
    start_time: datetime = datetime.now()
    end_time: Union[datetime, None] = None
    model_id: int
    model_name: str

    status: str

    model_params: Union[dict, None] = None


class TempTaskCreate(TempTaskBase):
    user_id: int
    file_id: int
    task_type: Union[str, None] = None


class TempTask(TempTaskCreate):
    task_id: int
    bucket_name: Union[str, None] = None
    object_name: Union[str, None] = None

    fit_time: Union[float, None] = None
    inference_time: Union[float, None] = None

    forecast_metrics: Union[dict, None] = None

    class Config:
        orm_mode = True
