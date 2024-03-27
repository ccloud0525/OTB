import schemas, models
from datetime import datetime

from sqlalchemy.orm import Session

from service.sql_utils import get_db_no_depends, handle_db_exception


@handle_db_exception
def complete_temp_task(task_id: int, end_time: datetime, status: str,db:Session, bucket_name: str = None,
                       object_name: str = None):
    """
    fill columns: end_time, status, bucket_name, object_name
    :param task_id:
    :param end_time:
    :param status:
    :param bucket_name:
    :param object_name:
    :param db:
    :return: 
    """
    db_temp_task = db.query(models.TempTask).filter(models.TempTask.task_id == task_id).first()

    db_temp_task.end_time = end_time
    db_temp_task.status = status
    if bucket_name:
        db_temp_task.bucket_name = bucket_name
    if object_name:
        db_temp_task.object_name = object_name
    db.commit()
