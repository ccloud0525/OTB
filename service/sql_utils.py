import contextlib
from typing import List
import pandas as pd
import pickle
from functools import wraps

from service.sqlconfig import SessionLocal
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, DataError, OperationalError, InvalidRequestError

import logging


def get_db():
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logging.error(f'error while executing get_db: {e}')
        print(f'error while executing get_db: {e}')
        raise e
    finally:
        db.close()


def handle_db_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SQLAlchemyError as e:
            logging.error(f'error while executing {func.__name__}: {e}')
            print(f'error while executing {func.__name__}: {e}')
            raise e

    return wrapper


@contextlib.contextmanager
def get_db_no_depends():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
