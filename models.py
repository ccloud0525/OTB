from service.sqlconfig import Base, SessionLocal
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, JSON, FLOAT, LargeBinary, \
    UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime


class DataSource(Base):
    __tablename__ = "datasource"
    datasourceid = Column(Integer, primary_key=True, index=True)
    config = Column(JSON)


class Chart(Base):
    __tablename__ = "chart"
    chartid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    config = Column(JSON)
    dataset = Column(JSON)
    mapping = Column(JSON)


class Dashboard(Base):
    __tablename__ = "dashboard"
    dashboardid = Column(Integer, primary_key=True, index=True)
    charts = Column(JSON)
    layout = Column(JSON)
    dashboard_name = Column(String)


class DataSet(Base):
    __tablename__ = "dataset"
    datasetid = Column(Integer, primary_key=True, index=True)
    query = Column(String)
    example_row = Column(JSON)
    config = Column(JSON)
    dataset_name = Column(String)


class ChartTemplate(Base):
    __tablename__ = "chart_template"
    cid = Column(Integer, primary_key=True, index=True)
    chart_name = Column(String)
    config = Column(JSON)


class Models(Base):
    __tablename__ = "model"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    model_config = Column(JSON)


class User(Base):
    __tablename__ = "user"

    user_id = Column(Integer, primary_key=True, index=True)
    # TODO: below 2 columns need to add length constraint
    username = Column(String, nullable=False, unique=True, index=True)
    password = Column(String, nullable=False)
    admin = Column(Boolean, nullable=False)


class InputFileInfo(Base):
    __tablename__ = "input_file_info"

    input_file_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False, index=True)

    # convenient for table join query
    user = relationship("User", backref="input_file_of_user")

    # TODO: should have length constraint many
    dataset_name = Column(String, nullable=False)

    file_name = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    if_univariate = Column(Boolean, nullable=False)
    has_timestamp = Column(Boolean, nullable=False)
    rugular = Column(Boolean, nullable=False)
    freq = Column(String, nullable=True)
    forecast_horizon = Column(Integer, nullable=True)
    abnormal_ratio = Column(FLOAT, nullable=True)
    train_lens = Column(String, nullable=False)
    val_lens = Column(String, nullable=False)
    train_has_label = Column(Boolean, nullable=True)
    val_has_label = Column(Boolean, nullable=True)
    data_sources = Column(String, nullable=False)
    licence = Column(String, nullable=False)
    other_situation = Column(String, nullable=True)

    example_row = Column(JSON, nullable=True)
    data_size = Column(String, nullable=True)
    task_type = Column(String, nullable=False)
    data_dimension_type = Column(String, nullable=False)

    pages_list = Column(JSON, nullable=True)


class LabelInputFilePages(Base):
    __tablename__ = "label_input_file_pages"

    input_file_id = Column(Integer, ForeignKey("input_file_info.input_file_id"), primary_key=True, index=True)
    page_num = Column(Integer, primary_key=True, index=True)
    page_data = Column(LargeBinary, nullable=False)


class InputData(Base):
    __tablename__ = "input_data"

    input_file_id = Column(Integer, ForeignKey("input_file_info.input_file_id"), primary_key=True, index=True)
    input_data = Column(LargeBinary, nullable=False)


class Task(Base):
    __tablename__ = "task"

    task_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False, index=True)

    # convenient for table join query
    user = relationship("User", backref="task_of_user")

    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    model_id = Column(Integer, ForeignKey("model.id"), nullable=False)
    model_name = Column(String, nullable=True)

    # input data file of this task
    input_file_id = Column(Integer, ForeignKey("input_file_info.input_file_id"), nullable=False, index=True)
    input_file_info = relationship("InputFileInfo", backref="task_of_input_file_info")

    # status of this task
    # TODO: meanings of different status code to be designed
    status = Column(Integer, nullable=False)

    task_type = Column(String, nullable=False)

    # model config
    strategy_args = Column(String, nullable=True)
    model_params = Column(String, nullable=True)


class ForecastResult(Base):
    __tablename__ = "forecast_result"

    forecast_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("task.task_id"), nullable=False, index=True)
    model_name = Column(String, nullable=False)
    strategy_args = Column(String, nullable=False)
    model_params = Column(String, nullable=False)
    # input data file's name
    file_name = Column(String, nullable=False)
    input_file_id = Column(Integer, ForeignKey("input_file_info.input_file_id"), nullable=False)

    forecast_metrics = Column(JSON, nullable=False)

    # train time cost
    fit_time = Column(FLOAT, nullable=True)
    # inference time cost
    inference_time = Column(FLOAT, nullable=True)

    # log info
    log_info = Column(String, nullable=True)

    __table_args__ = (UniqueConstraint('input_file_id', 'model_name', 'model_params', name='unique_set1'),)


class ForecastResultData(Base):
    __tablename__ = "forecast_result_data"

    forecast_id = Column(Integer, ForeignKey("forecast_result.forecast_id"), primary_key=True, index=True)
    forecast_data = Column(LargeBinary, nullable=False)
    date_df_idx_dict = Column(LargeBinary, nullable=False)


class LabelDetectResult(Base):
    __tablename__ = 'label_detect_result'

    label_detect_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("task.task_id"), nullable=False, index=True)

    model_name = Column(String, nullable=False)
    strategy_args = Column(String, nullable=False)
    model_params = Column(String, nullable=False)

    file_name = Column(String, nullable=False)
    input_file_id = Column(Integer, ForeignKey("input_file_info.input_file_id"), nullable=False)

    score_metrics = Column(JSON, nullable=True)
    label_metrics = Column(JSON, nullable=True)

    fit_time = Column(FLOAT, nullable=True)
    inference_time = Column(FLOAT, nullable=True)
    log_info = Column(String, nullable=True)


class LabelResultData(Base):
    __tablename__ = 'label_result_data'

    label_detect_id = Column(Integer, ForeignKey("label_detect_result.label_detect_id"), primary_key=True, index=True)
    inference_data = Column(LargeBinary, nullable=False)

    label_detect_result = relationship("LabelDetectResult", backref="label_result_data")


class ScoreResultData(Base):
    __tablename__ = 'score_result_data'

    label_detect_id = Column(Integer, ForeignKey("label_detect_result.label_detect_id"), primary_key=True, index=True)
    actual_data = Column(LargeBinary, nullable=False)
    inference_data = Column(LargeBinary, nullable=False)

    label_detect_result = relationship("LabelDetectResult", backref="score_result_data")


class TempFile(Base):
    __tablename__ = "temp_file"

    file_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False, index=True)

    user = relationship("User", backref="temp_file_of_user")

    file_name = Column(String, nullable=False, index=True)

    created_date = Column(DateTime, nullable=False)
    expire_date = Column(DateTime, nullable=False)

    example_row = Column(JSON, nullable=True)
    file_size = Column(String, nullable=False)
    data_dimension_type = Column(String, nullable=False)

    bucket_name = Column(String, nullable=False)
    object_name = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint('user_id', 'file_name', name='unique_set_temp_file'),)


class TempTask(Base):
    __tablename__ = "temp_task"

    task_id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False, index=True)
    user = relationship("User", backref="temp_task_of_user")

    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)

    model_id = Column(Integer, ForeignKey("train_model.model_id"), nullable=False)
    model_name = Column(String, nullable=False)

    file_id = Column(Integer, ForeignKey("temp_file.file_id"), nullable=False, index=True)

    status = Column(String, nullable=False)

    task_type = Column(String, nullable=True)

    model_params = Column(JSON, nullable=True)

    bucket_name = Column(String, nullable=True)
    object_name = Column(String, nullable=True)

    fit_time = Column(FLOAT, nullable=True)
    inference_time = Column(FLOAT, nullable=True)

    forecast_metrics = Column(JSON, nullable=True)


class TrainModel(Base):
    __tablename__ = "train_model"

    model_id = Column(Integer, primary_key=True, index=True)

    model_name = Column(String, nullable=False, index=True)
    model_path_name = Column(String, nullable=False)
