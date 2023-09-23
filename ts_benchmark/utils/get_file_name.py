# -*- coding: utf-8 -*-
import os
import socket
import time


def get_log_file_name():
    """
    生成一个日志文件名后缀，包括以下信息：

    - 主机名 (hostname)
    - 当前时间戳 (timestamp)，以秒为单位，自Unix纪元以来的秒数
    - 进程的 PID（进程标识符）

    返回：
    str: 生成的日志文件名，格式为 '.timestamp.hostname.pid.csv'

    例如，如果主机名为 'myhost'，当前时间戳为 1631655702，当前进程ID为 12345，
    则返回的文件名可能是 '.1631655702.myhost.12345.csv'。
    """
    # 获取主机名
    hostname = socket.gethostname()

    # 获取当前时间戳（自Unix纪元以来的秒数）
    timestamp = int(time.time())

    # 获取进程的 PID（进程标识符）
    pid = os.getpid()

    # 构建文件名
    log_filename = f".{timestamp}.{hostname}.{pid}.csv"
    return log_filename
