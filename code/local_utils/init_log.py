#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:53:41 2023
@author: Boxiang Yun   School:ECNU   Email:boxiangyun@gmail.com
"""
import logging
import os
import random
from argparse import ArgumentParser
from logging import Logger
from logging.handlers import TimedRotatingFileHandler

# Third party libraries
import cv2
import numpy as np
import pandas as pd
import torch

def mkdir(path: str):
    """Create directory.

     Create directory if it is not exist, else do nothing.

     Parameters
     ----------
     path: str
        Path of your directory.

     Examples
     --------
     mkdir("data/raw/train/")
     """
    try:
        if path is None:
            pass
        else:
            os.stat(path)
    except Exception:
        os.makedirs(path)
        
def init_logger(log_name, log_dir=None):
    """日志模块
    Reference: https://juejin.im/post/5bc2bd3a5188255c94465d31
    日志器初始化
    日志模块功能:
        1. 日志同时打印到到屏幕和文件
        2. 默认保留近一周的日志文件
    日志等级:
        NOTSET（0）、DEBUG（10）、INFO（20）、WARNING（30）、ERROR（40）、CRITICAL（50）
    如果设定等级为10, 则只会打印10以上的信息

    Parameters
    ----------
    log_name : str
        日志文件名
    log_dir : str
        日志保存的目录

    Returns
    -------
    RootLogger
        Python日志实例
    """

    mkdir(log_dir)

    # 若多处定义Logger，根据log_name确保日志器的唯一性
    if log_name not in Logger.manager.loggerDict:
        logging.root.handlers.clear()
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        # 定义日志信息格式
        datefmt = "%Y-%m-%d %H:%M:%S"
        format_str = "[%(asctime)s] %(filename)s[%(lineno)4s] : %(levelname)s  %(message)s"
        formatter = logging.Formatter(format_str, datefmt)

        # 日志等级INFO以上输出到屏幕
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir is not None:
            # 日志等级INFO以上输出到{log_name}.log文件
            file_info_handler = TimedRotatingFileHandler(
                filename=os.path.join(log_dir, "%s.log" % log_name), when="D", backupCount=7
            )
            file_info_handler.setFormatter(formatter)
            file_info_handler.setLevel(logging.INFO)
            logger.addHandler(file_info_handler)

    logger = logging.getLogger(log_name)

    return logger