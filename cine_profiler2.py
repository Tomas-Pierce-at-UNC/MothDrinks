#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:51:02 2023

@author: tomas
"""

import time
import datetime
import logging
import functools

logger = logging.getLogger(__name__)
filehandler = logging.FileHandler("exc_time.log", "a")
logger.addHandler(filehandler)
NANO = 10**9;

def log_exc_time(func):
    "Function decorator that causes the decorated function to log its execution time"
    
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        now = datetime.datetime.now()
        start = time.time_ns()
        out = func(*args, **kwargs)
        end = time.time_ns()
        elapsed_ns = end - start
        elapsed_s = elapsed_ns / NANO
        msg = f"function {func} took {elapsed_s} seconds or {elapsed_ns} nanoseconds to run at {now}"
        logger.log(logging.INFO, msg)
        return out
    
    return new_func

