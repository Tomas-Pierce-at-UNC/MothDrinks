
import time


def print_time(func):

    nano = 10 ** 9

    def new_func(*args, **kwargs):
        start = time.time_ns()
        out = func(*args, **kwargs)
        end = time.time_ns()
        elapsed = (end - start) / nano
        print(elapsed, "seconds", func)
        return out

    return new_func