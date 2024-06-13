from datetime import datetime


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = datetime.now()

    @property
    def seconds_elapsed(self):
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def formatted(self):
        return format_seconds(self.seconds_elapsed)

    # def stop(self):
    #     self.stop_time = datetime.now()


def format_seconds(seconds):
    """ Formats seconds into a human-readable string. """
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 60 * 60:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / (60 * 60):.1f} h"


def timed(f):
    def wrapper(*args, **kwargs):
        timer = Timer()
        timer.start()
        result = f(*args, **kwargs)
        time_str = format_seconds(timer.seconds_elapsed)
        # fn_call_str = f"{f.__name__}({args = }, {kwargs = })"
        # print(f"Function \"{fn_call_str}\" took {time_str}.")
        print(f"Function \"{f.__name__}()\" took {time_str}.")
        return result

    return wrapper
