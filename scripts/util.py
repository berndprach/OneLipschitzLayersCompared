import inspect
from typing import Callable


def convert_arguments_from_strings(fun: Callable):
    """ Convert arguments from strings to their annotated types."""
    sig = inspect.signature(fun)
    arg_types = [v.annotation for v in sig.parameters.values()]

    def new_fun(*args):
        new_args = tuple(arg_type(v) for arg_type, v in zip(arg_types, args))
        print(f"Converted arguments from {args} to {new_args}.")
        return fun(*new_args)

    return new_fun
