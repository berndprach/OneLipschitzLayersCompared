import importlib
import sys


def run_main():
    args = sys.argv[1:]
    file_path = args[0]
    sub_args = args[1:]
    sub_args_str = ", ".join(sub_args)

    print(f"Running main({sub_args_str}) from {file_path}")

    import_str = get_import_str(file_path)

    print(f"> import {import_str} as module")
    module = importlib.import_module(import_str)

    fn = getattr(module, "main")
    print(f"> module.main({sub_args_str})")
    fn(*sub_args)


def get_import_str(file_path):
    import_str = file_path.replace("/", ".")
    import_str = import_str.replace(".py", "")
    import_str = import_str.replace(".\\", "")
    import_str = import_str.replace("\\", ".")
    return import_str


def print_torch_version():
    import torch
    print(f"{torch.__version__ = }")


def print_versions():
    import torch
    python_version = sys.version
    print(f"{torch.__version__ = }, {python_version =}")


if __name__ == "__main__":
    print_versions()
    run_main()
