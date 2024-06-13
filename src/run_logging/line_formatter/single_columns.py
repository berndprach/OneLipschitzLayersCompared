from typing import Dict, Any

from src.run_logging.line_formatter import format_value


class SingleColumnLineFormatter:
    def __init__(self, column_width: int = 8, seperator: str = " | "):
        self.column_width = column_width
        self.seperator = seperator

        self.metric_names = None
        self.header = None

        self.print_header_in = 0

    def __call__(self, *args, **kwargs):
        return self.create_line(*args, **kwargs)

    def create_header(self):
        print_names = get_print_names(self.metric_names, self.column_width)
        header = self.seperator.join(print_names) + self.seperator
        return header

    def create_line(self, logs: Dict[str, Any]):
        if self.metric_names is None:
            self.metric_names = sorted(list(logs.keys()))

        values_line = self.get_values_line(logs)
        return self.add_header_if_time(values_line)

    def get_values_line(self, logs):
        value_strs = []
        for metric_name in self.metric_names:
            value = logs.get(metric_name, -1.)
            value_strs.append(format_value(value, self.column_width))
        value_line = self.seperator.join(value_strs) + self.seperator
        return value_line

    def add_header_if_time(self, line):
        if self.header is None:
            self.header = self.create_header()

        if self.print_header_in == 0:
            self.print_header_in = 10
            line = self.header + "\n" + line

        self.print_header_in -= 1
        return line


def get_print_names(metric_names, column_width):
    print_names = [mn + ":" for mn in metric_names]
    print_names = [mn.replace("Train_", "T") for mn in print_names]
    print_names = [mn.replace("Val_", "V") for mn in print_names]
    print_names = [f"{name:<{column_width}.{column_width}}"
                   for name in print_names]
    return print_names
