from typing import Dict, Any

from src.run_logging.line_formatter.format_value import format_value


class DoubleColumnsLineFormatter:
    def __init__(self,
                 column_width: int = 8,
                 seperator: str = " | ",
                 train_val_seperator: str = "/",
                 ):
        self.column_width = column_width
        self.seperator = seperator
        self.train_val_seperator = train_val_seperator

        self.train_val_metric_names = None
        self.other_metric_names = None
        self.header = None

        self.print_header_in = 0

    def __call__(self, *args, **kwargs):
        return self.create_line(*args, **kwargs)

    def set_metric_names(self, logs: Dict[str, Any]):
        train_val_mns = set()
        other_mns = set()

        for mn in logs.keys():
            if mn.startswith("Train_"):
                train_val_mns.add(mn.replace("Train_", ""))
            elif mn.startswith("Val_"):
                train_val_mns.add(mn.replace("Val_", ""))
            else:
                other_mns.add(mn)

        self.train_val_metric_names = sorted(list(train_val_mns))
        self.other_metric_names = sorted(list(other_mns))

    def create_header(self):
        cw = self.column_width
        cw2 = 2 * cw + len(self.train_val_seperator)

        other_print_names = [mn + ":" for mn in self.other_metric_names]
        other_print_names = [f"{name:{cw}.{cw}}" for name in other_print_names]

        tv_print_names = [mn + ":" for mn in self.train_val_metric_names]
        tv_print_names = [f"{name:{cw2}.{cw2}}" for name in tv_print_names]

        print_names = other_print_names + tv_print_names

        header = self.seperator.join(print_names) + self.seperator
        return header

    def create_line(self, logs: Dict[str, Any]):
        cw = self.column_width

        if self.train_val_metric_names is None:
            self.set_metric_names(logs)

        values_line = self.get_values_line(cw, logs)
        return self.add_header_if_time(values_line)

    def get_values_line(self, cw, logs):
        value_strs = []

        for metric_name in self.other_metric_names:
            value = logs.get(metric_name, -1.)
            value_str = format_value(value, cw)
            value_strs.append(value_str)

        for metric_name in self.train_val_metric_names:
            train_value = logs.get("Train_" + metric_name, -1.)
            train_value_str = format_value(train_value, cw)
            val_value = logs.get("Val_" + metric_name, -1.)
            val_value_str = format_value(val_value, cw)

            value_strs.append(train_value_str + "/" + val_value_str)

        values_line = self.seperator.join(value_strs) + self.seperator
        return values_line

    def add_header_if_time(self, line):
        if self.header is None:
            self.header = self.create_header()

        if self.print_header_in == 0:
            self.print_header_in = 10
            line = self.header + "\n" + line

        self.print_header_in -= 1
        return line


def remove_substrings(original: str, *args):
    return_str = original
    for remove_string in args:
        return_str = return_str.replace(remove_string, "")
    return return_str


def either_is_none(*args):
    return any([arg is None for arg in args])
