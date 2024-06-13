
import unittest

from src import run_logging
from src.run_logging import line_formatter
from src.run_logging.line_formatter import format_value


class TestLineFormatter(unittest.TestCase):
    def test_single_columns_line_formatter(self):
        formatter = line_formatter.SingleColumnLineFormatter(column_width=4)
        logs = {"Aaa": 1., "Bbbbbbbb": 2., "C": 0.03, "Train_Accuracy": 0.8}
        lines = formatter.create_line(logs)
        header = lines.split("\n")[0]
        values = lines.split("\n")[1]
        self.assertEqual(header, "Aaa: | Bbbb | C:   | TAcc | ")
        self.assertEqual(values, "1.00 | 2.00 | 0.03 | 0.80 | ")

    def test_double_columns_line_formatter(self):
        formatter = run_logging.DoubleColumnsLineFormatter(column_width=6)
        logs = {"Train_Acc": .8, "Val_Acc": .7, "Epoch": 1,
                "Train_Loss_Value": 1.2, "Val_Loss_Value": 1.3,
                "Small_Number": 1.2345e-8}
        lines = formatter.create_line(logs)

        header = lines.split("\n")[0]
        values = lines.split("\n")[1]

        goal_h = "Epoch: | Small_ | Acc:          | Loss_Value:   | "
        goal_v = "     1 | 1.2e-8 |   0.80/  0.70 |   1.20/  1.30 | "
        self.assertEqual(header, goal_h)
        self.assertEqual(values, goal_v)

    def test_wide_format_value(self):
        self.assertEqual(format_value(3.141526, 8, 3), "    3.14")
        self.assertEqual(format_value(12345.678, 8, 3), "   12346")
        self.assertEqual(format_value(4e-3, 8, 3), "   0.004")

        self.assertEqual(format_value(1.11111e12, 8, 3), " 1.11e12")
        self.assertEqual(format_value(123456789, 8, 3), "  1.23e8")
        self.assertEqual(format_value(123456789, 5, 3), "1.2e8")

        self.assertEqual(format_value(1.11111e-12, 8, 3), "1.11e-12")
        self.assertEqual(format_value(1.11111e-12, 6, 3), " 1e-12")

        self.assertEqual(format_value(-2.8, 6, 3), " -2.80")
        self.assertEqual(format_value(-123456789, 8, 3), " -1.23e8")

        self.assertEqual(format_value(1.1111, 8, 3), "    1.11")
        self.assertEqual(format_value(1.0000, 8, 3), "    1.00")

        self.assertEqual(format_value(1, 8, 3), "       1")

