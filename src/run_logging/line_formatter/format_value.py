import math


def format_value(value, column_width, max_precision=3):
    if isinstance(value, int):
        value_str = str(value)
        if len(value_str) > column_width:
            value = float(value)
            return format_value(value, column_width, max_precision)
    elif isinstance(value, float):
        try:
            rounded_value = round_significant_post_comma(value, max_precision)
            value_str = str(rounded_value)
            while sum(c.isdigit() for c in value_str) < max_precision:
                value_str = value_str + "0"

            if len(value_str) > column_width:
                exp_l = get_exp_notation_length(value)
                # if value < 0:
                #     exp_l += 1
                p = min(max_precision-1, column_width-exp_l)
                exponential_str = f"{value:.{p}e}"
                value_str = clean_exponential(exponential_str)
        except (OverflowError, ValueError):
            value_str = str(value)
    else:
        value_str = str(value)
    # print(f"{value_str = }")
    return f"{value_str:>{column_width}.{column_width}}"


def round_significant_post_comma(x, significant):
    leading_digit_index = int(math.floor(math.log10(abs(x))))
    rounding_index = leading_digit_index - significant + 1
    decimals = max(0, -rounding_index)
    # print(f"{leading_digit_index = }")
    # print(f"{rounding_index = }")
    # print(f"{decimals = }")
    rounded = round(x, decimals)
    if decimals <= 0:
        rounded = int(rounded)
    return rounded


def keep_first_digits_only(value, max_precision):
    value_str = ""
    digits_seen = 0
    for d in str(value):
        if d.isdigit() and digits_seen >= max_precision:
            value_str += "0"
        else:
            value_str += d
        if d.isdigit():
            digits_seen += 1
    return value_str


def remove_extra_decimals(value, max_precision):
    raise NotImplementedError("Fails for 1.11e-12!")
    value_str = ""
    digits_seen = 0
    dot_seen = False
    non_zero_seen = False

    for d in str(value):
        value_str += d
        if d == ".":
            dot_seen = True
        if d.isdigit() and d != "0":
            non_zero_seen = True
        if d.isdigit() and non_zero_seen:
            digits_seen += 1
        if dot_seen and digits_seen >= max_precision:
            break
    return value_str


def clean_exponential(v: str):
    v = v.replace("e+0", "e")
    v = v.replace("e+", "e")
    v = v.replace("e-0", "e-")
    return v


def get_exp_notation_length(v: float):
    if v >= 10**10:  # 1.1e10
        return len("1.1e10") - 1
    if v >= 1.:
        return len("1.1e0") - 1
    if v >= 10**-10:
        return len("1.1e-1") - 1
    else:
        return len("1.1e-10") - 1


def remove_substrings(original: str, *args):
    return_str = original
    for remove_string in args:
        return_str = return_str.replace(remove_string, "")
    return return_str

