import yaml
from collections import namedtuple

CLASS_TO_IDX = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
IDX_TO_CLASS = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}

def class_label_conv(tag):
    try:
        digit = int(tag)
        return IDX_TO_CLASS[digit]
    except:
        return CLASS_TO_IDX[tag]


def to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = to_namedtuple(value)
        return namedtuple("config", obj.keys())(**obj)
    elif isinstance(obj, list):
        return [to_namedtuple(item) for item in obj]
    else:
        return obj


def load_config(config_path: str):
    with open(config_path, "r") as stream:
        args_dict = yaml.safe_load(stream=stream)

    return to_namedtuple(args_dict)
