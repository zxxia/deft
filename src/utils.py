import collections
import json
import re
import sys
import time
import numpy as np
import pandas as pd

def read_json_file(filename):
    with open(filename, 'r') as f:
        content = json.load(f, object_pairs_hook=collections.OrderedDict)
    return content

def write_json_file(filename, content):
    with open(filename, 'w') as f:
        json.dump(content, f, indent=4)
        f.flush()

def read_yaml_file(filename):
    with open(filename, 'r') as f:
        content = yaml.safe_load(f)
    return content

def sem(data):
    return np.std(data) / np.sqrt(len(data))

def parse_log(filename):
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
        return df[1:]
    raise ValueError("Unrecoginized file extension.")


class print_time:
    def __init__(self, desc, file=sys.stdout):
        self.desc = desc
        self.file = file

    def __enter__(self):
        print(self.desc, file=self.file)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s', file=self.file)

def get_config_name(config):
    if "resize_size" in config:
        w, h = config['resize_size']
        return f"{config['model_name']}_{w}x{h}_batch_{config['batch_size']}_sleep_{config['sleep_time']}"
    else:
        return f"{config['model_name']}_batch_{config['batch_size']}_sleep_{config['sleep_time']}"

def get_configs_name(config):
    name = ""
    for model in config['models']:
        if len(name) != 0:
            name += "_vs_"
        name += get_config_name(model)
    return name

def get_config_from_name(name: str):

    batch_size = int(re.findall("batch_(\d+)", name)[0])
    sleep_time = int(re.findall("sleep_(\d+)", name)[0])
    idx = name.find('_batch')
    model_name = name[:idx]
    # print(re.findall("[^_batch]*", name))
    # print(batch_size, sleep_time, model_name)
    return model_name, sleep_time, batch_size
