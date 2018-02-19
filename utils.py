import torch
import yaml
import six

from collections import namedtuple


def load_yaml(config_path):
    if not isinstance(config_path, six.string_types):
        raise ValueError("Got {}, expected string", type(config_path))
    else:
        with open(config_path, "r") as yaml_file:
            config = yaml.load(yaml_file)
            return config
