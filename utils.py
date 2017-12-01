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


def maybe_use_cuda(module, use_cuda=True):
    """Helper function to convert Module for using cuda
    if its available and we intend to use it

    Args:
        module: nn.Module or autograd.Variable that contains `cuda` method
        use_cuda: boolean whose default value is true,
            since it's really recommended to train the
            model in GPU

    Return:
        void
    """
    if use_cuda and torch.cuda.is_available():
        return module.cuda()

    return module


def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')