# system libs
import os
import sys
import json
import random
import logging

# numpy lib
import numpy as np

# pytorch lib
import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

# set print precision
np.set_printoptions(precision=8)

# mkdir log
# mkdir torch_saved
# mkdir data

def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param

def prepare_env():
    dir_names = ["log", "torch_saved"]
    for dir in dir_names:
        try:
            os.makedirs(dir)
        except OSError:
            if os.path.exists(dir):
                # We are nearly safe
                pass
            else:
                # There was an error on creation, so make sure we know about it
                raise

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


def set_gpu(gpus):
    """
    Sets the GPU to be used for the run
    Parameters
    ----------
    gpus:           List of GPUs to be used for the run

    Returns
    -------

    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object
    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + \
        name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def get_combined_results(left_results, right_results):
    """
    Computes the average based on head and tail prediction results
    Parameters
    ----------
    left_results:   Head prediction results
    right_results: 	Left prediction results

    Returns
    -------
    Average prediction results

    """

    results = {}
    count = float(left_results['count'])

    results['left_mr'] = round(left_results['mr'] / count, 5)
    results['left_mrr'] = round(left_results['mrr']/count, 5)
    results['right_mr'] = round(right_results['mr'] / count, 5)
    results['right_mrr'] = round(right_results['mrr']/count, 5)
    results['mr'] = round(
        (left_results['mr'] + right_results['mr']) / (2*count), 5)
    results['mrr'] = round(
        (left_results['mrr'] + right_results['mrr'])/(2*count), 5)

    for k in range(10):
        results['left_hits@{}'.format(k+1)] = round(
            left_results['hits@{}'.format(k+1)]/count, 5)
        results['right_hits@{}'.format(k+1)] = round(
            right_results['hits@{}'.format(k+1)]/count, 5)
        results['hits@{}'.format(k+1)] = round((left_results['hits@{}'.format(
            k+1)] + right_results['hits@{}'.format(k+1)])/(2*count), 5)
    return results

def get_param(shape):
	param = Parameter(torch.Tensor(*shape)); 	
	xavier_normal_(param.data)
	return param

def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def conj(a):
	a[..., 1] = -a[..., 1]
	return a

def cconv(a, b):
	return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def ccorr(a, b):
	return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))