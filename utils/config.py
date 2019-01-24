import json
from bunch import Bunch
import os
import utils.dirs
import numpy as np


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(os.path.join(json_file), 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.n_gpus = len(config.gpus.split(','))
    config.devices = ['/gpu:{}'.format(i) for i in range(config.n_gpus)]
    config.output_dim = config.depth_dim*config.spatial_dim**2
    config.summary_dir = os.path.join("experiments", config.exp_name, "summary/")
    config.results_dir = os.path.join("experiments", config.exp_name, "results/")
    config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoint/")
    config.database_dir = os.path.join("databases/tfrecords", config.database_name)
    info = np.load(os.path.join(config.database_dir, "info.npy")).item()
    if config.model == "single_chart":
        config.num_iter_per_epoch = info["train_size"] // config.batch_size*(config.depth_dim//3)
        config.actual_depth = 3
    else:
        config.num_iter_per_epoch = info["train_size"]//config.batch_size
    # create the experiments dirs
    utils.dirs.create_dirs([config.summary_dir, config.checkpoint_dir, config.results_dir])
    return config
