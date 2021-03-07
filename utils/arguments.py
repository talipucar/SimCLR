"""
Main function for training for CNN-based encoder using self-supervised learning.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import os
import pprint
import torch as th
from argparse import ArgumentParser
from os.path import dirname, abspath
from utils.utils import get_runtime_and_model_config, print_config

def get_arguments():
    # Initialize parser
    parser = ArgumentParser()
    # Dataset can be provided via command line
    parser.add_argument("-d", "--dataset", type=str, default="STL10")
    # Input image size
    parser.add_argument("-img", "--image_size", type=int, default=96)
    # Input channel size
    parser.add_argument("-ch", "--channel_size", type=int, default=3)
    # GPU device number as in "cuda:0". Defaul is 0.
    parser.add_argument("-cuda", "--cuda_number", type=str, default='0')
    # Experiment number if MLFlow is on
    parser.add_argument("-ex", "--experiment", type=int, default=1)
    # Tune a pre-trained model
    parser.add_argument("-t", "--tune", type=bool, default=False)
    # Return parser arguments
    return parser.parse_args()

def get_config(args):
    # Get path to the root
    root_path = dirname(abspath(__file__))
    # Get path to the runtime config file
    config = os.path.join(root_path, "config", "runtime.yaml")
    # Load runtime config from config folder: ./config/
    config = get_runtime_and_model_config()
    # Copy models argument to config to use later
    config["dataset"] = args.dataset
    # Copy image size argument to config to use later
    config["img_size"] = args.image_size
    # Copy model tuning argument to config to use later
    config["tune_model"] = args.tune
    # Copy channel size argument to config to modify default architecture in model config
    config["conv_dims"][0][0] = args.channel_size
    # Define which device to use: GPU or CPU
    config["device"] = th.device('cuda:'+args.cuda_number if th.cuda.is_available() else 'cpu')
    # If device type is GPU, keep multi_gpu setting as the one defined by user. Otherwise, turn it off. 
    config["multi_gpu"] = config["multi_gpu"] if th.cuda.is_available() else False
    print(f"Device being used is {config['device']}")
    # Return
    return config

def print_config_summary(config, args):
    # Summarize config on the screen as a sanity check
    print(100*"=")
    print(f"Here is the configuration being used:\n")
    print_config(config)
    print(100*"=")
    print(f"Arguments being used:\n")
    print_config(args)
    print(100*"=")