"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Utility functions
"""

import os
import sys
import yaml
import joblib
import numpy as np
from numpy.random import seed
import random as python_random
import pandas as pd
from sklearn.utils import shuffle
from texttable import Texttable


def set_seed(options):
    seed(options["seed"])
    np.random.seed(options["seed"])
    python_random.seed(options["seed"])

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def set_dirs(config):
    """
    It sets up directory that will be used to load processed_data and src as well as saving results.
    Directory structure:
          results -> processed_data: contains processed k-fold processed_data file.
                  -> src: contains saved src, trained on this database
                  -> results-> training_plots
    :return: None
    """
    # Update the config file with model config and flatten runtime config
    config = update_config_with_model(config)
    # Set main results directory using database name. Exp:  processed_data/dpp19
    paths = config["paths"]
    # data > processed_data
    processed_data_dir = os.path.join(paths["data"], "processed_data")
    # results > training
    training_dir = os.path.join(paths["results"], "training")
    # results > evaluation
    evaluation_dir = os.path.join(paths["results"], "evaluation")
    # results > training > model_mode = vae
    model_mode_dir = os.path.join(training_dir, config["model_mode"])
    # results > training > model_mode > model
    training_model_dir = os.path.join(model_mode_dir, "model")
    # results > training > model_mode > plots
    training_plot_dir = os.path.join(model_mode_dir, "plots")
    # results > training > model_mode > loss
    training_loss_dir = os.path.join(model_mode_dir, "loss")
    # Create any missing directories
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    if not os.path.exists(training_model_dir):
        os.makedirs(training_model_dir)
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    if not os.path.exists(model_mode_dir):
        os.makedirs(model_mode_dir)
    if not os.path.exists(training_model_dir):
        os.makedirs(training_model_dir)
    if not os.path.exists(training_plot_dir):
        os.makedirs(training_plot_dir)
    if not os.path.exists(training_loss_dir):
        os.makedirs(training_loss_dir)
    # Print a message.
    print("Directories are set.")


def get_runtime_and_model_config():
    try:
        with open("./config/runtime.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading runtime config file")
    # Update the config by adding the model specific config to runtime config
    config = update_config_with_model(config)
    return config


def update_config_with_model(config):
    # Get model_mode from runtime.yaml
    model_config = config["model_mode"]
    # Load model specific configuration
    try:
        with open("./config/"+model_config+".yaml", "r") as file:
            model_config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading model config file")
    # Update the runtime configuration with the model specific configuration
    config.update(model_config)
    return config


def update_config_with_model_dims(data_loader, config):
    ((xi, xj), _) = next(iter(data_loader))
    # Get the number of features
    dim = xi.shape[-1]
    # Update the dims of model architecture by adding the number of features as the first dimension
    config["dims"].insert(0, dim)
    return config

def print_config(args):
    """
    Prints the YAML config and ArgumentParser arguments in a neat format.
    :param args: Parameters/config used for the model.
    """
    # Yaml config is a dictionary while parser arguments is an object. Use vars() only on parser arguments.
    if type(args) is not dict:
        args = vars(args)
    # Sort keys
    keys = sorted(args.keys())
    # Initialize table
    table = Texttable() 
    # Add rows to the table under two columns ("Parameter", "Value").
    table.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    # Print the table.
    print(table.draw())