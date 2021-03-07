#!/bin/bash

pip install pipenv
pipenv shell
pipenv install --skip-lock
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pipenv run python 0_train.py 
