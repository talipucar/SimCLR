# PyFlow_SimCLR: 
##### Author: Talip Ucar (ucabtuc@gmail.com)

Pytorch implementation of SimCLR (https://arxiv.org/pdf/2002.05709.pdf) with custom Encoder. 
- If you are in hurry, and just want to clone the project and run it, move to [Summary](#summary) section. 
- If you are really, really in hurry, move to [Feeling Lazy](#feeling-lazy) section.  

Otherwise, read on...

# Table of Contents:

1. [Model](#model)
2. [Datasets](#datasets)
3. [Environment](#environment)
4. [Configuration](#configuration)
5. [Training](#Training)
6. [Evaluation](#Evaluation)
7. [Results](#results)
8. [Experiment tracking](#experiment-tracking)
9. [Tricks and Warnings](#tricks-and-warnings)
10. [Summary](#summary)
11. [Feeling Lazy](#feeling-lazy)
12. [Citing this repo](#citing-this-repo)

![SimCLR](./assets/simclr.gif)

<sup>Source: https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html </sub>
# Model
It supports 3 models:

1. A custom CNN-based encoder model is provided, and its architecture is defined in 
yaml file of the model ("./config/contrastive_encoder.yaml"). 

Example: 
<pre>
conv_dims:                        
  - [ 3,  32, 5, 2, 1, 1]         
  - [32,  64, 5, 2, 1, 1]         
  - [64, 128, 5, 2, 1, 1]         
  - 128  
</pre>

```conv_dims``` defines first 3 convolutional layer as well as dimension of the projection head. You can change this architecture
by modifying it in yaml file. You can add more layers, or change the dimensions of existing ones. 
Architecture is agnostic to input image size.

2. Resnet18 ("./config/resnet18.yaml")
3. Resnet50 ("./config/resnet50.yaml") 

You can define which one to use by defining **model_mode** in ("./config/runtime.yaml") e.g. "model_mode: contrastive_encoder", or "model_mode: resnet50"

# Datasets
Following datasets are supported:
1. STL10    (Image size=96)
2. CIFAR10  (Image size=32)
3. MNIST    (Image size=28)

For example, the encoder can be trained on STL10, and evaluated on both STL10 and CIFAR10.

# Environment
It is tested with Python 3.7, or 3.8. You can set up the environment by following three steps:
1. Install pipenv using pip
2. Activate virtual environment
3. Install required packages 

Run following commands in order to set up the environment:
```
pip install pipenv             # To install pipenv if you don't have it already
pipenv shell                   # To activate virtual env
pipenv install --skip-lock     # To install required packages. 
```

If the last step causes issues, you can install packages defined in Pipfile by using pip i.e. "pip install package_name" one by one. 

#### Important Note: 
If you want to use Python 3.7, follow these steps:
- Change python_version in Pipfile: ```python_version = "3.7"``` 
- Comment out torch and torchvision in Pipfile
- Install the packages as described above in 3 steps.
- Pip install torch and torchvision using following command line:
```pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html```

You need to use this version of torch and torchvision if you want to use GPU when training. Other versions of torch with Python 3.7 
is not able to detect GPU even if it is available. In general, just make sure that your torch modules are compatible with the particular cuda driver installed on your machine.

# Configuration
Under the "./config/" directory, there are two kinds of yaml files. 

1) runtime.yaml is a high-level config file, where you define which model to run: 

- contrastive_encoder (i.e. custom encoder), 
- resnet18, or 
- resnet50. 

This is also where you define whether you want to track experiments using MLFlow, random seed to use, whether to use distributed training (not implemented, just a placeholder for now), and paths for data and results.

2) Other yaml files (resnet18, resnet50, contrastive_encoder) are the model specific configurations. The name of these yaml files need to match to the model_mode defined in runtime.yaml so that the script can find and load the model specific configuration.

# Training
You can train the model using any supported dataset. For now, STL10 is recommended to use. The more datasets will be 
supported in the future.

To pre-train the model using STL10 dataset, you can use following command:
```
python 0_train.py 
```
This will train the model on STL10 by default. If you want to prefer on another dataset, you can simply do:
```
python 0_train.py -d "dataset_name" -img image_size
```
## Fine-tuning a pre-trained model
If you already have a pre-trained model, you can fine-tune it by using following command:
```
python 0_train.py -t True
```
To fine-tune on STL10. You can choose to define the dataset when fine-tuning:
```
python 0_train.py -d "dataset_name" -img image_size  -t True
```

# Evaluation
## Evaluation of trained SSL representations
1. Logistic regression model is trained, using representations extracted from Encoder using training set of
specified dataset.
2. The results are  reported on both training and test sets.

## Baseline Evaluation
1. Raw images of specified dataset is reshaped to a 2D array. 
2. PCA is used to reduce the dimensionality so that the feature
dimension is same as that of representations from Encoder (to be a fair comparison). 
3. Logistic regression model is trained, using data obtained using PCA, the results are reported on both training and
test sets.

## Running evaluation script
Once you have a trained model, you can evaluate the model performance on any dataset. Correct image size should be provided
with corresponding dataset to get correct results since the model architecture is agnostic to image size and will not flag
error if correct image size is not specified.

Two examples: 
1. Evaluating on STL10
```
python 1_eval.py -d "STL10" -img 96
```
2. Evaluating on CIFAR10
```
python 1_eval.py -d "CIFAR10" -img 32
```

For further details on what arguments you can use (or to add your own arguments), you can check out "/utils/arguments.py"


# Results

Results at the end of training is saved under "./results" directory. Results directory structure:

<pre>
results
    |-evaluation 
    |-training
        |-model_mode (e.g. resnet50)   
             |-model
             |-plots
             |-loss
</pre>

You can save results of evaluations under "evaluation" folder. 

Note that model_mode corresponds to the model defined as a yaml file. For example, contrastive_encoder.yaml is saved under /config, and pointed to in runtime.yaml file so that the script uses this particular architecture & hyper-parameters, and saves the results under a folder with the same name. You can write your own custom config files, and point to them in runtime.yaml

## Performance with default parameters
The model with custom encoder (i.e. no resnet backbone) is trained on STL10 dataset with default hyper-parameters (no optimization). 
Following results compare performances using extraxted features via PCA and via trained Model in linear classification task on 
STL10, and CIFAR10 (transfer learning) datasets.

**Performance of SimCLR with small, custom encoder with default hyper-parameters, trained on STL10:**
<pre>
                |             STL10            |             CIFAR10          |
                |     Train     |    Test      |     Train     |    Test      |
| ------------  | :-----------: | -----------: | :-----------: | -----------: |
|   SimCLR      |     0.69      |     0.65     |     0.64      |     0.61     |
|   Baseline    |     0.43      |     0.36     |     0.41      |     0.40     |
    
Performance of SimCLR with custom encoder (trained on STL10) on classification task on STL10 and CIFAR10
</pre>
 

**Performance of SimCLR with ResNet50 with default hyper-parameters, trained on STL10:**
- Training time:  779 minutes on a single GPU with batch size of 256 for 100 epochs.

<pre>
                |             STL10            |             CIFAR10          |
                |     Train     |    Test      |     Train     |    Test      |
| ------------  | :-----------: | -----------: | :-----------: | -----------: |
|   SimCLR      |     0.83      |     0.74     |     0.62      |     0.58     |
|   Baseline    |     0.43      |     0.36     |     0.41      |     0.40     |
    
Performance of SimCLR with ResNet50 (trained on STL10) on classification task on STL10 and CIFAR10
</pre>
 
**Note:** For baseline PCA performance, the dimension of projection was 128 (same as the projection head in the SimCLR models to keep the comparison fair). If we increased PCA dimension to 512, we would get:
<pre>
                |             STL10            |             CIFAR10          |
                |     Train     |    Test      |     Train     |    Test      |
| ------------  | :-----------: | -----------: | :-----------: | -----------: |
|   Baseline    |     0.64      |     0.32     |     0.44      |     0.40     |
</pre>


# Experiment tracking
MLFlow is used to track experiments. It is turned off by default, but can be turned on by changing option in 
runtime config file in "./config/runtime.yaml"


# Tricks and Warnings
SimCLR is not stable when trained with small batch size. However, the model converges much faster with smaller batch sizes. So, you can try to keep batch size small enough to converge faster, but big enough to keep the training stable. You can also try to dynamically change batch size during the training for fast convergence at the beginning of training, and stability for the rest of the training. Batch size of 512 might be a good trade-off when training STL10. For other datasets, you need to experiment with hyper-parameters to see what works the best.



# Summary
1) Data 
When you run training script, it automatically downloads and saves the data if it is not downloaded already.


2) Installation of required packages:
```
pip install pipenv          # To install pipenv if you don't have it already
pipenv shell                # To activate virtual env
pipenv install --skip-lock  # To install required packages. 
```

3) Training and evaluation of the models:
```
  I) python 0_train.py                              # Train autoencoder using default dataset STL10
 II) python 1_eval.py -d "CIFAR10" -img 32          # Evaluations using CIFAR10 dataset
```
If you want to train on another dataset, you can simply define it:
```
python 0_train.py -d "dataset_name" -img image_size
```

If you want to use Python 3.7, please follow the steps described in [Important Note](#important-note).


# Feeling Lazy
If you are really, really in hurry and don't want to do anything manually, just run "./feeling_lazy.sh" on the shell once you clone the repo, and get into the project folder. 
It assumes Python 3.7, so change "python_version" in the Pipfile if you have a different Python version before you run the script.)


# Citing this repo
If you use this work in your own studies, and work, you can cite it by using the following:
```
@Misc{talip_ucar_2021_simclr,
  author =   {Talip Ucar},
  title =    {{Pytorch implementation of SimCLR}},
  howpublished = {\url{https://github.com/talipucar/PyFlow_SimCLR}},
  month        = Jan,
  year = {since 2021}
}
```
