# PyFlow_SimCLR: 
##### Author: Talip Ucar (ucabtuc@gmail.com)

Pytorch implementation of SimCLR (https://arxiv.org/pdf/2002.05709.pdf) with custom Encoder.

![SimCLR](./simclr.gif)

<sup>Source: https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html </sub>
# Model
A custom CNN-based encoder model is provided, and its architecture is defined in 
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

Resnet18 and Resnet50 models will be supported in the future.

# Datasets
Following datasets are supported:
1. STL10    (Image size=96)
2. CIFAR10  (Image size=32)
3. MNIST    (Image size=28)

For example, the encoder can be trained on STL10, and evaluated on both STL10 and CIFAR10.

# Environment - Installation
It requires Python 3.8. You can set up the environment by following three steps:
1. Install pipenv using pip
2. Activate virtual environment
3. Install required packages 

Run following commands in order to set up the environment:
```
pip install pipenv          # To install pipenv if you don't have it already
pipenv shell                # To activate virtual env
pipenv install              # To install required packages. 
```
<pre>
If the last step causes issues, you can try one of the following suggestions:
- removing lock file (if it exists) and re-do ```pipenv install```  
- removing lock file (if it exists) and use ```pipenv install --skip-lock```
- use ```pipenv lock --pre --clear``` 
- You can install packages defined in Pipfile by using pip i.e. "pip install package_name". 
</pre>

# Training
You can train the model using any supported dataset. For now, STL10 is recommended to use. The more datasets will be 
supported in the future.

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

# Results

Results at the end of training is saved under "./results" directory. Results directory structure:

<pre>
results
    |-evaluation 
    |-training 
         |-model
         |-plots
         |-loss
</pre>

You can save results of evaluations under "evaluation" folder.

# Running scripts
## Training
To pre-train the model using STL10 dataset, you can use following command:
```
python 0_train_encoder.py -d "STL10" -img 96
```
## Evaluation
Once you have a trained model, you can evaluate the model performance on any dataset. Correct image size should be provided
with corresponding dataset to get correct results since the model architecture is agnostic to image size and will not flag
error if correct image size is not specified.

Two examples: 
1. Evaluating on STL10
```
python 1_eval_linear_classifier.py -d "STL10" -img 96
```
2. Evaluating on CIFAR10
```
python 1_eval_linear_classifier.py -d "CIFAR10" -img 32
```

For further details on what arguments you can use (or to add your own arguments), you can check out "/utils/arguments.py"

# Experiment tracking
MLFlow is used to track experiments. It is turned off by default, but can be turned on by changing option in 
runtime config file in "./config/runtime.yaml"
