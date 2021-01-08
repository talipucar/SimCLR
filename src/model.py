"""
Class to train contrastive encoder in Self-Supervised setting.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import os
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as th
import itertools
from utils.utils import set_seed
from utils.loss_functions import  NTXentLoss
from utils.model_plot import save_loss_plot
from utils.model_utils import CNNEncoder, Classifier
th.autograd.set_detect_anomaly(True)


class ContrastiveEncoder:
    """
    Model: CNN-based encoder
    Loss function: NTXentLoss - https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, options):
        """
        :param dict options: Configuration dictionary.
        """
        # Get config
        self.options = options
        # Define which device to use: GPU, or CPU
        self.device = options["device"]
        # Create empty lists and dictionary
        self.model_dict, self.summary = {}, {}
        # Set random seed
        set_seed(self.options)
        # ------Network---------
        # Instantiate networks
        print("Building models...")
        # Set contrastive_encoder i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        self.set_contrastive_encoder()
        # Instantiate and set up Classifier if "supervised" i.e. loss, optimizer, device (GPU, or CPU)
        self.set_classifier() if self.options["supervised"] else None
        # Set scheduler (its use is optional)
        self.set_scheduler()
        # Set paths for results and Initialize some arrays to collect data during training
        self.set_paths()
        # Print out model architecture
        self.get_model_summary()

    def set_contrastive_encoder(self):
        # Instantiate the model
        self.contrastive_encoder = CNNEncoder(self.options)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"contrastive_encoder": self.contrastive_encoder})
        # Assign contrastive_encoder to a device
        self.contrastive_encoder.to(self.device)
        # Reconstruction loss
        self.contrastive_loss = NTXentLoss(self.options)
        # Set optimizer for contrastive_encoder
        self.optimizer_ce = self._adam([self.contrastive_encoder.parameters()], lr=self.options["learning_rate"])
        # Add items to summary to be used for reporting later
        self.summary.update({"contrastive_loss": [], "kl_loss": []})

    def set_classifier(self):
        # Instantiate Classifier
        self.classifier = Classifier(self.options)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"classifier": self.classifier})
        # Assign classifier to a device
        self.classifier.to(self.device)
        # Cross-entropy loss
        self.xloss = th.nn.CrossEntropyLoss(reduction='mean')
        # Set optimizer for classifier
        self.optimizer_cl = self._adam([self.classifier.parameters()], lr=self.options["learning_rate"])
        # Add items to summary to be used for reporting later
        self.summary.update({"class_train_acc": [], "class_test_acc": []})

    def fit(self, data_loader):
        """
        :param IterableDataset data_loader: Pytorch data loader.
        :return: None

        Fits model to the data using contrastive learning.
        """
        # Training dataset
        train_loader = data_loader.train_loader
        # Validation dataset. Note that it uses only one batch of data to check validation loss to save from computation.
        Xval = self.get_validation_batch(data_loader)
        # Loss dictionary: "ntx": NTXentLoss, "c": classification, "v": validation -- Suffixes: "_b": batch, "_e": epoch
        self.loss = {"ntx_loss_b": [], "ntx_loss_e": [], "closs_b": [], "closs_e": [], "vloss_e": []}
        # Turn on training mode for each model.
        self.set_mode(mode="training")
        # Compute batch size
        bs = self.options["batch_size"]
        # Compute total number of batches per epoch
        self.total_batches = len(train_loader)
        print(f"Total number of samples / batches in training set: {len(train_loader.dataset)} / {len(train_loader)}")
        # Start joint training of contrastive_encoder, and/or classifier
        for epoch in range(self.options["epochs"]):
            # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
            self.train_tqdm = tqdm(enumerate(train_loader), total=self.total_batches, leave=True)
            # Go through batches
            for i, ((xi, xj), _) in self.train_tqdm:
                # Concatenate xi, and xj, and turn it into a tensor
                Xbatch = self.process_batch(xi, xj)
                # Forward pass on contrastive_encoder
                z, _ = self.contrastive_encoder(Xbatch)
                # Compute reconstruction loss
                contrastive_loss = self.contrastive_loss(z)
                # Get contrastive loss for training per batch
                self.loss["ntx_loss_b"].append(contrastive_loss.item())
                # Update contrastive_encoder params
                self.update_model(contrastive_loss, self.optimizer_ce, retain_graph=True)
                # Clean-up for efficient memory use.
                del contrastive_loss
                gc.collect()
                # Update log message using epoch and batch numbers
                self.update_log(epoch, i)
            # Get reconstruction loss for training per epoch
            self.loss["ntx_loss_e"].append(sum(self.loss["ntx_loss_b"][-self.total_batches:-1]) / self.total_batches)
            # Validate every nth epoch. n=1 by default
            _ = self.validate(Xval) if epoch % self.options["nth_epoch"] == 0 else None
        # Save plot of training and validation losses
        save_loss_plot(self.loss, self._plots_path)
        # Convert loss dictionary to a dataframe
        loss_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.loss.items()]))
        # Save loss dataframe as csv file for later use
        loss_df.to_csv(self._loss_path + "/losses.csv")

    def update_log(self, epoch, batch):
        # For the first epoch, add losses for batches since we still don't have loss for the epoch
        if epoch < 1:
            description = f"Epoch:[{epoch-1}], Batch:[{batch}] loss:{self.loss['ntx_loss_b'][-1]:.4f}"
        # For sub-sequent epochs, display only epoch losses.
        else:
            description = f"Epoch:[{epoch-1}] loss:{self.loss['ntx_loss_e'][-1]:.4f}, val loss:{self.loss['vloss_e'][-1]:.4f}"
        # Update the displayed message
        self.train_tqdm.set_description(description)

    def set_mode(self, mode="training"):
        # Change the mode of models, depending on whether we are training them, or using them for evaluation.
        if mode == "training":
            self.contrastive_encoder.train()
            self.classifier.train() if self.options["supervised"] else None
        else:
            self.contrastive_encoder.eval()
            self.classifier.eval() if self.options["supervised"] else None

    def process_batch(self, xi, xj):
        # Combine xi and xj into a single batch
        Xbatch = np.concatenate((xi, xj), axis=0)
        # Convert the batch to tensor and move it to where the model is
        Xbatch = self._tensor(Xbatch)
        # Return batches
        return Xbatch

    def get_validation_batch(self, data_loader):
        # Validation dataset
        validation_loader = data_loader.test_loader
        # Use only the first batch of validation set to save from computation
        ((xi, xj), _) = next(iter(validation_loader))
        # Concatenate xi, and xj, and turn it into a tensor
        Xval = self.process_batch(xi, xj)
        # Return
        return Xval

    def validate(self, Xval):
        with th.no_grad():
            # Turn on evaluatin mode
            self.set_mode(mode="evaluation")
            # Define loss
            loss = NTXentLoss(self.options)
            # Forward pass on contrastive_encoder
            z, _ = self.contrastive_encoder(Xval)
            # Compute reconstruction loss
            contrastive_loss = loss(z)
            # Get contrastive loss for training per batch
            self.loss["vloss_e"].append(contrastive_loss.item())
            # Turn on training mode
            self.set_mode(mode="training")
            # Clean up to avoid memory issues
            del contrastive_loss, z, Xval
            gc.collect()

    def save_weights(self):
        """
        :return: None
        Used to save weights of contrastive_encoder, and (if options['supervision'] == 'supervised) Classifier
        """
        for model_name in self.model_dict:
            th.save(self.model_dict[model_name], self._model_path + "/" + model_name + ".pt")
        print("Done with saving models.")

    def load_models(self):
        """
        :return: None
        Used to load weights saved at the end of the training.
        """
        for model_name in self.model_dict:
            model = th.load(self._model_path + "/" + model_name + ".pt")
            setattr(self, model_name, model.eval())
            print(f"--{model_name} is loaded")
        print("Done with loading models.")

    def get_model_summary(self):
        """
        :return: None
        Sanity check to see if the models are constructed correctly.
        """
        # Summary of contrastive_encoder
        description  = f"{40*'-'}Summarize models:{40*'-'}\n"
        description += f"{34*'='}{self.options['model_mode'].upper().replace('_', ' ')} Model{34*'='}\n"
        description += f"{self.contrastive_encoder}\n"
        # Summary of Classifier if it is being used
        if self.options["supervised"]:
            description += f"{30*'='} Classifier {30*'='}\n"
            description += f"{self.classifier}\n"
        # Print model architecture
        print(description)

    def update_model(self, loss, optimizer, retain_graph=True):
        # Reset optimizer
        optimizer.zero_grad()
        # Backward propagation to compute gradients
        loss.backward(retain_graph=retain_graph)
        # Update weights
        optimizer.step()

    def set_scheduler(self):
        # Set scheduler (Its use will be optional)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer_ce, step_size=2, gamma=0.99)

    def set_paths(self):
        """ Sets paths to bse used for saving results at the end of the training"""
        # Top results directory
        self._results_path = self.options["paths"]["results"]
        # Directory to save model
        self._model_path = os.path.join(self._results_path, "training", self.options["model_mode"], "model")
        # Directory to save plots as png files
        self._plots_path = os.path.join(self._results_path, "training", self.options["model_mode"], "plots")
        # Directory to save losses as csv file
        self._loss_path = os.path.join(self._results_path, "training", self.options["model_mode"], "loss")

    def _adam(self, params, lr=1e-4):
        return th.optim.Adam(itertools.chain(*params), lr=lr, betas=(0.9, 0.999))

    def _tensor(self, data):
        return th.from_numpy(data).to(self.device).float()
