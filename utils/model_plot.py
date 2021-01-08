"""
Class to train contrastive encoder in Self-Supervised setting.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import matplotlib.pyplot as plt


def save_loss_plot(losses, plots_path):
    x_axis = list(range(len(losses["ntx_loss_e"])))
    plt.plot(x_axis, losses["ntx_loss_e"], c='r')
    title = "Training"
    if len(losses["ntx_loss_e"]) == len(losses["vloss_e"]):
        plt.plot(x_axis, losses["vloss_e"], c='b')
        title += " and Validation "
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title + " Loss", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_path + "/loss.png")


