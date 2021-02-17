import torch
from src.transforms import _crop
from src.functional import embedding_to_probability
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Callable
import numpy as np



class tversky_loss(nn.Module):
    def __init__(self):
        super(tversky_loss, self).__init__()

        self.false_positive = 0
        self.false_negative = 0
        self.true_positive = 0
        self.background = 0
        self.background_total = 0

        self.pred_shape = []
        self.ground_truth_shape = []

    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor, smooth: float = 1e-10,
                alpha: float = 0.1, beta: float = 0.9):

        self.pred_shape = predicted.shape
        self.ground_truth_shape = ground_truth.shape

        predicted = _crop(predicted, x=0, y=0, z=0,
                          w=ground_truth.shape[2], h=ground_truth.shape[3], d=ground_truth.shape[4])

        ground_truth = _crop(ground_truth, x=0, y=0, z=0,
                             w=predicted.shape[2], h=predicted.shape[3], d=predicted.shape[4])

        self.background = ((1-ground_truth.float())*predicted).sum()
        self.background_total = (1-ground_truth.float().sum())

        # -------------------------------------------------#
        # predicted = predicted * ground_truth  # EXPERIMENTAL
        # -------------------------------------------------#

        self.true_positive = (predicted * ground_truth).sum()
        self.false_negative = ((1 - predicted) * ground_truth).sum()
        self.false_positive = (torch.logical_not(ground_truth) * predicted).sum().add(1e-10)

        tversky = (self.true_positive + smooth) / (
                    self.true_positive + (alpha * self.false_positive) + (beta * self.false_negative) + smooth)
        return 1 - tversky

    def diagnostics(self, verbose: bool = False):

        total = self.true_positive + self.false_positive + self.false_negative
        if verbose:
            print('--------------')
            print(f'   Background: {self.background} - {self.background / self.background_total}%')
            print(f'   True Positive: {self.true_positive} - {self.true_positive / total}%')
            print(f'   False Positive: {self.false_positive} - {self.false_positive / total}%')
            print(f'   False Negative: {self.false_negative} - {self.false_negative / total}%')
            print(f'   Ground Truth Shape: {self.ground_truth_shape}')
            print(f'   Predicted Shape: {self.pred_shape}')
        return self.background.item(), self.true_positive.item(), self.false_positive.item(), self.false_negative.item()


def sigma_sweep(embedding: torch.Tensor = None, centroids: torch.Tensor = None,
                ground_truth: torch.Tensor = None,
                loss_fun: Callable = tversky_loss()) -> None:

    sigma = torch.logspace(-5, 0, 1000)
    loss = []
    background = []
    true_positive = []
    false_positive = []
    false_negative = []
    for s in tqdm(sigma):
        prob = embedding_to_probability(embedding, centroids, torch.tensor([s], device=embedding.device))
        loss.append(loss_fun(prob, ground_truth).item())
        b, tp, fp, fn = loss_fun.diagnostics()
        background.append(b)
        true_positive.append(tp)
        false_positive.append(fp)
        false_negative.append(fn)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.semilogx(sigma.numpy(), np.array(loss))
    ax1.set_title('Loss')
    ax2.semilogx(sigma.numpy(), np.array(true_positive))
    ax2.set_title('True Positive')
    ax3.semilogx(sigma.numpy(), np.array(false_negative))
    ax3.set_title('False Negative')

    ax4.semilogx(sigma.numpy(), np.array(true_positive))
    ax4.semilogx(sigma.numpy(), np.array(false_negative))
    ax4.set_title('Sum')
    plt.tight_layout()
    plt.show()



