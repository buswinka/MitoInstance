import torch
from src.transforms import _crop
import torch.nn as nn


class jaccard_loss:
    def __call__(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Returns intersection over union

        :param predicted: [B, I, X, Y, Z] torch.Tensor of probabilities calculated from src.utils.embedding_to_probability
                          where B: is batch size, I: instances in image
        :param ground_truth: [B, I, X, Y, Z] segmentation mask for each instance (I).
        :return:
        """

        # print(predicted.shape, ground_truth.shape)
        # print(ground_truth.shape[3]-1, ground_truth.shape[4]-1)

        predicted = _crop(predicted, x=0, y=0, z=0,
                          w=ground_truth.shape[2], h=ground_truth.shape[3], d=ground_truth.shape[4])

        ground_truth = _crop(ground_truth, x=0, y=0, z=0,
                             w=predicted.shape[2], h=predicted.shape[3], d=predicted.shape[4])

        # intersection = (predicted * ground_truth).sum().mul(2)
        # union = (predicted + ground_truth).sum()

        # predicted[predicted>=0.5]=1
        # predicted[predicted<0.5]=0

        intersection = (predicted * ground_truth).sum().add(1e-10)
        union = (predicted + ground_truth).sum().sub(intersection).add(2e-10)

        assert not torch.isnan(intersection)
        assert not torch.isnan(union)

        return 1.0 - (intersection/union)


class dice(nn.Module):
    def __init__(self):
        super(dice, self).__init__()

    def __call__(self, pred: torch.Tensor, mask: torch.Tensor, **kwargs):
        """
        Calculates the dice loss between pred and mask

        :param pred: torch.Tensor | probability map of shape [B,C,X,Y,Z] predicted by hcat.unet
        :param mask: torch.Tensor | ground truth probability map of shape [B, C, X+dx, Y+dy, Z+dz] that will be cropped
                     to identical size of pred
        :return: torch.float | calculated dice loss
        """

        pred_shape = pred.shape
        n_dim = len(pred_shape)

        if n_dim == 5:
            mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
        elif n_dim == 4:
            mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
        else:
            raise IndexError(f'Unexpected number of predicted mask dimensions. Expected 4 (2D) or 5 (3D) but got' +
                             f' {n_dim} dimensions: {pred_shape}')

        # pred = torch.sigmoid(pred)
        loss = (2 * (pred * mask).sum() + 1e-10) / ((pred + mask).sum() + 1e-10)

        return 1-loss


class tversky_loss(nn.Module):
    def __init__(self):
        super(tversky_loss, self).__init__()

    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor, smooth: float = 1e-10,
                alpha: float = 0.1, beta: float = 0.9):
        """


        :param predicted:
        :param ground_truth:
        :param smooth:
        :param alpha: FALSE POSITIVE
        :param beta: FALSE NEGATIVE
        :return:
        """

        predicted = _crop(predicted, x=0, y=0, z=0,
                          w=ground_truth.shape[2], h=ground_truth.shape[3], d=ground_truth.shape[4])

        ground_truth = _crop(ground_truth, x=0, y=0, z=0,
                             w=predicted.shape[2], h=predicted.shape[3], d=predicted.shape[4])

        #-------------------------------------------------#
        predicted = predicted * ground_truth  # EXPERIMENTAL
        #-------------------------------------------------#

        true_positive = (predicted * ground_truth).sum()
        false_negative = ((1 - predicted) * ground_truth).sum()
        false_positive = (torch.logical_not(ground_truth) * predicted).sum().add(1e-10)

        tversky = (true_positive + smooth) / (true_positive + (alpha * false_positive) + (beta * false_negative) + smooth)

        return 1 - tversky


class l1_loss(nn.Module):
    def __call__(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Returns intersection over union

        :param predicted: [B, I, X, Y, Z] torch.Tensor of probabilities calculated from src.utils.embedding_to_probability
                          where B: is batch size, I: instances in image
        :param ground_truth: [B, I, X, Y, Z] segmentation mask for each instance (I).
        :return:
        """

        predicted = _crop(predicted, x=0, y=0, z=0,
                          w=ground_truth.shape[2], h=ground_truth.shape[3], d=ground_truth.shape[4])

        ground_truth = _crop(ground_truth, x=0, y=0, z=0,
                             w=predicted.shape[2], h=predicted.shape[3], d=predicted.shape[4])



        loss = torch.nn.L1Loss()

        return loss(predicted, ground_truth)


if __name__ == '__main__':

    a = torch.rand((1,50,200,200,30))
    b = torch.rand((1, 50, 200, 200, 30))
    loss = tversky_loss()
    print(loss(a, b > 0.5))
    print(loss(a, a > 0.5))
    print(loss(a > 0.5, a > 0.5))

