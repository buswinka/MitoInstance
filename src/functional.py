from sklearn.cluster import DBSCAN, OPTICS
from hdbscan import HDBSCAN
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
import skimage
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.feature
import skimage.segmentation
import skimage.transform
import skimage.feature

import scipy.ndimage
import scipy.ndimage.morphology
from scipy.interpolate import splprep, splev
import GPy

import torchvision.ops


@torch.jit.script
def vector_to_embedding(vector: torch.Tensor) -> torch.Tensor:
    """
    Constructs a mesh grid and adds the vector matrix to it

    :param vector:
    :return:
    """
    num = 128
    x_factor = 1 / num  # has to be a fixed size!
    y_factor = 1 / num
    z_factor = 1 / num

    xv, yv, zv = torch.meshgrid([torch.linspace(0, x_factor * vector.shape[2], vector.shape[2], device=vector.device),
                                 torch.linspace(0, y_factor * vector.shape[3], vector.shape[3], device=vector.device),
                                 torch.linspace(0, z_factor * vector.shape[4], vector.shape[4], device=vector.device)])

    mesh = torch.cat((xv.unsqueeze(0).unsqueeze(0),
                      yv.unsqueeze(0).unsqueeze(0),
                      zv.unsqueeze(0).unsqueeze(0)), dim=1)

    return mesh + vector


@torch.jit.script
def embedding_to_probability(embedding: torch.Tensor, centroids: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Vectorizing this is slower than the loop!!!

    #                     /    (e_ix - C_kx)^2       (e_iy - C_ky)^2        (e_iz - C_kz)^2   \
    #  prob_k(e_i) = exp |-1 * ----------------  -  -----------------   -  ------------------  |
    #                    \     2*sigma_kx ^2         2*sigma_ky ^2          2 * sigma_kz ^2  /

    :param embedding: [B, K=3, X, Y, Z] torch.Tensor where K is the likely centroid component: {X, Y, Z}
    :param centroids: [B, I, K_true=3] torch.Tensor where I is the number of instances in the image and K_true is centroid
                        {x, y, z}
    :param sigma: torch.Tensor of shape = (1) or (embedding.shape)
    :return: [B, I, X, Y, Z] of probabilities for instance I
    """

    sigma = sigma + 1e-10  # when sigma goes to zero, things tend to break

    centroids = centroids / 128  # Half the size of the image so vectors can be +- 1

    # Calculates the euclidean distance between the centroid and the embedding
    # embedding [B, 3, X, Y, Z] -> euclidean_norm[B, 1, X, Y, Z]
    # euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)

    prob = torch.zeros((embedding.shape[0],
                        centroids.shape[1],
                        embedding.shape[2],
                        embedding.shape[3],
                        embedding.shape[4]), device=embedding.device)

    sigma = sigma.pow(2).mul(2)

    for i in range(centroids.shape[1]):

        # Calculate euclidean distance between centroid and embedding for each pixel
        euclidean_norm = (embedding - centroids[:, i, :].view(centroids.shape[0], 3, 1, 1, 1)).pow(2)

        # Turn distance to probability and put it in preallocated matrix
        if sigma.shape[0] == 3:
            prob[:, i, :, :, :] = torch.exp(
                (euclidean_norm / sigma.view(centroids.shape[0], 3, 1, 1, 1)).mul(-1).sum(dim=1)).squeeze(1)
        else:
            prob[:, i, :, :, :] = torch.exp((euclidean_norm / sigma).mul(-1).sum(dim=1)).squeeze(1)

    return prob


def estimate_maximum_sigma(data: Iterable = None) -> torch.Tensor:
    """
    Hackey code to give you an estimate on sigma for all objects in your scene

    :param data:
    :return:
    """
    for dd in data:
        mask_shape = dd['masks'].shape[1]  # number of objects
        for m in range(mask_shape):
            mask = dd['masks'][0, m, ...]
            ind = torch.nonzero(mask).float()

            try:
                max = torch.cat((max, torch.max(ind, dim=0)[0].unsqueeze(0)))
                min = torch.cat((min, torch.min(ind, dim=0)[0].unsqueeze(0)))
            except NameError:
                max = torch.max(ind, dim=0)[0].unsqueeze(0)
                min = torch.min(ind, dim=0)[0].unsqueeze(0)

    dim = torch.mean(max - min, dim=0)
    print(dim)
    dim = dim / (512 * torch.sqrt(-2 * torch.log(torch.tensor([0.5]))))
    return dim


def estimate_centroids(embedding: torch.Tensor, eps: float = 0.2, min_samples: int = 100,
                       p: float = 2.0, leaf_size: int = 30) -> torch.Tensor:
    """
    Assume [B, 3, X, Y, Z]
    Warning moves everything to cpu!

    :param embedding:
    :param eps:
    :param min_samples:
    :param p:
    :param leaf_size:
    :return:
    """

    device = embedding.device
    embed_shape = embedding.shape
    embedding = embedding.detach().cpu().squeeze(0).reshape((3, -1))

    x = embedding[0, :]
    y = embedding[1, :]
    z = embedding[2, :]

    scale = 128

    ind_x = torch.logical_and(x > 0, x < embed_shape[2] / scale)
    ind_y = torch.logical_and(y > 0, y < embed_shape[3] / scale)
    ind_z = torch.logical_and(z > 0, z < embed_shape[4] / scale)
    ind = torch.logical_and(ind_x, ind_y)
    ind = torch.logical_and(ind, ind_z)

    x = x[ind]
    y = y[ind]
    z = z[ind]

    # ind = torch.randperm(len(x))
    # n_samples = 500000
    # ind = ind[0:n_samples:1]

    x = x[0:-1:5]
    y = y[0:-1:5]
    z = z[0:-1:5]

    # x = x[ind].mul(512).round().numpy()
    # y = y[ind].mul(512).round().numpy()
    # z = z[ind].mul(512).round().numpy()

    x = x.mul(scale).round().numpy()
    y = y.mul(scale).round().numpy()
    z = z.mul(scale).round().numpy()

    X = np.stack((x, y, z)).T
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1, p=p, leaf_size=leaf_size).fit(X)
    # db = HDBSCAN(min_samples=min_samples).fit(X)
    labels = db.labels_

    unique, counts = np.unique(labels, return_counts=True)

    centroids = []
    scores = []

    for u, s in zip(unique, counts):
        if u == -1:
            continue

        index = labels == u
        c = X[index, :].mean(axis=0)
        centroids.append(c)
        scores.append(s)

    if len(centroids) == 0:
        centroids = torch.empty((1, 0, 3)).to(device)
    else:
        centroids = torch.tensor(centroids).to(device).unsqueeze(0)
        # centroids[:, :, 0] *= 10
        # centroids[:, :, 1] *= 10
        # centroids[:, :, 2] *= 10

    # Works best with non maximum supression
    centroids_xy = centroids[:, :, [0, 1]]
    wh = torch.ones(centroids_xy.shape) * 12  # <- I dont know why this works but it does so deal with it????
    boxes = torch.cat((centroids_xy, wh.to(centroids_xy.device)), dim=-1)
    boxes = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy')
    keep = torchvision.ops.nms(boxes.squeeze(0), torch.tensor(scores).float().to(centroids_xy.device), 0.075)

    return centroids[:, keep, :]


if __name__ == '__main__':
    dd = torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/embed_data.trch')
    embed = dd['embed']
    centroids = dd['centroids']
    cent = estimate_centroids(embed, 0.003, 20)  # 0.0081, 160
    x = embed.detach().cpu().numpy()[0, 0, ...].flatten()
    y = embed.detach().cpu().numpy()[0, 1, ...].flatten()
    plt.figure(figsize=(10, 10))
    plt.hist2d(x, y, bins=256, range=((0, 1), (0, 1)))
    plt.plot(cent[0, :, 0].div(512).detach().cpu().numpy(), cent[0, :, 1].div(512).detach().cpu().numpy(), 'ro')
    plt.plot(centroids[0, :, 0].cpu() / 256, centroids[0, :, 1].cpu() / 256, 'bo')
    plt.show()
