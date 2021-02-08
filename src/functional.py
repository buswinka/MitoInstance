from sklearn.cluster import DBSCAN, OPTICS
from hdbscan import HDBSCAN
import torch
import numpy as np
import matplotlib.pyplot as plt
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
def calculate_vector(mask: torch.Tensor) -> torch.Tensor:
    """
    Have to use a fixed deltas for each pixel
    else it is size variant and doenst work for big images...

    :param mask:
    :return: [1,1,x,y,z] vector to the center of mask
    """
    x_factor = 1 / 512
    y_factor = 1 / 512
    z_factor = 1 / 512

    # com = torch.zeros(mask.shape)
    vector = torch.zeros((1, 3, mask.shape[2], mask.shape[3], mask.shape[4]))
    xv, yv, zv = torch.meshgrid([torch.linspace(0, x_factor * mask.shape[2], mask.shape[2]),
                                 torch.linspace(0, y_factor * mask.shape[3], mask.shape[3]),
                                 torch.linspace(0, z_factor * mask.shape[4], mask.shape[4])])

    for u in torch.unique(mask):
        if u == 0:
            continue
        index = ((mask == u).nonzero()).float().mean(dim=0)

        # Set between 0 and 1
        index[2] = index[2] / mask.shape[2]
        index[3] = index[3] / mask.shape[3]
        index[4] = index[4] / mask.shape[4]

        vector[0, 0, :, :, :][mask[0, 0, :, :, :] == u] = -xv[mask[0, 0, :, :, :] == u] + index[2]
        vector[0, 1, :, :, :][mask[0, 0, :, :, :] == u] = -yv[mask[0, 0, :, :, :] == u] + index[3]
        vector[0, 2, :, :, :][mask[0, 0, :, :, :] == u] = -zv[mask[0, 0, :, :, :] == u] + index[4]

    return vector


@torch.jit.script
def vector_to_embedding(vector: torch.Tensor) -> torch.Tensor:
    """
    Constructs a mesh grid and adds the vector matrix to it

    :param vector:
    :return:
    """
    x_factor = 1 / 512
    y_factor = 1 / 512
    z_factor = 1 / 512

    xv, yv, zv = torch.meshgrid([torch.linspace(0, x_factor * vector.shape[2], vector.shape[2]),
                                 torch.linspace(0, y_factor * vector.shape[3], vector.shape[3]),
                                 torch.linspace(0, z_factor * vector.shape[4], vector.shape[4])])

    mesh = torch.cat((xv.unsqueeze(0).unsqueeze(0),
                      yv.unsqueeze(0).unsqueeze(0),
                      zv.unsqueeze(0).unsqueeze(0)), dim=1).to(vector.device)

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

    sigma = sigma + 1e-10  # when sigma goes to zero, shit hits the fan

    centroids /= 512

    # Calculates the euclidean distance between the centroid and the embedding
    # embedding [B, 3, X, Y, Z] -> euclidean_norm[B, 1, X, Y, Z]
    # euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)

    prob = torch.zeros((embedding.shape[0],
                        centroids.shape[1],
                        embedding.shape[2],
                        embedding.shape[3],
                        embedding.shape[4])).to(embedding.device)

    sigma = (2 * sigma.to(embedding.device) ** 2)

    for i in range(centroids.shape[1]):
        # Calculate euclidean distance between centroid and embedding for each pixel
        euclidean_norm = (embedding - centroids[:, i, :].reshape(centroids.shape[0], 3, 1, 1, 1)).pow(2)

        # Turn distance to probability and put it in preallocated matrix
        prob[:, i, :, :, :] = torch.exp((euclidean_norm / sigma).mul(-1).sum(dim=1)).squeeze(1)

    return prob


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

    ind_x = torch.logical_and(x > 0, x < embed_shape[2]/512)
    ind_y = torch.logical_and(y > 0, y < embed_shape[3]/512)
    ind_z = torch.logical_and(z > 0, z < embed_shape[4]/512)
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

    x = x.mul(512).round().numpy()
    y = y.mul(512).round().numpy()
    z = z.mul(512).round().numpy()


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
    wh = torch.ones(centroids_xy.shape) * 12 # <- I dont know why this works but it does so deal with it????
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

