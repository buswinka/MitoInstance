import torch
from torch.utils.data import DataLoader
import numpy as np
import glob
import os.path
import skimage.io as io
from typing import Dict, Callable


class dataset(DataLoader):
    def __init__(self, path: str = None, transforms: Callable = None, random_ind: bool = False):
        """
        Loads data into a dataset structure for training deep learning architecture.

        :param path: str - Path to data
        :param transforms: Callable Function of transforms
        :param random_ind: Bool - If True, shuffles data every time self.step() is called
        """
        super(DataLoader, self).__init__()

        # Find only files with a label
        files = glob.glob(os.path.join(path, '*.labels.tif'))

        self.mask = []
        self.image = []
        self.centroids = []
        self.transforms = transforms

        # Load in all files and convert to float
        for f in files:
            image_path = os.path.splitext(f)[0]
            image_path = os.path.splitext(image_path)[0] + '.tif'
            image = torch.from_numpy(io.imread(image_path).astype(np.uint16) / 2 ** 16).unsqueeze(-1)

            image = image.transpose(1, 3).transpose(0, -1).squeeze().unsqueeze(0)
            mask = torch.from_numpy(io.imread(f)).transpose(0, 2).unsqueeze(0)                           # Import Mask

            self.mask.append(mask.float())
            self.image.append(image.float())
            self.centroids.append(torch.tensor([0]))

        # implement random permutations of the indexing
        self.random_ind = random_ind

        if self.random_ind:
            self.index = torch.randperm(len(self.mask))
        else:
            self.index = torch.arange((len(self.mask)))

    def __len__(self) -> int:
        """
        :return: int - number of training pairs
        """
        return len(self.mask)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves training dictionary

        :param item: int - index of image
        :return:  Dict[str, torch.Tensor] {'image': torch.Tensor, 'masks': torch.Tensor, 'centroids': torch.Tensor}
                        image:         [C, X, Y, Z]
                        masks:         [N, X, Y, Z]
                        centroids:     [N, X, Y, Z]
        """

        item = self.index[item]  # Get a random index here

        data_dict = {'image': self.image[item], 'masks': self.mask[item], 'centroids': self.centroids[item]}
        did_we_get_an_output = False

        while not did_we_get_an_output:
            try:
                if self.transforms is not None:
                    data_dict = self.transforms(data_dict)
                    did_we_get_an_output = True
            except RuntimeError:
                continue

        return data_dict

    def step(self) -> None:
        if self.random_ind:
            self.index = torch.randperm(len(self.mask))


@torch.jit.script
def colormask_to_torch_mask(colormask: torch.Tensor) -> torch.Tensor:
    """

    :param colormask: [C=1, X, Y, Z]
    :return:
    """
    uni = torch.unique(colormask)
    uni = uni[uni != 0]
    num_cells = len(uni)

    shape = (num_cells, colormask.shape[1], colormask.shape[2], colormask.shape[3])
    mask = torch.zeros(shape)

    for i, u in enumerate(uni):
        mask[i, :, :, :] = (colormask[0, :, :, :] == u)

    return mask


@torch.jit.script
def colormask_to_centroids(colormask: torch.Tensor) -> torch.Tensor:
    uni = torch.unique(colormask)
    uni = uni[uni != 0]
    num_cells = len(uni)  # cells are denoted by integers 1->max_cell
    shape = (num_cells, 3)
    centroid = torch.zeros(shape)

    for i, u in enumerate(uni):
        indexes = torch.nonzero(colormask[0, :, :, :] == u).float()
        centroid[i, :] = torch.mean(indexes, dim=0)

    # centroid[:, 0] /= colormask.shape[1]
    # centroid[:, 1] /= colormask.shape[2]
    # centroid[:, 2] /= colormask.shape[3]

    return centroid
