import torch
import torchvision.transforms.functional
from PIL.Image import Image
import numpy as np
from typing import Dict, Tuple, Union, Sequence
import elasticdeform

import skimage.io as io


# -------------------------------- Assumptions ------------------------------#
#                Every image is expected to be [C, X, Y, Z]                  #
#        Every transform's input has to be Dict[str, torch.Tensor]           #
#       Every transform's output has to be Dict[str, torch.Tensor]           #
#       Preserve whatever device the tensor was on when it started           #
# ---------------------------------------------------------------------------#

class save_image:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = data_dict['image'][[2, 1, 0], ...].cpu().transpose(0, -1).numpy()[10, ...]
        _, mask = torch.max(data_dict['masks'].cpu(), 0)

        print(mask.max())

        mask = mask.float().numpy().transpose((2, 0, 1))[10, ...]

        io.imsave(self.name + '_image.png', image)
        io.imsave(self.name + '_mask.png', mask)

        return data_dict


class nul_crop:
    def __init__(self, rate: float = 0.80) -> None:
        self.rate = rate

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Removes blank space around cells to ensure training images has something the network can learn
        Doesnt remove Z


        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        if torch.rand(1) < self.rate:
            ind = torch.nonzero(data_dict['masks'])  # -> [I, 4] where 4 is ndims

            x_max = ind[:, 1].max().int().item()
            y_max = ind[:, 2].max().int().item()
            z_max = ind[:, 3].max().int().item()

            x = ind[:, 1].min().int().item()
            y = ind[:, 2].min().int().item()

            w = x_max - x
            h = y_max - y

            data_dict['image'] = _crop(data_dict['image'], x=x, y=y, z=0, w=w, h=h, d=z_max)
            data_dict['masks'] = _crop(data_dict['masks'], x=x, y=y, z=0, w=w, h=h, d=z_max)

        return data_dict


class random_crop:
    def __init__(self, shape: Tuple[int, int, int] = (256, 256, 26)) -> None:
        self.w = shape[0]
        self.h = shape[1]
        self.d = shape[2]

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly crops an image to a designated size. If the crop size is too big, it just takes as much as it can.

        for example:
            in_image = torch.rand((300, 150, 27))
            transform = random_crop(shape = (256, 256, 26))
            out_image = transform(in_image)
            out_image.shape
            %% (256, 150, 26)


        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        shape = data_dict['image'].shape

        x_max = shape[1] - self.w if shape[1] - self.w > 0 else 1
        y_max = shape[2] - self.h if shape[2] - self.h > 0 else 1
        z_max = shape[3] - self.d if shape[3] - self.d > 0 else 1

        x = torch.randint(x_max, (1, 1)).item()
        y = torch.randint(y_max, (1, 1)).item()
        z = torch.randint(z_max, (1, 1)).item()

        # Check if the crop doesnt contain any positive labels.
        # If it does, try generating new points
        # We want to make sure every training image has something to learn
        while _crop(data_dict['masks'], x=x, y=y, z=z, w=self.w, h=self.h, d=self.d).sum() == 0:
            x = torch.randint(x_max, (1, 1)).item()
            y = torch.randint(y_max, (1, 1)).item()
            z = torch.randint(z_max, (1, 1)).item()

        data_dict['image'] = _crop(data_dict['image'], x=x, y=y, z=z, w=self.w, h=self.h, d=self.d)
        data_dict['masks'] = _crop(data_dict['masks'], x=x, y=y, z=z, w=self.w, h=self.h, d=self.d)

        return data_dict


class random_v_flip:
    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.vflip)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly flips the mask vertically.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """

        if torch.rand(1) < self.rate:
            data_dict['image'] = _reshape(self.fun(_shape(data_dict['image'])))
            data_dict['masks'] = _reshape(self.fun(_shape(data_dict['masks'])))

        return data_dict


class random_h_flip:
    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.hflip)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly flips the mask vertically.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """

        if torch.rand(1) < self.rate:
            data_dict['image'] = _reshape(self.fun(_shape(data_dict['image'])))
            data_dict['masks'] = _reshape(self.fun(_shape(data_dict['masks'])))

        return data_dict


class normalize:
    def __init__(self, mean: Sequence[float] = (0.5), std: Sequence[float] = (0.5)) -> None:
        self.mean = mean
        self.std = std
        self.fun = torch.jit.script(torchvision.transforms.functional.normalize)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """


        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        data_dict['image'] = self.fun(data_dict['image'], self.mean, self.std)
        return data_dict


class random_noise:
    def __init__(self, gamma: float = 0.1, rate: float = 0.5):
        self.gamma = gamma
        self.rate = rate

    def __call__(self, data_dict: Dict[str, torch.Tensor], ) -> Dict[str, torch.Tensor]:
        """
        Adds noise to the image. Noise are values between 0 and 0.3

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        if torch.rand(1).item() < self.rate:
            device = data_dict['masks'].device
            noise = torch.rand(data_dict['image'].shape).to(device) * torch.tensor([self.gamma]).to(device)
            data_dict['image'] = data_dict['image'] + noise

        return data_dict


class gaussian_blur:
    def __init__(self, kernel_targets: torch.Tensor = torch.tensor([3, 5, 7]), rate: float = 0.5) -> None:
        self.kernel_targets = kernel_targets
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.gaussian_blur)

    def __call__(self, data_dict):
        """
        Randomly applies a gaussian blur

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictionary with identical keys as input, but with transformed values
        """
        if torch.rand(1) < self.rate:
            kern = self.kernel_targets[int(torch.randint(0, len(self.kernel_targets), (1, 1)).item())].item()
            data_dict['image'] = _reshape(self.fun(_shape(data_dict['image']), [kern, kern]))
        return data_dict


class adjust_brightness:
    def __init__(self, rate: float = 0.5, range_brightness: Tuple[float, float] = (-0.5, 0.5)) -> None:
        self.rate = rate
        self.range = range_brightness
        self.fun = _adjust_brightness

    def __call__(self, data_dict):
        if torch.rand(1) < self.rate:
            # funky looking but FAST
            val = torch.FloatTensor(data_dict['image'].shape[0]).uniform_(self.range[0], self.range[1])
            data_dict['image'] = self.fun(data_dict['image'], val)

        return data_dict


class adjust_gamma:
    def __init__(self, rate: float = 0.5, gamma: Tuple[float, float] = (0.5, 1.5),
                 gain: Tuple[float, float] = (.75, 1.25)) -> None:
        self.rate = rate
        self.gain = gain
        self.gamma = gamma

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly adjusts gamma of image color channels independently

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """

        if torch.rand(1) < self.rate:
            gamma = torch.FloatTensor(data_dict['image'].shape[0]).uniform_(self.gamma[0], self.gamma[1])
            gain = torch.FloatTensor(data_dict['image'].shape[0]).uniform_(self.gain[0], self.gain[1])

            data_dict['image'] = _adjust_gamma(data_dict['image'], gamma, gain)

        return data_dict


class elastic_deformation:
    def __init__(self, grid_shape: Tuple[int, int, int] = (2, 2, 2), scale: int = 2):
        self.x_grid = grid_shape[0]
        self.y_grid = grid_shape[1]
        self.z_grid = grid_shape[2] if len(grid_shape) > 2 else None
        self.scale = scale

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = data_dict['image'].device
        image = data_dict['image'].cpu().numpy()
        mask = data_dict['masks'].cpu().numpy()
        dtype = image.dtype

        displacement = np.random.randn(3, self.x_grid, self.y_grid, self.z_grid) * self.scale
        image = elasticdeform.deform_grid(image, displacement, axis=(1, 2, 3))
        mask = elasticdeform.deform_grid(mask, displacement, axis=(1, 2, 3), order=0)

        image[image < 0] = 0.0
        image[image > 1] = 1.0
        image.astype(dtype)

        data_dict['image'] = torch.from_numpy(image).to(device)
        data_dict['masks'] = torch.from_numpy(mask).to(device)

        return data_dict


class random_affine:
    def __init__(self, rate: float = 0.5, angle: Tuple[int, int] = (-180, 180),
                 shear: Tuple[int, int] = (-5, 5), scale: Tuple[float, float] = (0.9, 1.1)) -> None:
        self.rate = rate
        self.angle = angle
        self.shear = shear
        self.scale = scale

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Performs an affine transformation on the image and mask
        Shears, rotates and scales the image randomly based on parameters defined at initialization

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        if torch.rand(1) < self.rate:
            angle = torch.FloatTensor(1).uniform_(self.angle[0], self.angle[1])
            shear = torch.FloatTensor(1).uniform_(self.shear[0], self.shear[1])
            scale = torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1])
            translate = torch.tensor([0, 0])

            data_dict['image'] = _reshape(_affine(_shape(data_dict['image']), angle, translate, scale, shear))
            data_dict['masks'] = _reshape(_affine(_shape(data_dict['masks']), angle, translate, scale, shear))

        return data_dict


class to_cuda:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move every element in a dict containing torch tensor to cuda.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()
        return data_dict


class to_tensor:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, Union[torch.Tensor, Image, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Convert a PIL image or numpy.ndarray to a torch.Tensor

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        data_dict['image'] = torchvision.transforms.functional.to_tensor(data_dict['image'])
        return data_dict


class adjust_centroids:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Some Geometric transforms may alter the locations of cells so drastically that the centroid may no longer
        be accurate. This recalculates the centroids based on the current mask.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """

        shape = data_dict['masks'].shape
        device = data_dict['masks'].device
        centroid = torch.zeros(shape[0], 3, dtype=torch.float)
        ind = torch.ones(shape[0], dtype=torch.long)

        for i in range(shape[0]):  # num of instances
            data_dict['masks'][i, ...] = self._remove_edge_cells(data_dict['masks'][i, ...])
            indexes = torch.nonzero(data_dict['masks'][i, ...] > 0).float()

            #
            if indexes.shape[0] == 0:
                centroid[i, :] = torch.tensor([-1, -1, -1])
                ind[i] = 0
            # else:
            #     centroid[i, :] = torch.mean(indexes, dim=0)
            else:
                z_max = indexes[..., -1].max()
                z_min = indexes[..., -1].min()
                z = torch.round((z_max - z_min)/2 + z_min) - 2

                indexes = indexes[indexes[..., -1] == z, :]

                centroid[i, :] = torch.cat((torch.mean(indexes, dim=0)[0:2], torch.tensor([z]))).float()


        data_dict['centroids'] = centroid[ind.bool()].to(device)
        data_dict['masks'] = data_dict['masks'][ind.bool(), :, :, :]

        return data_dict

    @staticmethod
    @torch.jit.script
    def _remove_edge_cells(image: torch.Tensor) -> torch.Tensor:
        """
        meant to take in an bool tensor - if any positive value is touching the edges, remove it!

        :param image: [X, Y, Z]
        :return:
        """

        ind = torch.nonzero(image)
        for i in range(2):
            remove_bool = torch.any(ind[:, i] == 0) or torch.any(ind[:, i] == image.shape[i]-1)
            remove_bool = remove_bool if torch.sum(image) < 3000 else False

            # Remove cell if it touches the edge and is small
            if remove_bool:
                image = torch.zeros(image.shape)
                break

        return image


class colormask_to_mask:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Some Geometric transforms may alter the locations of cells so drastically that the centroid may no longer
        be accurate. This recalculates the centroids based on the current mask.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """

        data_dict['masks'] = self._colormask_to_torch_mask(data_dict['masks'])

        return data_dict

    @staticmethod
    @torch.jit.script
    def _colormask_to_torch_mask(colormask: torch.Tensor) -> torch.Tensor:
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

class debug:
    def __init__(self, ind: int = 0):
        self.ind = ind

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = data_dict['image']
        mask = data_dict['masks']
        try:
            assert image.shape[-1] == mask.shape[-1]
            assert image.shape[-2] == mask.shape[-2]
            assert image.shape[-3] == mask.shape[-3]

            assert image.max() <= 1
            assert mask.max() <= 1
            assert image.min() >= 0
            assert mask.min() >= 0
        except Exception as ex:
            print(self.ind)
            raise ex

        return data_dict


@torch.jit.script
def _affine(img: torch.Tensor, angle: torch.Tensor, translate: torch.Tensor, scale: torch.Tensor,
            shear: torch.Tensor) -> torch.Tensor:
    """
    Not to be publicly accessed! Only called through src.transforms.affine

    A jit scriptable wrapped version of torchvision.transforms.functional.affine
    Cannot, by rule, pass a dict to a torchscript function, necessitating this function


    WARNING: Performs an affine transformation on the LAST TWO DIMENSIONS
    ------------------------------------------------------------------------------------------------------------
        call _shape(img) prior to _affine such that this transformation is performed on the X and Y dimensions
        call _reshape on the output of _affine to return to [C, X, Y, Z]

        correct implementation looks like
         ```python
         from src.transforms import _shape, _reshape, _affine
         angle = torch.tensor([0])
         scale = torch.tensor([0])
         shear = torch.tensor([0])
         translate = torch.tensor([0])
         transformed_image = _reshape(_affine(_shape(img), angle, translate, scale, shear))

         ```


    :param img: torch.Tensor from data_dict of shape [..., X, Y]
    :param angle: torch.Tensor float in degrees
    :param translate: torch.Tensor translation factor. If zero, any transformations are done around center of image
    :param scale: torch.Tensor float Scale factor of affine transformation, if 1, no scaling is performed
    :param shear: torch.Tensor float shear factor of affine transformation, if 0, no shearing is performed
    :return: torch.Tensor
    """
    angle = float(angle.item())
    scale = float(scale.item())
    shear = [float(shear.item())]
    translate_list = [int(translate[0].item()), int(translate[1].item())]
    return torchvision.transforms.functional.affine(img, angle, translate_list, scale, shear)


@torch.jit.script
def _shape(img: torch.Tensor) -> torch.Tensor:
    """
    Shapes a 4D input tensor from shape [C, X, Y, Z] to [C, Z, X, Y]

    * some torchvision functional transforms only work on last two dimensions *

    :param img: torch.Tensor image of shape [C, X, Y, Z]
    :return:
    """
    # [C, X, Y, Z] -> [C, 1, X, Y, Z] ->  [C, Z, X, Y, 1] -> [C, Z, X, Y]
    return img.unsqueeze(1).transpose(1, -1).squeeze(-1)


@torch.jit.script
def _reshape(img: torch.Tensor) -> torch.Tensor:
    """
    Reshapes a 4D input tensor from shape [C, Z, X, Y] to [C, X, Y, Z]

    Performs corrective version of _shape

    * some torchvision functional transforms only work on last two dimensions *

    :param img: torch.Tensor image of shape [C, Z, X, Y]
    :return:
    """
    # [C, Z, X, Y] -> [C, Z, X, Y, 1] ->  [C, 1, X, Y, Z] -> [C, Z, X, Y]
    return img.unsqueeze(-1).transpose(1, -1).squeeze(1)


@torch.jit.script
def _crop(img: torch.Tensor, x: int, y: int, z: int, w: int, h: int, d: int) -> torch.Tensor:
    """
    torch scriptable function which crops an image

    :param img: torch.Tensor image of shape [C, X, Y, Z]
    :param x: x coord of crop box
    :param y: y coord of crop box
    :param z: z coord of crop box
    :param w: width of crop box
    :param h: height of crop box
    :param d: depth of crop box
    :return:
    """
    if img.ndim == 4:
        img = img[:, x:x + w, y:y + h, z:z + d]
    elif img.ndim == 5:
        img = img[:, :, x:x + w, y:y + h, z:z + d]
    else:
        raise IndexError('Unsupported number of dimensions')

    return img


@torch.jit.script
def _adjust_brightness(img: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    """
    Adjusts brigtness of img with val

    :param img: [C, X, Y, Z]
    :param val: Tensor[float] [C]
    :return:
    """
    img = img.add_(val.reshape(img.shape[0], 1, 1, 1).to(img.device))
    img[img < 0] = 0
    img[img > 1] = 1
    return img


@torch.jit.script
def _adjust_gamma(img: torch.Tensor, gamma: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """
    Assume img in shape [C, X, Y, Z]
    
    :param img: 
    :param gamma: 
    :param gain: 
    :return: 
    """
    for c in range(img.shape[0]):
        img[c, ...] = torchvision.transforms.functional.adjust_gamma(img[c, ...], gamma=gamma[c], gain=gain[c])
    return img


if __name__ == '__main__':
    a = torch.rand((500, 500, 30))
    print(adjust_centroids()._remove_edge_cells(a).max())
