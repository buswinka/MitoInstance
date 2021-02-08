import torch
from typing import List


def calculate_indexes(pad_size: int, eval_image_size: int, image_shape: int, padded_image_shape: int) -> List[List[int]]:
    """
    This calculates indexes for the complete evaluation of an arbitrarily large image by unet.
    each index is offset by eval_image_size, but has a width of eval_image_size + pad_size * 2.
    Unet needs padding on each side of the evaluation to ensure only full convolutions are used
    in generation of the final mask. If the algorithm cannot evenly create indexes for
    padded_image_shape, an additional index is added at the end of equal size.

    :param pad_size: int corresponding to the amount of padding on each side of the
                     padded image
    :param eval_image_size: int corresponding to the shape of the image to be used for
                            the final mask
    :param image_shape: int Shape of image before padding is applied
    :param padded_image_shape: int Shape of image after padding is applied

    :return: List of lists corresponding to the indexes
    """

    # We want to account for when the eval image size is super big, just return index for the whole image.
    if eval_image_size > image_shape:
        return [[0, image_shape]]

    try:
        ind_list = torch.arange(0, image_shape, eval_image_size)
    except RuntimeError:
        raise RuntimeError(f'Calculate_indexes has incorrect values {pad_size} | {image_shape} | {eval_image_size}:\n'
                           f'You are likely trying to have a chunk smaller than the set evaluation image size. '
                           'Please decrease number of chunks.')
    ind = []
    for i, z in enumerate(ind_list):
        if i == 0:
            continue
        z1 = int(ind_list[i-1])
        z2 = int(z-1) + (2 * pad_size)
        if z2 < padded_image_shape:
            ind.append([z1, z2])
        else:
            break
    if not ind:  # Sometimes z is so small the first part doesnt work. Check if z_ind is empty, if it is do this!!!
        z1 = 0
        z2 = eval_image_size + pad_size * 2
        ind.append([z1, z2])
        ind.append([padded_image_shape - (eval_image_size+pad_size * 2), padded_image_shape])
    else:  # we always add at the end to ensure that the whole thing is covered.
        z1 = padded_image_shape - (eval_image_size + pad_size * 2)
        z2 = padded_image_shape - 1
        ind.append([z1, z2])
    return ind

@torch.jit.script
def remove_edge_cells(mask: torch.Tensor) -> torch.Tensor:
    """
    Removes cells touching the border

    :param mask: (B, X, Y, Z)
    :return: mask (B, X, Y, Z)
    """

    left = torch.unique(mask[:, 0, :, :])
    right = torch.unique(mask[:, -1, :, :])
    top = torch.unique(mask[:, :, 0, :])
    bottom = torch.unique(mask[:, :, -1, :])

    cells = torch.unique(torch.cat((left, right, top, bottom)))

    for c in cells:
        if c == 0:
            continue
        mask[mask == c] = 0

    return mask

#
@torch.jit.script
def remove_small_cells(mask: torch.Tensor) -> torch.Tensor:

    for u in torch.unique(mask).nonzero():

        if (mask == u).sum() < 4000:
            mask[mask == u] = 0

    return mask