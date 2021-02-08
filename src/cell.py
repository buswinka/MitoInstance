import torch


class cell:
    def __init__(self, image: torch.Tensor, mask: torch.Tensor, place: float = 0):
        """

        :param im: [B, 4, X, Y, Z]
        :param mask: [B, C=1, X, Y, Z]
        :param location:
        """

        self.volume = mask.gt(0.5).sum()
        self.place = place
        self.indexes = torch.nonzero(mask[0, 0, ...]).float()

        # 0:DAPI
        # 1:GFP
        # 2:MYO7a
        # 3:Actin
        # assume image is in [B, C, X, Y, Z]

        self.dapi = image[0, 0, ...][mask[0, 0, ...].gt(0.5)].mean()
        self.gfp = image[0, 1, ...][mask[0, 0, ...].gt(0.5)].mean()
        self.myo7a = image[0, 2, ...][mask[0, 0, ...].gt(0.5)].mean()
        self.actin = image[0, 3, ...][mask[0, 0, ...].gt(0.5)].mean()
