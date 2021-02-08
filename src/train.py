import src.dataloader
import src.loss
import src.functional
from src.models.HCNet import HCNet
import torch.nn
from torch.utils.data import DataLoader
import time
import numpy as np
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter

import src.transforms as t
import skimage.io as io
import os.path

from tqdm import trange

epochs = 100

model = torch.jit.script(HCNet(in_channels=1, out_channels=3, complexity=10)).cuda()
model.train()
model.load_state_dict(torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/MitoInstance/modelfiles/Feb08_16-43-52_chris-MS-7C37.hcnet'))

writer = SummaryWriter()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fun = src.loss.jaccard_loss()

transforms = torchvision.transforms.Compose([
    t.nul_crop(2),
    t.random_crop(shape=(256, 256, 30)),
    t.to_cuda(),
    # t.random_h_flip(),
    # t.random_v_flip(),
    # t.random_affine(),
    # t.adjust_brightness(),
])
data = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/MitoInstance/data',
                              transforms=transforms)
data = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

epoch_range = trange(epochs, desc='Loss: {1.00000}', leave=True)

sigma = torch.tensor([0.005])


for e in epoch_range:
    epoch_loss = []
    model.train()
    for data_dict in data:
        image = (data_dict['image'] - 0.5) / 0.5
        mask = data_dict['masks'] > 0.5
        centroids = data_dict['centroids']

        optimizer.zero_grad()                                                                      # Zero Gradients

        out = model(image.cuda(), 5)                                                               # Eval Model
        out = src.functional.vector_to_embedding(out)                               # Calculate Embedding
        out = src.functional.embedding_to_probability(out, centroids.cuda(), sigma)                # Generate Prob Map

        try:
            loss = loss_fun(out, mask.cuda())                                                          # Calculate Loss
        except AssertionError:
            print(image.shape, mask.shape, centroids.shape, out.max())
            raise RuntimeError

        try:
            loss.backward()                                                                            # Backpropagation
        except RuntimeError:
            print(image.shape)
            print(mask.shape)
            print(centroids.shape)
            raise RuntimeError

        optimizer.step()                                                                           # Update Adam

        epoch_loss.append(loss.detach().cpu().item())

    epoch_range.desc = 'Loss: ' + '{:.5f}'.format(torch.tensor(epoch_loss).mean().item())
    writer.add_scalar('Loss/train', torch.mean(torch.tensor(epoch_loss)).item(), e)
    torch.save(model.state_dict(), '../modelfiles/' + os.path.split(writer.log_dir)[-1] + '.hcnet')



render = (out > 0.5).int().squeeze(0)
for i in range(render.shape[0]):
    render[i, :, :, :] = render[i, :, :, :] * (i + 1)
io.imsave('test.tif', render.sum(0).detach().cpu().int().numpy().astype(np.int).transpose((2, 1, 0)) / i + 1)
io.imsave('im.tif', image.squeeze(0).squeeze(0).detach().cpu().int().numpy().transpose((2, 1, 0)) * 0.5 + 0.5)
