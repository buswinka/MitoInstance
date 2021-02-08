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

from tqdm import trange

epochs = 150

model = torch.jit.script(HCNet(in_channels=3, out_channels=6, complexity=10)).cuda()
model.train()
# model.load_state_dict(torch.load('./trained_model_hcnet_long.mdl'))

writer = SummaryWriter()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fun = src.loss.jaccard_loss()

transforms = torchvision.transforms.Compose([
    t.nul_crop(),
    t.random_crop(shape=(256, 256, 23)),
    # t.to_cuda(),
    # t.random_h_flip(),
    # t.random_v_flip(),
    # t.random_affine(),
    # t.adjust_brightness(),
    t.adjust_centroids(),
])
data = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test',
                              transforms=transforms)
data = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

epoch_range = trange(epochs, desc='Loss: {1.00000}', leave=True)

sigma = torch.tensor([0.0008])

for e in epoch_range:
    epoch_loss = []
    model.train()
    for data_dict in data:
        image = (data_dict['image'] - 0.5) / 0.5
        mask = data_dict['masks'] > 0.5
        centroids = data_dict['centroids']

        optimizer.zero_grad()                                                                      # Zero Gradients

        out = model(image.cuda(), 5)                                                               # Eval Model
        out = src.functional.vector_to_embedding(out[:, 0:3:1, ...])                               # Calculate Embedding
        out = src.functional.embedding_to_probability(out, centroids.cuda(), sigma)                # Generate Prob Map

        loss = loss_fun(out, mask.cuda())                                                          # Calculate Loss
        loss.backward()                                                                            # Backpropagation
        optimizer.step()                                                                           # Update Adam

        epoch_loss.append(loss.detach().cpu().item())

    epoch_range.desc = 'Loss: ' + '{:.5f}'.format(torch.tensor(epoch_loss).mean().item())
    writer.add_scalar('Loss/train', torch.mean(torch.tensor(epoch_loss)).item(), e)
    torch.save(model.state_dict(), 'modelfiles' + writer.log_dir + '.hcnet')



render = (out > 0.5).int().squeeze(0)
for i in range(render.shape[0]):
    render[i, :, :, :] = render[i, :, :, :] * (i + 1)
io.imsave('test.tif', render.sum(0).detach().cpu().int().numpy().astype(np.int).transpose((2, 1, 0)) / i + 1)
