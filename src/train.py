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
import torch.autograd.profiler as profiler

import src.transforms as t
import skimage.io as io
import os.path

from tqdm import trange

epochs = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.jit.script(HCNet(in_channels=1, out_channels=3, complexity=15)).to(device)
model.train()
# model.load_state_dict(torch.load('../modelfiles/Feb09_12-56-49_chris-MS-7C37.hcnet'))

writer = SummaryWriter()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 250, gamma=0.5, last_epoch=-1, verbose=False)
loss_fun = src.loss.tversky_loss()

transforms = torchvision.transforms.Compose([
    t.nul_crop(2),
    t.random_crop(shape=(256, 256, 24)),
    t.to_cuda(),
    t.random_h_flip(),
    t.random_v_flip(),
    # t.random_affine(),
    # t.adjust_brightness(),
])
data = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/MitoInstance/data',
                              transforms=transforms)
data = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

epoch_range = trange(epochs, desc='Loss: {1.00000}', leave=True)


sigma = torch.tensor([0.1, 0.1, 0.05], device=device)  # [x, y, z]


for e in epoch_range:
    epoch_loss = []
    model.train()

    if e == 100 and e == 250 and e == 500:
        sigma /= 2

    # EVERYTHING SHOULD BE ON CUDA
    for data_dict in data:
        image = data_dict['image'].sub_(0.5).div_(0.5)
        mask = data_dict['masks'].gt_(0.5)
        centroids = data_dict['centroids']

        optimizer.zero_grad()                                                                      # Zero Gradients

        out = model(image, 5)                                                                      # Eval Model
        out = src.functional.vector_to_embedding(out)                                              # Calculate Embedding
        out = src.functional.embedding_to_probability(out, centroids, sigma)                       # Generate Prob Map

        loss = loss_fun(out, mask, 1, 1)                                                           # Calculate Loss
        loss.backward()                                                                            # Backpropagation

        optimizer.step()                                                                           # Update Adam
        scheduler.step()

        epoch_loss.append(loss.detach().cpu().item())

    epoch_range.desc = 'Loss: ' + '{:.5f}'.format(torch.tensor(epoch_loss).mean().item())
    writer.add_scalar('Loss/train', torch.mean(torch.tensor(epoch_loss)).item(), e)

torch.save(model.state_dict(), '../modelfiles/' + os.path.split(writer.log_dir)[-1] + '.hcnet')

render = (out > 0.5).int().squeeze(0)
for i in range(render.shape[0]):
    render[i, :, :, :] = render[i, :, :, :] * (i + 1)
io.imsave('test.tif', render.sum(0).detach().cpu().int().numpy().astype(np.int).transpose((2, 1, 0)) / i + 1)
io.imsave('im.tif', image.squeeze(0).squeeze(0).detach().cpu().numpy().transpose((2, 1, 0)) * 0.5 + 0.5)

print(centroids.shape, centroids)
print(image.shape)


with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True) as prof:
    with profiler.record_function("src.functional.embedding_to_probability"):
        out = model(image, 5)  # Eval Model
        out = src.functional.vector_to_embedding(out)
        out = src.functional.embedding_to_probability(out, centroids, sigma)                # Generate Prob Map
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))


