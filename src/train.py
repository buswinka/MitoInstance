import src.dataloader
import src.loss
import src.functional
import src.diagnostics
from src.models.HCNet import HCNet as HCNet
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

torch.random.manual_seed(0)

# Hyperparams and perallocation
epochs = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# sigma = lambda e, epochs: torch.tensor([.08], device=device) * (0.5 * e > 20 or 1) * ((0.5 * (e > 50)) + 1)  # [x, y, z] OUTPUT FROM BEST GUESTIMATE

def sigma(e, epochs):
    a = 0.5 if e > 20 else 1
    b = 0.5 if e > 200 else 1
    return torch.tensor([0.08, 0.08, 0.06], device=device) * a * b


# temp_data = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/MitoInstance/data',
#                                    transforms=torchvision.transforms.Compose([t.nul_crop(2)]))
# temp_data = DataLoader(temp_data, batch_size=1, shuffle=False, num_workers=0)
# sigma = src.functional.estimate_maximum_sigma(temp_data)
# del temp_data
# print(sigma)

writer = SummaryWriter()

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

# Load Model
model = torch.jit.script(HCNet(in_channels=1, out_channels=3, complexity=15)).to(device)
model.train()
# model.load_state_dict(torch.load('../modelfiles/Feb09_12-56-49_chris-MS-7C37.hcnet'))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.75, last_epoch=-1)

i = 0
epoch_range = trange(epochs, desc='Loss: {1.00000}', leave=True)
for e in epoch_range:

    # if torch.any(torch.eq(torch.tensor([25, 50]), torch.tensor([e]))): #20 or e == 50 or e == 150 or e == 250:
    #     sigma /= 2

    # EVERYTHING SHOULD BE ON CUDA
    imax = 5
    for i in range(imax):
        epoch_loss = []
        for data_dict in data:
            image = data_dict['image'].sub(0.5).div(0.5)
            mask = data_dict['masks'].gt(0.5)
            centroids = data_dict['centroids']

            optimizer.zero_grad()  # Zero Gradients

            out = model(image, 5)  # Eval Model

            if i == imax - 1:
                img = torch.clone(out).detach().cpu()
                img = img.squeeze(0).numpy()[:, :, :, 12] * 0.5 + 0.5
                writer.add_image('vector', img.astype(np.float64), e, dataformats='CHW')

            out = src.functional.vector_to_embedding(out)  # Calculate Embedding
            out = src.functional.embedding_to_probability(out, centroids, sigma(e, epochs))  # Generate Prob Map

            if i == imax - 1:
                img = torch.clone(out).detach().cpu().sum(1)
                img = img/img.max()
                img = img.squeeze(0).numpy()[:, :, 12] * 0.5 + 0.5
                writer.add_image('probability', img.astype(np.float64), e, dataformats='HW')

                img = torch.clone(image).detach().cpu().squeeze(0).squeeze(0).numpy()[:,:,12] * 0.5 + 0.5
                writer.add_image('image', img.astype(np.float64), e, dataformats='HW')

                img = torch.clone(mask).detach().cpu().squeeze(0).sum(0).numpy()
                img = (img/img.max())[:, :, 12]
                writer.add_image('mask', img.astype(np.float64), e, dataformats='HW')

            loss = loss_fun(out, mask)  # , 0.5, 0.5)  # Calculate Loss
            epoch_loss.append(loss.item())
            loss.backward()  # Backpropagation

            if i == imax - 1:  # Accumulate gradients
                optimizer.step()  # Update weights

            scheduler.step()

    epoch_loss.append(torch.tensor(epoch_loss).mean().detach().cpu().item())

    epoch_range.desc = 'Loss: ' + '{:.5f}'.format(torch.tensor(epoch_loss).mean().item())
    writer.add_scalar('Loss/train', torch.mean(torch.tensor(epoch_loss)).item(), e)
    writer.add_scalar('Hyperparam/sigma_x', sigma(e, epochs)[0].item(), e)

torch.save(model.state_dict(), '../modelfiles/' + os.path.split(writer.log_dir)[-1] + '.hcnet')

render = (out > 0.5).int().squeeze(0)
for i in range(render.shape[0]):
    render[i, :, :, :] = render[i, :, :, :] * (i + 1)
io.imsave('test.tif', render.sum(0).detach().cpu().int().numpy().astype(np.int).transpose((2, 1, 0)) / i + 1)
io.imsave('im.tif', image.squeeze(0).squeeze(0).detach().cpu().numpy().transpose((2, 1, 0)) * 0.5 + 0.5)

print(centroids.shape, centroids)
print(image.shape)

out = model(image, 5)  # Eval Model
out = src.functional.vector_to_embedding(out)
src.diagnostics.sigma_sweep(out, centroids, mask, loss_fun)

with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True) as prof:
    with profiler.record_function("src.functional.embedding_to_probability"):
        out = model(image, 5)  # Eval Model
        out = src.functional.vector_to_embedding(out)
        out = src.functional.embedding_to_probability(out, centroids, sigma * 2)  # Generate Prob Map
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
