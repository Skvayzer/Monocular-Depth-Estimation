import matplotlib
import torchvision.utils as vutils
import torch
import numpy as np
import os
import random

# MAKING ALL WEIGHTS OF NN NOT RANDOM, DETERMINISTIC
# TO INVESTIGATE THE TRUE EFFECTIVENESS OF THE MODEL
def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def depthNorm(depth, maxdepth=1000):
  return maxdepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def LogProgress(model, writer, test_loader, epoch, device):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].to(device))
    depth = torch.autograd.Variable(sample_batched['depth'].to(device, non_blocking=True))

    image = image.permute(0, 3, 1, 2)
    depth_n = depth.permute(0, 3, 2, 1)

    # print(image.data.size())
    # print(depth.data.size())
    # print(vutils.make_grid(image.data, nrow=6, normalize=True).size())
    # print(colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)).size())
    if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)),
                                    epoch)
    output = depthNorm(model(image))

    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff',
                     colorize(vutils.make_grid(torch.abs(output - depth).data, nrow=6, normalize=False)), epoch)
    del image
    del depth
    del output


def colorize(value, vmin=None, vmax=None, cmap='plasma'):
    value = value.cpu().numpy()[0, :, :]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    return torch.from_numpy(img.transpose((2, 0, 1)))

