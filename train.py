import torch.nn.functional as F
import time
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from utils import AverageMeter, depthNorm, LogProgress
from loss import ssim

def train(model,  optimizer, device,
          train_loader, test_loader, num_epochs=1, lr = 0.0001,
          batch_size=4, logging_interval=300,
          save_model="DepthEstimator.pt"):

    print(device)
    model.to(device)
    prefix = 'densenet_' + str(batch_size)

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, lr, num_epochs, batch_size), flush_secs=30)

    min_loss = None

    # Start training...
    for epoch in range(num_epochs):
        print(epoch)
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time()

        for i, sample_batched in enumerate(test_loader):

            optimizer.zero_grad()
            # print("Batch")
            # print(sample_batched)
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].to(device))
            image = image.permute(0, 3, 1, 2)
            # print(image.size())
            depth = torch.autograd.Variable(sample_batched['depth'].to(device, non_blocking=True))

            # Normalize depth
            depth_n = depthNorm( depth )
            depth_n = depth_n.permute(0, 3, 2, 1)

            # Predict
            output = model(image)

            # Loss
            l1_criterion = nn.L1Loss()
            # Compute the loss
            l_depth = l1_criterion(output, depth_n)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

            loss = (1.0 * l_ssim) + (0.1 * l_depth)


            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val *(N - i))))

            # Log progress
            niter = epoch * N +i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            if i % logging_interval == 0:
                LogProgress(model, writer, test_loader, niter, device)

            if min_loss is None or loss < min_loss:
                min_loss = loss
                torch.save(model.state_dict(), save_model)


        # Record epoch's intermediate results
        LogProgress(model, writer, test_loader, niter, device)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)
