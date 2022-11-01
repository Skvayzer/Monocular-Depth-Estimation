import torch
from data import getTrainingTestingData
from utils import set_deterministic, set_all_seeds, colorize
from model import DepthEstimator
from train import train
import matplotlib.pyplot as plt


##########################
### SETTINGS
##########################

# Device
CUDA_DEVICE_NUM = 0
DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# Hyperparameters
RANDOM_SEED = 123
LEARNING_RATE = 0.0001
BATCH_SIZE = 4
NUM_EPOCHS = 1
LOGGING_INTERAL = 300


set_deterministic()
set_all_seeds(RANDOM_SEED)

# Load data
train_loader, test_loader = getTrainingTestingData(batch_size=BATCH_SIZE, path_prefix="")

# Create model
model = DepthEstimator()
print('Model created.')


# Training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

##########################
### TRAINING
##########################

# train(model, optimizer, DEVICE, train_loader, test_loader,  NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, LOGGING_INTERAL,
#       save_model="DepthEstimator.pt")


# TESTING
model = DepthEstimator()
model.load_state_dict(torch.load('./DepthEstimator.pt'))
model.eval()

batch = next(iter(iter(test_loader)))
image = batch['image'][1]
plt.imshow(image)
batch = batch['image'].permute(0, 3, 1, 2)
output = model(batch)
colorized = colorize(output[1].detach()).permute(1,2,0)

plt.imshow(colorized)
plt.imsave('test.png', colorized.cpu().numpy())
print("DONE")

