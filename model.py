import torch
from torchvision import models
import torch.nn.functional as F
from torch import nn

##########################
### MODEL
##########################


class UpsampleBlock(nn.Module):
    def __init__(self, skip_input_channels, output_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_block = nn.Sequential(
            nn.Conv2d(skip_input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, encodedFeatures):
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        # print(encodedFeatures.size(), upsampled.size())
        concat = torch.cat([encodedFeatures, upsampled], dim=1)
        result = self.upsample_block(concat)
        return result


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet169(pretrained=True)

    def forward(self, x):
        features = [x]
        for name, module in self.model.features._modules.items():
            x = module(x)
            features.append(x)
        # print('END')
        return features


class Decoder(nn.Module):
    def __init__(self, num_features=1664):
        super().__init__()
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpsampleBlock(skip_input_channels=1920, output_channels=832, scale_factor=2)
        self.up2 = UpsampleBlock(skip_input_channels=960, output_channels=416, scale_factor=2)
        self.up3 = UpsampleBlock(skip_input_channels=480, output_channels=208, scale_factor=2)
        self.up4 = UpsampleBlock(skip_input_channels=272, output_channels=104, scale_factor=2)
        self.conv3 = nn.Conv2d(in_channels=104, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[
            12]
        x = self.conv2(F.relu(x_block4))
        x = self.up1(x, x_block3)
        x = self.up2(x, x_block2)
        x = self.up3(x, x_block1)
        x = self.up4(x, x_block0)
        x = self.conv3(x)
        return x


class DepthEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded