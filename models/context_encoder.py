import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, channels=3, masked_height=64, mask_width=64):
        super(Generator, self).__init__()
        
        self.masked_h = masked_height
        self.masked_w = mask_width
        
        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers
        
        self.down1 = nn.Sequential(*downsample(channels, 64, normalize=False))
        self.down2 = nn.Sequential(*downsample(64, 64))
        self.down3 = nn.Sequential(*downsample(64, 128))
        self.down4 = nn.Sequential(*downsample(128, 256))
        self.down5 = nn.Sequential(*downsample(256, 512))

        self.middle = nn.Conv2d(512, 4000, 1)

        self.up1 = nn.Sequential(*upsample(4000, 512))
        self.up2 = nn.Sequential(*upsample(512, 256))
        self.up3 = nn.Sequential(*upsample(256, 128))
        self.up4 = nn.Sequential(*upsample(128, 64))

        # Calculate the padding needed to get the exact output size
        self.final_conv = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)
        self.upsample_to_masked = nn.Upsample(size=(self.masked_h, self.masked_w), mode='bilinear', align_corners=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        x = self.middle(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_conv(x)
        x = self.upsample_to_masked(x)
        
        return self.tanh(x)


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
