import torch.utils.checkpoint

from .model_parts import *


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.input_layer = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 / factor))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.out_layer = (OutConv(64, n_classes))
    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_layer(x)
        return logits

    def use_checkpointing(self):
        self.input_layer = torch.utils.checkpoint.checkpoint(self.input_layer, 'input_layer')
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1, 'down1')
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2, 'down2')
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3, 'down3')
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4, 'down4')
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1, 'up1')
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2, 'up2')
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3, 'up3')
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4, 'up4')
        self.out_layer = torch.utils.checkpoint.checkpoint(self.out_layer, 'out_layer')