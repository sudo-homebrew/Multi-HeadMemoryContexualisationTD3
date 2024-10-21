""" Full assembly of the parts to form the complete network """

from .fbe_unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        channel = 8

        self.inc = (DoubleConv(n_channels, channel))
        self.down1 = (Down(channel, channel * (2 ** 1)))
        self.down2 = (Down(channel * (2 ** 1), channel * (2 ** 2)))
        self.down3 = (Down(channel * (2 ** 2), channel * (2 ** 3)))
        factor = 2 if bilinear else 1
        self.down4 = (Down(channel * (2 ** 3), channel * (2 ** 4) // factor))
        self.up1 = (Up(channel * (2 ** 4), channel * (2 ** 3) // factor, bilinear))
        self.up2 = (Up(channel * (2 ** 3), channel * (2 ** 2) // factor, bilinear))
        self.up3 = (Up(channel * (2 ** 2), channel * (2 ** 1) // factor, bilinear))
        self.up4 = (Up(channel * (2 ** 1), channel, bilinear))
        self.outc = (OutConv(channel, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)