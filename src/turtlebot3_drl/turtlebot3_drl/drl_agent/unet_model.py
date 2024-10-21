""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        inner_channal = 2 ** 4

        self.inc = (DoubleConv(n_channels, inner_channal))
        self.down1 = (Down(inner_channal, inner_channal * (2 ** 1)))
        self.down2 = (Down(inner_channal * (2 ** 1), inner_channal * (2 ** 2)))
        self.down3 = (Down(inner_channal * (2 ** 2), inner_channal * (2 ** 3)))
        factor = 2 if bilinear else 1
        self.down4 = (Down(inner_channal * (2 ** 3), inner_channal * (2 ** 4) // factor))
        self.up1 = (Up(inner_channal * (2 ** 4), inner_channal * (2 ** 3) // factor, bilinear))
        self.up2 = (Up(inner_channal * (2 ** 3), inner_channal * (2 ** 2) // factor, bilinear))
        self.up3 = (Up(inner_channal * (2 ** 2), inner_channal * (2 ** 1) // factor, bilinear))
        self.up4 = (Up(inner_channal * (2 ** 1), inner_channal, bilinear))
        self.outc = (OutConv(inner_channal, n_classes))

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