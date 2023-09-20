# 2022/7/18 14:03
import torch
import torch.nn.modules as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """
    1. 转置矩阵：通道变为一半，shape变为2倍   对应图中的up-conv 2x2
    2. 通道调整
    3. 普通卷积操作
    """

    def __init__(self, in_channels, out_channels, up_in_channel, up_channel):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_channel, up_channel, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.size()[2:] != skip.size()[2:]:
                x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CenterBlock(nn.Sequential):
    """
    功能是实现Unet最下面一层的512-1024-1024
    """

    def __init__(self, in_channel, out_channel):
        center = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        super().__init__(center)


class UnetDecoder(nn.Module):
    """
    拼接各个部分
    """

    def __init__(self, encoder_channels):
        super().__init__()

        encoder_channels = encoder_channels[::-1]
        in_channels = [2 * i for i in encoder_channels]
        up_in_channels = [2 * encoder_channels[0], encoder_channels[0],
                          encoder_channels[1], encoder_channels[2]]
        # print(up_in_channels)
        out_channels = encoder_channels

        # 32, 64, 160, 256

        self.center = CenterBlock(encoder_channels[0], in_channels[0])

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(in_ch, out_ch, up_in_ch, up_ch)
            for in_ch, out_ch, up_in_ch, up_ch in zip(in_channels, out_channels, up_in_channels, encoder_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):
        x = features[-1]

        x = self.center(x)
        skips = features[::-1]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i]
            x = decoder_block(x, skip)

        return x


if __name__ == '__main__':
    a = torch.rand([1, 3, 512, 512])
    from backbone.resnet import ResNetEncoder

    r = ResNetEncoder(model_name='resnet50', pretrained=False)
    b = r(a)
    u = UnetDecoder(encoder_channels=r.out_channels)
    c = u(b)
    print(c.shape)
