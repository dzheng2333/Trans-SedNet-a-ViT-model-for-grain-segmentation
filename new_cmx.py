import torch.nn as nn
import torch.nn.functional as F
try:
    from .backbone.dual_segformer_w import *

    from .decoder import UnetDecoder

except:
    from backbone.dual_segformer_w import *

    from decoder import UnetDecoder


class new_CMX(nn.Module):
    """
    """

    def __init__(self, classes=10, backbone='cmx-b0', pretrained=False):
        super().__init__()
        if pretrained:
            pretrained = cmx[backbone]['backbone']['checkpoint']
        else:
            pretrained = None
        self.backbone = cmx[backbone]['backbone']['name'](pretrained=pretrained)
        self.decode_head = UnetDecoder(encoder_channels=cmx[backbone]['decode_head']['embed_dims'])

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(cmx[backbone]['decode_head']['embed_dims'][0], classes, kernel_size=1)
        )

    def forward(self, x):

        x0, x1 = x[0], x[1]
        hw = x0.size()[2:]

        x = self.backbone(x0, x1)
        x = self.decode_head(x)
        x = self.segmentation_head(x)

        x = F.interpolate(x, size=hw, mode='bilinear', align_corners=True)

        return x

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False


cmx = {
    "cmx-b0": {
        "backbone": dict(
            name=mit_b0,
            checkpoint=r'/home/gg/opt/experiment_detail/checkpoints/pretrained/mit_b0.pth'),
        "decode_head": dict(
            embed_dims=[32, 64, 160, 256]),
    }
}
