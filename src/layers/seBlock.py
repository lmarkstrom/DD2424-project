import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        # Global average pooling layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            # Downscale
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            # Upscale
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

        def forward(self, x):
            b, c, _, _ = x.size()

            squeeze = self.gap(x).view(b, c)
            excitation = self.fc(squeeze).view(b, c, 1, 1)

            # Conditioned output (self attention-ish)
            return x * excitation
