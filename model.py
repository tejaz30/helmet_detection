import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional


# -------------------------------
# 🔹 1. Basic Conv Block
# Convolution -> BatchNorm -> LeakyReLU
# This is the most common pattern in YOLOv3
# -------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2  # automatic padding for "same" convs

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)  # LeakyReLU instead of ReLU

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


# -------------------------------
# 🔹 2. Residual Block
# This is the ResNet trick:
# output = input + F(input)
# Helps deep networks train better
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels, num_blocks=1):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Sequential(
                # Bottleneck structure:
                ConvBlock(channels, channels // 2, kernel_size=1),   # reduce channels
                ConvBlock(channels // 2, channels, kernel_size=3)    # process and restore
            ))
        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual   # add skip connection
        return x


# -------------------------------
# 🔹 3. Darknet-53 Backbone
# Extracts features at multiple scales
# -------------------------------
class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = ConvBlock(3, 32, 3)  # first conv layer (RGB -> 32)

        # Each "_make_layer" = ConvBlock(stride=2) + ResidualBlocks
        self.layer1 = self._make_layer(32, 64, num_blocks=1)     # 1 resblock
        self.layer2 = self._make_layer(64, 128, num_blocks=2)    # 2 resblocks
        self.layer3 = self._make_layer(128, 256, num_blocks=8)   # 8 resblocks
        self.layer4 = self._make_layer(256, 512, num_blocks=8)   # 8 resblocks
        self.layer5 = self._make_layer(512, 1024, num_blocks=4)  # 4 resblocks

    def _make_layer(self, in_channels, out_channels, num_blocks):
        # Downsample first (stride=2)
        layers = [ConvBlock(in_channels, out_channels, 3, stride=2)]
        # Then stack residual blocks
        layers.append(ResidualBlock(out_channels, num_blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        route_1 = self.layer3(x)  # output for small objects
        route_2 = self.layer4(route_1)  # medium objects
        route_3 = self.layer5(route_2)  # large objects
        return route_1, route_2, route_3


# -------------------------------
# 🔹 4. Detection Head
# Converts feature maps -> predictions
# -------------------------------
class YOLOv3DetectionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # small CNN head (5 conv layers + final conv)
        self.layers = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            ConvBlock(out_channels, in_channels, 3),
            ConvBlock(in_channels, out_channels, 1),
            ConvBlock(out_channels, in_channels, 3),
            ConvBlock(in_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.layers(x)


# -------------------------------
# 🔹 5. Full YOLOv3 Model
# Backbone + FPN + Detection Heads
# -------------------------------
class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = Darknet53()

        # Detection head channels
        self.head1 = YOLOv3DetectionHead(1024, 512)
        self.det1 = nn.Conv2d(512, 3 * (5 + num_classes), 1)  # 3 anchors × (x,y,w,h,obj,classes)

        self.head2_conv = ConvBlock(512, 256, 1)
        self.head2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.head2 = YOLOv3DetectionHead(768, 256)
        self.det2 = nn.Conv2d(256, 3 * (5 + num_classes), 1)

        self.head3_conv = ConvBlock(256, 128, 1)
        self.head3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.head3 = YOLOv3DetectionHead(384, 128)
        self.det3 = nn.Conv2d(128, 3 * (5 + num_classes), 1)

    def forward(self, x):
        # Extract multi-scale features
        route_1, route_2, route_3 = self.backbone(x)

        # Scale 1: large objects
        x = self.head1(route_3)
        out1 = self.det1(x)

        # Scale 2: medium objects
        x = self.head2_conv(x)
        x = self.head2_upsample(x)
        x = torch.cat([x, route_2], dim=1)  # combine deep + medium
        x = self.head2(x)
        out2 = self.det2(x)

        # Scale 3: small objects
        x = self.head3_conv(x)
        x = self.head3_upsample(x)
        x = torch.cat([x, route_1], dim=1)  # combine deep + shallow
        x = self.head3(x)
        out3 = self.det3(x)

        return out1, out2, out3


# -------------------------------
# 🔹 6. Decode Predictions
# Converts raw predictions -> usable boxes
# -------------------------------
def decode_predictions(pred, anchors, num_classes, input_dim):
    """
    pred: raw output from one scale [B, A*(5+C), H, W]
    anchors: list of anchor boxes for this scale
    input_dim: size of input image (e.g. 416x416)
    """
    batch_size, _, grid_size, _ = pred.shape
    stride = input_dim // grid_size
    num_anchors = len(anchors)

    # Reshape: [B, A, (5+C), H, W]
    pred = pred.view(batch_size, num_anchors, 5 + num_classes, grid_size, grid_size)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()

    # Sigmoid center coords & objectness
    pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])  # x,y
    pred[..., 4:] = torch.sigmoid(pred[..., 4:])    # obj + classes

    # Width/Height (exp on anchors)
    pred[..., 2:4] = torch.exp(pred[..., 2:4]) * torch.tensor(anchors, device=pred.device)

    # Grid offsets
    grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).float()
    grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).float()
    pred[..., 0] += grid_x.to(pred.device)
    pred[..., 1] += grid_y.to(pred.device)

    # Scale to input image size
    pred[..., :4] *= stride
    return pred.view(batch_size, -1, 5 + num_classes)


# -------------------------------
# 🔹 7. Test the Model
# -------------------------------
if __name__ == "__main__":
    model = YOLOv3(num_classes=3)  # Example: 3 classes (helmet, person, no-helmet)
    x = torch.randn(2, 3, 416, 416)  # 2 fake images

    out1, out2, out3 = model(x)

    print("Output shapes:")
    print("Scale 1:", out1.shape)  # -> [2, 3*(5+C), 13, 13]
    print("Scale 2:", out2.shape)  # -> [2, 3*(5+C), 26, 26]
    print("Scale 3:", out3.shape)  # -> [2, 3*(5+C), 52, 52]

    # Count params
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
