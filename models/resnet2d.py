import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out += identity
        out = self.relu(out)
        return out


class ResNet18Small(nn.Module):
    def __init__(self, in_ch: int = 2, num_classes: int = 3, temp_feat_dim: int = 6, use_vibration: bool = True):
        super().__init__()
        self.use_vibration = bool(use_vibration)
        if self.use_vibration:
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 32, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.layer1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))
            self.layer2 = nn.Sequential(BasicBlock(32, 64, stride=2), BasicBlock(64, 64))
            self.layer3 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
            self.layer4 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.stem = self.layer1 = self.layer2 = self.layer3 = self.layer4 = self.pool = None
        self.use_temp = bool(temp_feat_dim and temp_feat_dim > 0)
        if self.use_temp:
            self.temp_proj = nn.Sequential(
                nn.Linear(temp_feat_dim, 32),
                nn.ReLU(inplace=True),
            )
            cls_in = (256 if self.use_vibration else 0) + 32
        else:
            self.temp_proj = None
            cls_in = 256
        if not self.use_vibration and not self.use_temp:
            raise ValueError("At least one of vibration or temperature must be enabled")
        self.classifier = nn.Linear(cls_in, num_classes)

    def forward(self, x, temp_feats):
        # x: (B,2,H,W); temp_feats: (B,6)
        z = None
        if self.use_vibration:
            z = self.stem(x)
            z = self.layer1(z)
            z = self.layer2(z)
            z = self.layer3(z)
            z = self.layer4(z)
            z = self.pool(z).flatten(1)

        if self.use_temp:
            t = self.temp_proj(temp_feats)
            out = torch.cat([z, t], dim=1) if z is not None else t
            out = self.classifier(out)
        else:
            out = self.classifier(z)
        return out
