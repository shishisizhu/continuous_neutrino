# generate_data.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset

# -------------------------------
# 配置
# -------------------------------
SEED = 42
DATASET_SIZE = 50000
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
DATA_FILE = 'fixed_dataset.pth'
STATE_FILE = 'fixed_model_state.pth'

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------------
# ResNet-18 模型定义（内联写入，避免 import 问题）
# -------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# -------------------------------
# 生成固定数据和标签
# -------------------------------
print("Generating fixed dataset...")

# 模拟图像数据：3x32x32
X = torch.randn(DATASET_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
y = torch.randint(0, NUM_CLASSES, (DATASET_SIZE,))

# 保存数据
torch.save({
    'images': X,
    'labels': y,
    'seed': SEED
}, DATA_FILE)

print(f"✅ Dataset saved to {DATA_FILE}")

# -------------------------------
# 生成并保存固定的模型初始权重
# -------------------------------
torch.manual_seed(SEED)  # 确保模型参数初始化一致
model = ResNet18(num_classes=NUM_CLASSES)
model.to('cpu')  # 保存时统一在 CPU 上

# 强制参数初始化（虽然 torch 默认会初始化，但确保可复现）
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# 保存模型状态字典
torch.save({
    'model_state_dict': model.state_dict(),
    'seed': SEED
}, STATE_FILE)

print(f"✅ Model initial state saved to {STATE_FILE}")

