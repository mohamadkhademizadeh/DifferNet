import torch, torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_backbone(feature='layer3'):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    layers = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)
    feat_dim = 256 * 14 * 14  # for 224x224 input
    return layers, feat_dim

def extract_feat(x, backbone):
    f = backbone(x)
    return torch.flatten(f, start_dim=1)
