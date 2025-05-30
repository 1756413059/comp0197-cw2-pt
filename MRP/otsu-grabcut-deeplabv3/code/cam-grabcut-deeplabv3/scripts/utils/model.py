import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50

# resnet18 and resnet50 models for classification tasks
def get_resnet18(num_classes=37, pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def get_resnet50(num_classes=37, pretrained=True):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# UNet model for segmentation tasks
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out(d1)
        return out


def get_segmentor(model_name='deeplabv3', num_classes=1):
    if model_name == 'deeplabv3':
        model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
    elif model_name == 'unet':
        model = UNet(in_channels=3, out_channels=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def freeze_backbone(model, unfreeze_layers=('layer4',)):
    """
    Freeze backbone except specific layers.
    """
    for name, param in model.backbone.named_parameters():
        if any(name.startswith(layer) for layer in unfreeze_layers):
            param.requires_grad = True
        else:
            param.requires_grad = False
            print(f" - Frozen: backbone.{name}")

    # Make sure classifier head is trainable
    for name, param in model.classifier.named_parameters():
        param.requires_grad = True
