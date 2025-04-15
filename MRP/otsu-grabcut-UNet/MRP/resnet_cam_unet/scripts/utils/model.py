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




# ======================
# UNet Class
# ======================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
        A basic U-Net architecture.
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels in the output segmentation map.
            features (list): List of feature sizes for each encoder stage.
        """
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Encoder (Downsampling path)
        in_ch = in_channels
        for feature in features:
            self.downs.append(self.double_conv(in_ch, feature))
            in_ch = feature
        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)
        # Decoder (Upsampling path)
        self.ups = nn.ModuleList()
        rev_features = features[::-1]
        for feature in rev_features:
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(self.double_conv(feature * 2, feature))
        # Final 1x1 convolution to get the desired output channels.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        """
        Returns a sequential block with two conv layers, each followed by BatchNorm and ReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []
        # Encoder: store outputs for skip connections.
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        # Reverse the skip connections for the decoder.
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            # Adjust if needed in case the shape differs due to rounding during pooling.
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)
        x = self.final_conv(x)
        return torch.sigmoid(x)  # Ensure output is in [0, 1]


# ======================
# Combined Loss Class
# ======================
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        """
        Combines Binary Cross-Entropy (BCE) loss and Dice loss.
        Args:
            bce_weight (float): Weight for BCE loss.
            dice_weight (float): Weight for Dice loss.
        """
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        """
        Computes the combined loss
        Args:
            pred (Tensor): Predicted segmentation (B, 1, H, W) with values in [0, 1].
            target (Tensor): Ground truth binary segmentation (B, 1, H, W) with values 0 or 1.
        Returns:
            loss (Tensor): Weighted sum of BCE and Dice losses.
        """
        # Binary Cross Entropy Loss
        bce_loss = self.bce(pred, target)
        # Dice Loss computation.
        smooth = 1e-7
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice_loss = 1 - ((2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth))
        dice_loss = dice_loss.mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss



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
