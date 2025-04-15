import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


# ======================
# Dataset Class
# ======================
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, pseudo_labels_dir, transform=None, mask_transform=None):
        """
        Args:
            images_dir (str): Path to folder with original images.
            pseudo_labels_dir (str): Path to folder with pseudo segmentation masks (CAMs).
            transform (callable, optional): Transform to apply to the original image.
            mask_transform (callable, optional): Transform to apply to the pseudo label.
                If not provided, a default transform that resizes to (256, 256) and converts to tensor is used.
        """
        self.images_dir = images_dir
        self.pseudo_labels_dir = pseudo_labels_dir
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        # For masks, we want to preserve their discrete nature, so we use NEAREST interpolation.
        self.mask_transform = mask_transform if mask_transform is not None else transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        # List image files (assumed to be .jpg or .png) that have a corresponding pseudo label.
        self.image_files = sorted([f for f in os.listdir(images_dir)
                                   if f.lower().endswith(('.jpg', '.png'))])
        self.image_files = [f for f in self.image_files
                            if os.path.exists(os.path.join(pseudo_labels_dir, os.path.splitext(f)[0] + ".png"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_filename)
        pseudo_label_path = os.path.join(self.pseudo_labels_dir, os.path.splitext(image_filename)[0] + ".png")

        # Load the original image as RGB.
        image = Image.open(image_path).convert("RGB")
        # Load the pseudo label (CAM) as a grayscale image.
        pseudo_label = Image.open(pseudo_label_path).convert("L")

        # Apply transforms to both image and mask.
        image = self.transform(image)
        # For the mask, we use our mask_transform.
        mask_tensor = self.mask_transform(pseudo_label)
        # The mask tensor is now in range [0, 1] (float values) but we want a binary mask.
        # Here, we threshold at 0.5.
        binary_mask = (mask_tensor >= 0.5).float()

        return image, binary_mask


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


def generate_and_save_segmentation(dataset, model, split, output_dir):
    """
    Processes a dataset to generate segmentation

    Args:
        dataset: A dataset instance; __getitem__ should return (image_tensor, label)
        model: trained Unet model
        split (str): A string indicating the dataset split ('train' or 'test')
        output_dir: directory for saving the outputs
    """
    output_dir = os.path.join(output_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(dataset)):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Retrieve the preprocessed image and label
        image_tensor, label = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device)

        # Run inference.
        with torch.no_grad():
            output = model(input_tensor)

        # Get original image dimensions
        original_filepath = dataset.data_info[idx]['image']
        with Image.open(original_filepath) as orig_img:
            original_width, original_height = orig_img.size

        # Upscale the predicted output to match the original image dimensions
        upscaled_output = F.interpolate(output, size=(original_height, original_width),
                                        mode='bilinear', align_corners=False)
        pred_mask_np = upscaled_output.squeeze().cpu().numpy()

        # Convert the mask to 0-255 uint8 values for visualization.
        pred_mask_img = (pred_mask_np * 255).astype(np.uint8)
        mask_image = Image.fromarray(pred_mask_img)

        # Use the original image's filename for saving the CAM
        original_filename = os.path.basename(original_filepath)
        filename = os.path.splitext(original_filename)[0] + ".png"

        # Save the image into the output directory
        output_path = os.path.join(output_dir, filename)
        mask_image.save(output_path)