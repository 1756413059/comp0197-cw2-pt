import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PetClassificationDataset(Dataset):
    def __init__(self, image_dir, list_file, transform=None, train_only=True):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        with open(list_file, 'r') as f:
            for line in f:
                if line.startswith('#') or len(line.strip()) == 0:
                    # Skip comments or empty lines
                    continue  
                parts = line.strip().split()
                if len(parts) != 4:
                    # Skip malformed lines
                    continue  

                image_name, class_id, species, split = parts
                
                if train_only and int(split) != 1:
                    continue

                # Make label 0-based
                self.samples.append((image_name + ".jpg", int(class_id) - 1))  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, label = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, image_name

class PetSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, list_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 读取 list.txt（只取训练集）
        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                image_name, _, _, split = parts
                if int(split) == 1:  # 只选训练集
                    mask_name = image_name + '_mask.png'
                    self.samples.append((image_name + '.jpg', mask_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 灰度图（0/255）

        # 转为 tensor 并归一化（和分类时一样）
        image = TF.resize(image, (224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        # mask: 转为 tensor 并将 255 → 1.0
        mask = TF.resize(mask, (224, 224))
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()  # 转为 0/1 单通道 mask

        return image, mask
    
class GTMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, list_file, split=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = []

        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                # Handle trainval.txt (4 columns) or test.txt (3 columns)
                if len(parts) == 4:
                    image_name, _, _, set_id = parts
                    if split is None or int(set_id) == split:
                        self.image_names.append(image_name)
                elif len(parts) == 3 and split is None:  # test.txt
                    image_name = parts[0]
                    self.image_names.append(image_name)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{image_name}_mask.png")

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Resize and normalize
        image = TF.resize(image, (224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        mask = TF.resize(mask, (224, 224), interpolation=Image.NEAREST)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()  # Convert binary mask from [0, 255] to [0, 1]

        return image, mask, image_name