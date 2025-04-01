import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# #Image CLASS-ID SPECIES BREED-ID
# #ID: 1:37 Class ids
# #SPECIES: 1:Cat 2:Dog
# #BREED ID: 1-25:Cat 1:12:Dog
# #All images with 1st letter as captial are cat images
# #images with small first letter are dog images
# Abyssinian_100 1 1 1
# Abyssinian_101 1 1 1
# Abyssinian_102 1 1 1
# Abyssinian_103 1 1 1

class PetClassificationDataset(Dataset):
    def __init__(
            self, 
            image_dir, 
            list_file, 
            transform=None, 
            # train_only=True
        ):
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

                image_name, class_id, species, breed_id = parts
                
                # if train_only and int(split) != 1:
                #     continue

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
                image_name, _, _, _ = parts
                # if int(split) == 1:  # 只选训练集
                #     mask_name = image_name + '_mask.png'
                #     self.samples.append((image_name + '.jpg', mask_name))
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