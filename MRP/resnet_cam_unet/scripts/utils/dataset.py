import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PetClassificationDataset(Dataset):
    """
    Dataset for pet classification using the Oxford-IIIT Pet dataset.

    Each sample returns:
        - A preprocessed RGB image (Tensor)
        - A class label in [0, 36] (int)
        - The original image filename (str)

    Args:
        image_dir (str): Path to the directory containing image files.
        list_file (str): Path to the .txt file listing image names and labels.
        transform (callable, optional): Optional image transform (e.g., torchvision transforms).
    """

    def __init__(self, image_dir, list_file, transform=None):
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
                    continue  

                image_name, class_id, species, breed_id = parts
                
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
    """
    Dataset for pet segmentation using pseudo masks or ground truth trimaps.

    Each sample returns:
        - A preprocessed RGB image (Tensor), shape [3, 224, 224]
        - A binary mask (Tensor), shape [1, 224, 224] with values {0.0, 1.0}

    Args:
        image_dir (str): Path to images.
        mask_dir (str): Path to masks (e.g., pseudo_masks/ or trimaps/).
        list_file (str): Path to .txt file with image IDs and splits.
        transform (callable, optional): Optional transform (currently unused; handled manually).
    """

    def __init__(self, image_dir, mask_dir, list_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                image_name, _, _, split = parts
                if int(split) == 1:  
                    mask_name = image_name + '_mask.png'
                    self.samples.append((image_name + '.jpg', mask_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  

        image = TF.resize(image, (224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        mask = TF.resize(mask, (224, 224))
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()  

        return image, mask

class PetDataset(Dataset):
    """
    Minimal dataset for inference only (e.g., for CAM or segmentation prediction).

    Each sample returns:
        - A preprocessed RGB image (Tensor), shape [3, 224, 224]

    Args:
        image_dir (str): Path to image folder.
        list_file (str): Path to list file (.txt).
        transform (callable, optional): Optional image preprocessing transform.
    """
    def __init__(self, image_dir, list_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                image_name, class_id, _, split = parts

                self.samples.append((image_name + ".jpg", int(class_id) - 1))  


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, _ = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert('RGB')

        image = TF.resize(image, (224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        return image