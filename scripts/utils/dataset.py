import os
from PIL import Image
from torch.utils.data import Dataset

class PetClassificationDataset(Dataset):
    def __init__(self, image_dir, list_file, transform=None, train_only=True):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        with open(list_file, 'r') as f:
            for line in f:
                if line.startswith('#') or len(line.strip()) == 0:
                    continue  # Skip comments or empty lines
                parts = line.strip().split()
                if len(parts) != 4:
                    continue  # Skip malformed lines

                image_name, class_id, species, split = parts
                if train_only and int(split) != 1:
                    continue
                self.samples.append((image_name + ".jpg", int(class_id) - 1))  # Make label 0-based

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, label = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, image_name
