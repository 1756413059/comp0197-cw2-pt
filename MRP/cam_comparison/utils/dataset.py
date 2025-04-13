import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image


class PetDataset(Dataset):
    def __init__(self, data_root, split, transform=None):
        """
        Args:
            data_root (str): Path to the main dataset folder.
            split (str): 'train' or 'test' specifying the dataset split.
            transform (callable, optional): Torchvision transforms to apply.
        """
        self.transform = transform
        self.data_info = []
        annotations_dir = os.path.join(data_root, 'annotations')
        images_folder = os.path.join(data_root, 'images')
        list_file = os.path.join(annotations_dir, 'list.txt')
        trainval_file = os.path.join(annotations_dir, 'trainval.txt')
        test_file = os.path.join(annotations_dir, 'test.txt')
        xmls_folder = os.path.join(annotations_dir, 'xmls')

        # Load image-level labels from list.txt if available.
        image_info = {}
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 4:
                            image_id, class_id, species, breed_id = parts[:4]
                            image_info[image_id] = {
                                'class_id': class_id,
                                'species': species,
                                'breed_id': breed_id
                            }

        # Select the appropriate split file.
        split_file = trainval_file if split == 'train' else test_file

        # Read image names from the split file.
        image_names = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    image_names.append(line.split()[0])

        # Build the dataset information list.
        for name in image_names:
            img_filename = name + '.jpg'
            img_file = os.path.join(images_folder, img_filename)
            xml_file = os.path.join(xmls_folder, name + '.xml')

            # Parse the XML file for bounding box information if it exists.
            bbox = None
            if os.path.exists(xml_file):
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    bbox = []
                    for obj in root.findall('object'):
                        bndbox = obj.find('bndbox')
                        if bndbox is not None:
                            try:
                                xmin = int(bndbox.find('xmin').text)
                                ymin = int(bndbox.find('ymin').text)
                                xmax = int(bndbox.find('xmax').text)
                                ymax = int(bndbox.find('ymax').text)
                                bbox.append((xmin, ymin, xmax, ymax))
                            except Exception as e:
                                print(f"Error parsing {xml_file}: {e}")
                except Exception as e:
                    print(f"Error parsing XML file {xml_file}: {e}")

            # Use the parsed label from list.txt if available; otherwise, extract from the filename.
            if name in image_info:
                label = image_info[name]
            else:
                # Assume filename format: <image_id> <CLASS-ID> <SPECIES> <BREED-ID>
                parts = os.path.splitext(img_filename)[0].split()
                if len(parts) < 4:
                    label = None
                else:
                    image_id = " ".join(parts[:-3])
                    class_id, species, breed_id = parts[-3], parts[-2], parts[-1]
                    label = {
                        'image_id': image_id,
                        'class_id': class_id,
                        'species': species,
                        'breed_id': breed_id
                    }

            self.data_info.append({
                'image': img_file,
                'bbox': bbox,
                'label': label
            })

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        sample = self.data_info[idx]
        image = Image.open(sample['image']).convert("RGB")
        # Convert class_id to a zero-indexed integer label.
        label = int(sample['label']['class_id']) - 1
        if self.transform:
            image = self.transform(image)
        return image, label