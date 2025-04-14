import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from utils.CAM import CAM, generate_and_save_cam
from utils.dataset import PetDataset

# # Data augmentation and normalization for training
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])
#
# # Load dataset
# data_root = 'oxford-iiit-pet'
# train_dataset = PetDataset(data_root, split='train', transform=transform)
# test_dataset = PetDataset(data_root, split='test', transform=transform)
#
# # Load model
# model = models.resnet50()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 37)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.load_state_dict(torch.load("Resnet50_FT.pth", map_location=device))
# model = model.to(device)
# model.eval()
#
# # Choose the target convolutional layer
# target_layer = model.layer4[2].conv3
#
# # Instantiate CAM
# grad_cam_pp = CAM(model, target_layer, device, mode='gradcam++')
# grad_cam = CAM(model, target_layer, device, mode='gradcam')
# cam = CAM(model, target_layer, device, mode='cam')
#
# # List models
# models = {
#     "cam": cam,
#     "grad_cam": grad_cam,
#     "grad_cam_pp": grad_cam_pp,
# }
#
# # Loop through each CAM, create a corresponding output directory, and process
# for name, model in models.items():
#     # Define the output directory based on the variable name
#     output_dir = os.path.join("output", "raw", name)
#     generate_and_save_cam(train_dataset, model, "train", output_dir)
#     generate_and_save_cam(train_dataset, model, "test", output_dir)
#     print(f"CAMs successfully generated for {name}, saved in {output_dir}")


def generate_cam():
    # Data augmentation and normalization for training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load dataset
    data_root = 'oxford-iiit-pet'
    train_dataset = PetDataset(data_root, split='train', transform=transform)
    test_dataset = PetDataset(data_root, split='test', transform=transform)

    # Load model
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 37)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("Resnet50_FT.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # Choose the target convolutional layer
    target_layer = model.layer4[2].conv3

    # Instantiate CAM
    grad_cam_pp = CAM(model, target_layer, device, mode='gradcam++')
    grad_cam = CAM(model, target_layer, device, mode='gradcam')
    cam = CAM(model, target_layer, device, mode='cam')

    # List models
    model_names = {
        "cam": cam,
        "grad_cam": grad_cam,
        "grad_cam_pp": grad_cam_pp,
    }

    # Loop through each CAM, create a corresponding output directory, and process
    for name, model in model_names.items():
        # Define the output directory based on the variable name
        output_dir = os.path.join("output", "raw", name)
        generate_and_save_cam(train_dataset, model, "train", output_dir)
        generate_and_save_cam(train_dataset, model, "test", output_dir)
        print(f"CAMs successfully generated for {name}, saved in {output_dir}")