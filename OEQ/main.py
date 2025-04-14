# main.py
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from generate_pseudo_labels import run_generate_pseudo_labels
from train_unet import run_train_unet
from evaluate_unet import run_evaluate

if __name__ == "__main__":
    print("Downlodaing Data...")
    _ = OxfordIIITPet(
        root="oxford-iiit-pet", split="trainval", target_types="segmentation",
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
        download=True
    )
    _ = OxfordIIITPet(
        root="oxford-iiit-pet", split="test", target_types="segmentation",
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
        download=True
    )

    run_generate_pseudo_labels()
    run_train_unet()
    run_evaluate()
