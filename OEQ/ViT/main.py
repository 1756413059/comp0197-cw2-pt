# main.py
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from generate_pseudo_labels import run_generate_pseudo_labels
from vit_pipeline import run_vit_pipeline

if __name__ == "__main__":
    print("Generating pseudo labels using ViT...")
    run_generate_pseudo_labels()

    print("Running ViT pipeline (training & evaluation)...")
    run_vit_pipeline() 