from train_classifier import train_classifier
from generate_cam import generate_cam
from train_segmentor import train_segmentor
from generate_evaluate_segmentor import generate_evaluate_segmentor
from torchvision.datasets import OxfordIIITPet


def run_experiment():
    print("Downloading dataset......")
    data = OxfordIIITPet(
        root='.', split='trainval', target_types='segmentation', download=True
    )
    print("Training Classifier......")
    train_classifier()
    print("Generating CAM masks for all CAM methods......")
    generate_cam()
    print("Training Segmentor for each CAM method......")
    train_segmentor()
    print("Generating Pseudo Maks and Evaluating......")
    generate_evaluate_segmentor()