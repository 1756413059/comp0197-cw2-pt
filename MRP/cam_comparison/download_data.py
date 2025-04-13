from torchvision.datasets import OxfordIIITPet

dataset = OxfordIIITPet(
    root='.', split='trainval', target_types='segmentation', download=True
)