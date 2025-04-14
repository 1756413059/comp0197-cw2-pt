import os
import sys
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Add project root to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.config import IMAGE_DIR

# === 获取所有图像尺寸
sizes = []
for filename in os.listdir(IMAGE_DIR):
    if filename.endswith('.jpg'):
        img = Image.open(os.path.join(IMAGE_DIR, filename))
        sizes.append(img.size)

# === 分析尺寸
widths, heights = zip(*sizes)
avg_width = sum(widths) / len(widths)
avg_height = sum(heights) / len(heights)

print("📊 Image Size Summary")
print("-------------------------")
print(f"Min size:  {min(widths)} x {min(heights)}")
print(f"Max size:  {max(widths)} x {max(heights)}")
print(f"Avg size:  {avg_width:.2f} x {avg_height:.2f}")
print(f"Total images: {len(sizes)}")

# === 选择一张图像可视化 resize 效果
sample_image_path = os.path.join(IMAGE_DIR, 'Abyssinian_100.jpg')
image = Image.open(sample_image_path).convert('RGB')

resize_transform = transforms.Resize((224, 224))
resized_image = resize_transform(image)

# === 可视化原图 vs resized
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image)
plt.axis('off')
plt.text(0, -10, f"Size: {image.size}", fontsize=10)

plt.subplot(1, 2, 2)
plt.title("Resized (224x224)")
plt.imshow(resized_image)
plt.axis('off')
plt.text(0, -10, f"Size: {resized_image.size}", fontsize=10)

plt.tight_layout()
plt.show()
