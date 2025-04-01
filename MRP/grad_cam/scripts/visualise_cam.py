import os
from PIL import Image, ImageDraw, ImageFont
from config import TRIMAP_DIR
import numpy as np
from torchvision import transforms

def create_comparison_grid(grad_cam_dir, suppress_cam_dir, output_path,trimap_dir=TRIMAP_DIR,plot_gt=False):
    # Get all JPGs from both folders (assuming matching filenames)
    grad_cam_files = sorted([f for f in os.listdir(grad_cam_dir) if f.lower().endswith('.jpg')])
    neg_grad_cam_files = sorted([f for f in os.listdir(suppress_cam_dir) if f.lower().endswith('.jpg')])
    
    # Verify equal number of images
    if len(grad_cam_files) != len(neg_grad_cam_files):
        print("Error: Folders contain different numbers of images!")
        return
    
    grad_cam_images = []
    for f in grad_cam_files:
        cam_img = Image.open(os.path.join(grad_cam_dir, f))
        class_id = f.split('.')[0]
        gt_mask_path = os.path.join(TRIMAP_DIR, f"{class_id}.png")
        if not os.path.exists(gt_mask_path):
            raise FileNotFoundError(f"Trimap not found: {gt_mask_path}")
        gt_mask = Image.open(gt_mask_path)
        # print("shape of cam_img: ", np.array(cam_img).shape)
        # print("shape of gt_mask: ", np.array(gt_mask).shape)
        # resize the cam_img to the same size as the gt_mask
        cam_img = transforms.Resize((gt_mask.size[1], gt_mask.size[0]))(cam_img)

        # where the gt_mask is 3, write the corresponding pixel in the image to white
        cam_img = np.array(cam_img)  # Convert to NumPy array (H, W, 3)
        gt_mask = np.array(gt_mask)  # Convert to NumPy array (H, W)
        # Set pixels to white where trimap == 3
        if plot_gt:
            cam_img[gt_mask == 3] = [255, 255, 255]
        # Convert back to PIL Image
        cam_img = Image.fromarray(cam_img)
        grad_cam_images.append(cam_img)

    neg_grad_cam_images = []
    for f in neg_grad_cam_files:
        cam_img = Image.open(os.path.join(suppress_cam_dir, f))
        class_id = f.split('.')[0]
        gt_mask_path = os.path.join(TRIMAP_DIR, f"{class_id}.png")
        if not os.path.exists(gt_mask_path):
            raise FileNotFoundError(f"Trimap not found: {gt_mask_path}")
        gt_mask = Image.open(gt_mask_path)
        cam_img = transforms.Resize((gt_mask.size[1], gt_mask.size[0]))(cam_img)

        # where the gt_mask is 3, write the corresponding pixel in the image to white
        cam_img = np.array(cam_img)  # Convert to NumPy array (H, W, 3)
        gt_mask = np.array(gt_mask)  # Convert to NumPy array (H, W)
        # Set pixels to white where trimap == 3
        if plot_gt:
            cam_img[gt_mask == 3] = [255, 255, 255]
        # Convert back to PIL Image
        cam_img = Image.fromarray(cam_img)
        neg_grad_cam_images.append(cam_img)

    
    # # Open all image pairs
    # grad_cam_images = [Image.open(os.path.join(grad_cam_dir, f)) for f in grad_cam_files]
    # neg_grad_cam_images = [Image.open(os.path.join(suppress_cam_dir, f)) for f in neg_grad_cam_files]
    
    # Get dimensions by summing
    total_width = sum(img.width for img in grad_cam_images)
    img_height = max(img.height for img in grad_cam_images)
    
    # Font settings (larger size)
    label_height = 60  # Increased space for labels
    try:
        font = ImageFont.truetype("arial.ttf", 60)  
    except:
        font = ImageFont.truetype("LiberationSans-Bold.ttf", 60)  
    finally:
        font = font or ImageFont.load_default().font_variant(size=60)

    # Create canvas (2 image rows + 2 label rows)
    result = Image.new('RGB', 
                     (total_width, 2 * img_height + 2 * label_height +40), 
                     color=(255, 255, 255))
    draw = ImageDraw.Draw(result)

    # ROW 1: grad-CAM images
    x_offset = 0
    for img in grad_cam_images:
        result.paste(img, (x_offset, label_height))
        x_offset += img.width
    
    # ROW 1 Label (BOTTOM of first row, just above second row)
    label1 = "grad-CAM"
    text_width = draw.textlength(label1, font=font)
    draw.text(
        ((total_width - text_width) // 2, label_height + img_height - 30),  # 30px above divider
        label1, fill="black", font=font
    )

    # ROW 2: neg-grad-cam images
    x_offset = 0
    for img in neg_grad_cam_images:
        result.paste(img, (x_offset, label_height + img_height + label_height))
        x_offset += img.width
    
    # ROW 2 Label (TRUE BOTTOM of the image)
    label2 = "suppressCAM"
    text_width = draw.textlength(label2, font=font)
    draw.text(
        ((total_width - text_width) // 2, label_height + 2*img_height+30),
        label2, fill="black", font=font
    )

    # Save result
    result.save(output_path)
    print(f"Saved labeled comparison to {output_path}")

# Usage
create_comparison_grid(
    grad_cam_dir="cam_examples/grad_cam",
    suppress_cam_dir="cam_examples/suppress_cam", 
    output_path="cam_examples/cam_comparison.jpg",
    trimap_dir=TRIMAP_DIR,
    plot_gt=False
)