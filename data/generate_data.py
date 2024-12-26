import os
import numpy as np
from PIL import Image

def generate_images(output_dir, num_classes=3, images_per_class=10, img_size=(64, 64)):
    """
    Generate synthetic images for classification.

    Args:
        output_dir (str): Directory to save generated images.
        num_classes (int): Number of classes.
        images_per_class (int): Number of images per class.
        img_size (tuple): Size of the images (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)
    for class_id in range(num_classes):
        class_dir = os.path.join(output_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)
        for img_id in range(images_per_class):
            # Create a random image
            img_array = np.random.randint(0, 256, img_size + (3,), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(class_dir, f"img_{img_id}.png"))
    print(f"Images generated in {output_dir}")

if __name__ == "__main__":
    generate_images("data/images", num_classes=3, images_per_class=50, img_size=(64, 64))
