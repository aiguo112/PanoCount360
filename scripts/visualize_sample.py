import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

# Project-relative paths (default sample id "1"; run from repo root or set PANOCOUNT_DATASET_ROOT)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.environ.get("PANOCOUNT_DATASET_ROOT", os.path.join(PROJECT_ROOT, "data", "dataset"))
SAMPLE_ID = os.environ.get("PANOCOUNT_SAMPLE_ID", "1")
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
density_path = os.path.join(DATASET_ROOT, "density_maps", SAMPLE_ID + ".npy")
json_path = os.path.join(DATASET_ROOT, "json", SAMPLE_ID + ".json")


def _find_image_path(images_dir, image_id):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = os.path.join(images_dir, image_id + ext)
        if os.path.exists(p):
            return p
    return os.path.join(images_dir, image_id + ".jpg")  # fallback for error message

# Load image
def load_image(path):
	return np.array(Image.open(path))

# Load density map
def load_density(path):
	return np.load(path)

# Load annotation
def load_annotation(path):
	with open(path, 'r') as f:
		return json.load(f)

# Visualization
def visualize(img, density, annotation):
	fig, axs = plt.subplots(1, 3, figsize=(18, 6))
	# Show image
	axs[0].imshow(img)
	axs[0].set_title('Image')
	axs[0].axis('off')
	# Show density map
	axs[1].imshow(density, cmap='jet')
	axs[1].set_title('Density Map')
	axs[1].axis('off')
	# Overlay points
	axs[2].imshow(img)
	points = annotation['points']
	xs = [p['x'] for p in points]
	ys = [p['y'] for p in points]
	axs[2].scatter(xs, ys, s=10, c='red', alpha=0.6)
	axs[2].set_title(f"Image with Annotations\nHuman Count: {annotation['human_num']}")
	axs[2].axis('off')
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	img_path = _find_image_path(IMAGES_DIR, SAMPLE_ID)
	if not os.path.exists(img_path):
		raise FileNotFoundError(f"Image not found for id={SAMPLE_ID} in {IMAGES_DIR}")
	img = load_image(img_path)
	density = load_density(density_path)
	annotation = load_annotation(json_path)
	visualize(img, density, annotation)
