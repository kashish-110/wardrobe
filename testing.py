import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
#import webcolors

# -------------------------------
# Step 1: Get dominant color
# -------------------------------
def get_clean_dominant_color(image, k=3):
    # Resize for speed
    image = cv2.resize(image, (120, 120))

    # Convert to HSV to filter noise
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Remove very dark pixels
    mask_bright = hsv[:, :, 2] > 40

    # Remove very low saturation (skin / gray)
    mask_sat = hsv[:, :, 1] > 40

    mask = mask_bright & mask_sat

    filtered_pixels = image[mask]

    if len(filtered_pixels) < 50:
        # fallback if too few pixels
        filtered_pixels = image.reshape(-1, 3)

    # KMeans on cleaned pixels
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(filtered_pixels)

    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]

    return dominant.astype(int)


# -------------------------------
# Step 2: Convert RGB to LAB
# -------------------------------
def rgb_to_lab(rgb_color):
    rgb_array = np.uint8([[rgb_color]])
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    return lab[0][0]


# -------------------------------
# Step 3: Load 140 CSS colors
# -------------------------------
import webcolors

def load_css_colors_lab():
    css_colors_lab = {}

    # Get all CSS3 color names
    css_names = webcolors.names("css3")

    for name in css_names:
        rgb = webcolors.name_to_rgb(name)
        lab = rgb_to_lab((rgb.red, rgb.green, rgb.blue))
        css_colors_lab[name] = lab

    return css_colors_lab


# -------------------------------
# Step 4: DeltaE distance
# -------------------------------
def delta_e(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)


# -------------------------------
# Step 5: Find closest CSS color
# -------------------------------
def get_closest_color_name(dominant_rgb, css_colors_lab):
    dominant_lab = rgb_to_lab(dominant_rgb)

    min_distance = float("inf")
    closest_color = None

    for name, lab in css_colors_lab.items():
        distance = delta_e(dominant_lab, lab)

        if distance < min_distance:
            min_distance = distance
            closest_color = name

    return closest_color


# -------------------------------
# Clothing Classes
# -------------------------------
class_names = [
    "short_sleeve_top", "long_sleeve_top",
    "short_sleeve_outwear", "long_sleeve_outwear",
    "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeve_dress",
    "long_sleeve_dress", "vest_dress", "sling_dress"
]

# -------------------------------
# Load Model
# -------------------------------
model = YOLO(r"C:\Users\Kashish Gupta\python\FashionMirror\perception\yolov8m_deepfashion_best.pt")

# Load CSS colors once
css_colors_lab = load_css_colors_lab()

# -------------------------------
# Test Image
# -------------------------------
image = cv2.imread(r"C:\Users\Kashish Gupta\python\FashionMirror\test_images\formal_man_001.png")
results = model(image)

for r in results:
    for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):

        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box)

        # Crop detected clothing
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        dominant_rgb = get_clean_dominant_color(crop_rgb)
        color_name = get_closest_color_name(dominant_rgb, css_colors_lab)

        clothing_name = class_names[int(cls_id)]

        print(f"{clothing_name} → {color_name}")