import os
import shutil
import random

from tqdm import tqdm

# Paths (Modify these to match your dataset location)
dataset_path = "casia-webface"
train_path = "CassiaWebFace/train"
val_path = "CassiaWebFace/val"

# Ensure output directories exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Set train/val split ratio
train_ratio = 0.9

# Iterate over subject folders
for subject in tqdm(os.listdir(dataset_path)):
    subject_path = os.path.join(dataset_path, subject)
    
    # Ensure it's a directory
    if not os.path.isdir(subject_path):
        continue
    
    # Get all images in the subject's folder
    images = os.listdir(subject_path)
    random.shuffle(images)  # Shuffle images randomly
    
    # Compute train/val split
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Create subject folders in train and val directories
    os.makedirs(os.path.join(train_path, subject), exist_ok=True)
    os.makedirs(os.path.join(val_path, subject), exist_ok=True)

    # Move images
    for img in train_images:
        shutil.copy(os.path.join(subject_path, img), os.path.join(train_path, subject, img))
    
    for img in val_images:
        shutil.copy(os.path.join(subject_path, img), os.path.join(val_path, subject, img))

print("Dataset split complete!")