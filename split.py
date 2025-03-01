import os
import shutil
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Paths (Modify these to match your dataset location)
dataset_path = "casia-webface"
train_path = "CassiaWebFace/train"
val_path = "CassiaWebFace/val"

# Ensure output directories exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Set train/val split ratio
train_ratio = 0.9

# Gather copy tasks using os.scandir for faster directory traversal
copy_tasks = []  # Each element is a tuple (source_path, destination_path)

with os.scandir(dataset_path) as subjects:
    for subject in subjects:
        if not subject.is_dir():
            continue
        subject_path = subject.path
        # List image file names in the subject folder
        images = [entry.name for entry in os.scandir(subject_path) if entry.is_file()]
        if not images:
            continue

        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create subject folders in train and val directories
        train_subject_path = os.path.join(train_path, subject.name)
        val_subject_path = os.path.join(val_path, subject.name)
        os.makedirs(train_subject_path, exist_ok=True)
        os.makedirs(val_subject_path, exist_ok=True)

        # Append copy tasks for training images
        for img in train_images:
            src = os.path.join(subject_path, img)
            dest = os.path.join(train_subject_path, img)
            copy_tasks.append((src, dest))
        # Append copy tasks for validation images
        for img in val_images:
            src = os.path.join(subject_path, img)
            dest = os.path.join(val_subject_path, img)
            copy_tasks.append((src, dest))

def copy_file(task):
    src, dest = task
    shutil.copy(src, dest)

# Use ThreadPoolExecutor to copy files concurrently
max_workers = 16  # Adjust the number of threads as needed
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(tqdm(executor.map(copy_file, copy_tasks), total=len(copy_tasks), desc="Copying images"))

print("Dataset split complete!")
