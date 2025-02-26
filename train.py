import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from NCNN import NCNN

def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG on CASIA-WebFace with resume and early stopping")
    parser.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint if available")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory to save/load checkpoints")
    parser.add_argument("--model", type=str, default="vgg16", help="Model to use for training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--opt", type=str, default="sgd", help="Optimizer to use for training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for training")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay for training")
    args = parser.parse_args()
    return args

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    return os.path.join(checkpoint_dir, "checkpoint_epoch.pth")

def save_checkpoint(state, checkpoint_dir, filename):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def main():
    torch.manual_seed(1234)
    args = parse_args()
    model_name = args.model

    # --------------------------------------
    # Hyperparameters & Settings
    # --------------------------------------
    torch.manual_seed(1234)

    num_epochs = args.epochs
    batch_size = args.batch_size
    weight_decay = args.wd
    learning_rate = args.lr
    momentum = 0.9 
    train_split = 0.85

    # Early stopping parameters
    early_stopping_patience = 20  # Number of epochs to wait without improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0

    checkpoint_dir = args.checkpoint_dir

    # Path to your CASIA-WebFace dataset (organized as root/class/images)
    dataset_dir = 'casia-webface'  # UPDATE THIS PATH

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------
    # Data Preparation & Augmentation
    # --------------------------------------
    img_size = 224 if model_name == "vgg16" else 120

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52026823, 0.40445255, 0.34655508],
                             std=[0.28127891, 0.24436931, 0.23583611]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52026823, 0.40445255, 0.34655508],
                             std=[0.28127891, 0.24436931, 0.23583611])
    ])

    full_dataset = ImageFolder(root=dataset_dir, transform=train_transforms)
    num_classes = len(full_dataset.classes)
    print(f"Number of classes: {num_classes}")

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # --------------------------------------
    # Model Setup
    # --------------------------------------
    if model_name == "vgg16":
        model = models.vgg16(weights=None, num_classes=num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None, num_classes=num_classes)
    elif model_name == "vit":
        model = models.vit_b_16(weights=None, num_classes=num_classes)
    elif model_name == "ncnn":
        model = NCNN()
        in_features = model.output.in_features
        model.output = nn.Linear(in_features, num_classes)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        
    model = model.to(device)

    # --------------------------------------
    # Optimizer, Loss, Scheduler, and Mixed Precision
    # --------------------------------------
    if args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                            momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    #scaler = GradScaler()
    # --------------------------------------
    # Resume Training (if applicable)
    # --------------------------------------
    start_epoch = 0
    if args.resume:
        checkpoint_info = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_info is not None:
            checkpoint_path = checkpoint_info
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            #scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            print(f"Resumed at epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting training from scratch.")

    # --------------------------------------
    # Training Loop with Early Stopping
    # --------------------------------------
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_start = time.time()  # start time for epoch
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            #with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            num_batches += 1

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        batches_per_sec = num_batches / epoch_time
        images_per_sec = (num_batches * batch_size) / epoch_time

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / total_val
        val_acc = 100. * correct_val / total_val

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | {batches_per_sec:.2f} batches/s, {images_per_sec:.2f} images/s")

         # Early Stopping: Check for improvement in validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    #'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve
                }
            save_checkpoint(checkpoint, checkpoint_dir, f'checkpoint_epoch.pth')
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered!")
                # Save final checkpoint before stopping.
                
                return  # Stop training early

    print("Training complete.")

if __name__ == '__main__':
    main()