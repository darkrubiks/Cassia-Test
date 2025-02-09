import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG on CASIA-WebFace with resume and early stopping")
    parser.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint if available")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory to save/load checkpoints")
    args = parser.parse_args()
    return args

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if not files:
        return None
    epochs = []
    for f in files:
        try:
            epoch_str = f.split("_")[-1].replace(".pth", "")
            epoch_num = int(epoch_str)
            epochs.append((epoch_num, f))
        except Exception as e:
            continue
    if not epochs:
        return None
    latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
    return os.path.join(checkpoint_dir, latest_file), latest_epoch

def save_checkpoint(state, checkpoint_dir, filename):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def main():
    args = parse_args()

    # --------------------------------------
    # Hyperparameters & Settings
    # --------------------------------------
    #AdamW
    num_epochs = 1000
    batch_size = 128
    base_lr = 0.01            # for AdamW
    weight_decay = 5e-4
    warmup_epochs = 5
    T_max = num_epochs - warmup_epochs  # for cosine annealing after warmup

    """#SGD
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    step_size = 7
    gamma = 0.1
    """

    # Early stopping parameters
    early_stopping_patience = 5  # Number of epochs to wait without improvement
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
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    full_dataset = ImageFolder(root=dataset_dir, transform=train_transforms)
    num_classes = len(full_dataset.classes)
    print(f"Number of classes: {num_classes}")

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # --------------------------------------
    # Model Setup
    # --------------------------------------
    model = models.vgg16(weights=None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # --------------------------------------
    # Optimizer, Loss, Scheduler, and Mixed Precision
    # --------------------------------------
    #AdamW
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / T_max))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    """SGD
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    """


    scaler = GradScaler()

    # --------------------------------------
    # Resume Training (if applicable)
    # --------------------------------------
    start_epoch = 0
    if args.resume:
        checkpoint_info = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_info is not None:
            checkpoint_path, loaded_epoch = checkpoint_info
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
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
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        # Early Stopping: Check for improvement in validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered!")
                # Save final checkpoint before stopping.
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve
                }
                save_checkpoint(checkpoint, checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                return  # Stop training early

        # Save checkpoint after each epoch, including early stopping info.
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve
        }
        save_checkpoint(checkpoint, checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')

    print("Training complete.")

if __name__ == '__main__':
    main()
