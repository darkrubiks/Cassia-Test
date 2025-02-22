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

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Train NCNN on CASIA-WebFace with DDP, resume, and early stopping")
    parser.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint if available")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory to save/load checkpoints")
    # local_rank is passed automatically when launching with torchrun or the distributed launcher.
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for DistributedDataParallel")
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

def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    print(local_rank)
    # Initialize process group for DDP
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    

    torch.manual_seed(1234)

    # --------------------------------------
    # Hyperparameters & Settings
    # --------------------------------------
    num_epochs = 200
    batch_size = 128
    weight_decay = 5e-4
    learning_rate = 1e-2
    momentum = 0.9 

    # Early stopping parameters
    early_stopping_patience = 20  # Number of epochs to wait without improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0

    checkpoint_dir = args.checkpoint_dir

    # Path to your CASIA-WebFace dataset (organized as root/class/images)
    dataset_dir = 'casia-webface'  # UPDATE THIS PATH

    # Use GPU based on local_rank
    device = torch.device("cuda", local_rank)
    if rank == 0:
        print(f"Using device: {device} | World Size: {world_size}")

    # --------------------------------------
    # Data Preparation & Augmentation
    # --------------------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52026823, 0.40445255, 0.34655508],
                             std=[0.28127891, 0.24436931, 0.23583611]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52026823, 0.40445255, 0.34655508],
                             std=[0.28127891, 0.24436931, 0.23583611])
    ])

    full_dataset = ImageFolder(root=dataset_dir, transform=train_transforms)
    num_classes = len(full_dataset.classes)
    if rank == 0:
        print(f"Number of classes: {num_classes}")

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    # Create Distributed Samplers for training and validation
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=8, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=8, pin_memory=True)
    
    # --------------------------------------
    # Model Setup
    # --------------------------------------
    model = models.vgg16(weights=None, dropout=0.5)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


    # --------------------------------------
    # Optimizer, Loss, Scheduler, and Mixed Precision
    # --------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    #scaler = GradScaler()
    # --------------------------------------
    # Resume Training (if applicable)
    # --------------------------------------
    start_epoch = 0
    if args.resume:
        checkpoint_info = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_info is not None:
            if rank == 0:
                print(f"Resuming from checkpoint: {checkpoint_info}")
            checkpoint = torch.load(checkpoint_info, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            #scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            if rank == 0:
                print(f"Resumed at epoch {start_epoch}")
        else:
            if rank == 0:
                print("No checkpoint found. Starting training from scratch.")

    # --------------------------------------
    # Training Loop with Early Stopping
    # --------------------------------------
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)  # Shuffle data differently each epoch for DDP
        model.train()
        epoch_start = time.time()  # start time for epoch
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        # Only show progress bar on rank 0
        if rank == 0:
            loader = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        else:
            loader = train_loader

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
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

        # Aggregate training loss and accuracy across processes
        total_tensor = torch.tensor(total, device=device, dtype=torch.float32)
        running_loss_tensor = torch.tensor(running_loss, device=device)
        correct_tensor = torch.tensor(correct, device=device, dtype=torch.float32)

        dist.barrier()
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)

        global_total = total_tensor.item()
        train_loss = running_loss_tensor.item() / global_total
        train_acc = 100. * correct_tensor.item() / global_total

        # Only show progress bar on rank 0
        if rank == 0:
            loader = tqdm(val_loader, desc=f"Test Epoch {epoch+1}")
        else:
            loader = val_loader

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Aggregate training loss and accuracy across processes
        total_tensor = torch.tensor(total_val, device=device, dtype=torch.float32)
        running_loss_tensor = torch.tensor(val_loss, device=device)
        correct_tensor = torch.tensor(correct_val, device=device, dtype=torch.float32)

        dist.barrier()
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)

        global_total = total_tensor.item()
        val_loss = running_loss_tensor.item() / global_total
        val_acc = 100. * correct_tensor.item() / global_total

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                    f"LR: {current_lr:.6f} | {batches_per_sec:.2f} batches/s, {images_per_sec:.2f} images/s")

        # Early Stopping: Check for improvement in validation loss (only rank 0)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),  # save the underlying model
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    #'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve
                }
            if rank == 0:
                save_checkpoint(checkpoint, checkpoint_dir, f'checkpoint_epoch.pth')
                print(f"Checkpoint saved at {checkpoint_dir}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered!")
                # Optionally save final checkpoint here
                break

    if rank == 0:
        print("Training complete.")
    dist.destroy_process_group()

if __name__ == '__main__':
    main()