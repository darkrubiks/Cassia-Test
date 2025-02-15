import os
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from NCNN import NCNN  # Your custom model

def parse_args():
    parser = argparse.ArgumentParser(description="Train NCNN on CASIA-WebFace with DDP, resume, and early stopping")
    parser.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint if available")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory to save/load checkpoints")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for DistributedDataParallel")
    args = parser.parse_args()
    return args

def init_distributed(args):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    return device

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
    if dist.get_rank() == 0:
        print(f"Checkpoint saved at {checkpoint_path}")

def main():
    args = parse_args()
    device = init_distributed(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(1234)
    
    # --------------------------------------
    # Hyperparameters & Settings
    # --------------------------------------
    num_epochs = 1000
    batch_size = 128
    weight_decay = 5e-4
    learning_rate = 1e-2
    momentum = 0.9 
    T_0 = 5

    early_stopping_patience = 20  # Number of epochs to wait without improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0

    checkpoint_dir = args.checkpoint_dir
    dataset_dir = 'casia-webface'  # UPDATE THIS PATH

    if rank == 0:
        print(f"Rank {rank} | Using device: {device} | World Size: {world_size}")

    # --------------------------------------
    # Data Preparation & Augmentation
    # --------------------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    full_dataset = ImageFolder(root=dataset_dir, transform=train_transforms)
    num_classes = len(full_dataset.classes)
    if rank == 0:
        print(f"Number of classes: {num_classes}")

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    # For training, use DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True)
    # For validation, only rank 0 runs it using a standard (non-distributed) DataLoader.
    if rank == 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
    else:
        val_loader = None

    # --------------------------------------
    # Model Setup
    # --------------------------------------
    model = NCNN()
    in_features = model.output.in_features
    model.output = nn.Linear(in_features, num_classes)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # --------------------------------------
    # Optimizer, Loss, Scheduler, and Mixed Precision Setup
    # --------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=1e-5)
    scaler = GradScaler()

    # --------------------------------------
    # Resume Training (if applicable)
    # Only rank 0 loads the checkpoint, then all processes sync
    # --------------------------------------
    start_epoch = 0
    if args.resume and rank == 0:
        checkpoint_info = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_info is not None:
            checkpoint_path, _ = checkpoint_info
            print(f"Rank {rank}: Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Rank {rank}: Resumed at epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting training from scratch.")
    dist.barrier()  # Sync all processes

    # --------------------------------------
    # Training Loop with Early Stopping and Batch/s Metrics
    # --------------------------------------
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for images, labels in tqdm(train_loader, desc=f"Rank {rank} Epoch {epoch+1} training", disable=(rank != 0)):
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
            num_batches += 1

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        batches_per_sec = num_batches / epoch_time
        images_per_sec = (num_batches * batch_size) / epoch_time

        # Aggregate training metrics across processes
        total_tensor = torch.tensor(total, device=device, dtype=torch.float32)
        running_loss_tensor = torch.tensor(running_loss, device=device)
        correct_tensor = torch.tensor(correct, device=device, dtype=torch.float32)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        global_total = total_tensor.item()
        train_loss = running_loss_tensor.item() / global_total
        train_acc = 100. * correct_tensor.item() / global_total

        # Only rank 0 performs validation
        if rank == 0 and val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Rank {rank} Epoch {epoch+1} validation"):
                    images, labels = images.to(device), labels.to(device)
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            val_loss /= total_val
            val_acc = 100. * correct_val / total_val

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{num_epochs}]: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f} | {batches_per_sec:.2f} batches/s, {images_per_sec:.2f} images/s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve
                }
                save_checkpoint(checkpoint, checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered!")
                    # Optionally save final checkpoint here
                    break
        else:
            scheduler.step()

        dist.barrier()  # synchronize all processes

    if rank == 0:
        print("Training complete.")
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
