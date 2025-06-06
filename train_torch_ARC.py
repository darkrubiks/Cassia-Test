import datetime
import csv
import os
import time
import warnings
import matplotlib.pyplot as plt   # <-- For plotting

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix

from NCNN import NCNN
from ArcFace import *

VAL_LOSS = 0.0

# ------------------------------------------------------------------
# Modified: Now return both training loss and top-1 training accuracy for the epoch
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image, target)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    
    # Compute average training loss and training accuracy (acc1)
    train_loss = metric_logger.meters["loss"].global_avg  
    train_acc = metric_logger.meters["acc1"].global_avg
    return train_loss, train_acc  

# ------------------------------------------------------------------
def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_data(traindir, args):
    print("Loading data")
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=args.crop_size,
                resize_size=args.resize_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
                mean=(0.5, 0.5, 0.5), # Changed here
                std=(0.5, 0.5, 0.5),
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)

    return dataset, train_sampler

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
    utils.init_distributed_mode(args)
    print(args)

    is_distributed = args.distributed
    rank = torch.distributed.get_rank() if is_distributed else 0

    # decide whether we (this process) should write metrics
    do_write = (rank == 0)

    device = torch.device(args.device)
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "ms1m-retinaface-images")
    dataset, train_sampler = load_data(train_dir, args)

    num_classes = len(dataset.classes)
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))
    else:
        collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print("Creating model")
    if args.model == "ncnn":
        backbone = NCNN()
        classifier = CosineClassifier(512, num_classes)
        arcface = CosFace()
        model = FaceModel(backbone, classifier, arcface)
    else:
        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return

    print("Start training")
    start_time = time.time()

    if do_write:
         # decide whether to append or overwrite
        is_resuming = bool(args.resume) or args.start_epoch > 0
        csv_path = os.path.join(args.output_dir, "metrics.csv")
        mode = "a" if is_resuming and os.path.exists(csv_path) else "w"
        csv_file = open(csv_path, mode, newline="")
        csv_writer = csv.writer(csv_file)
        if mode == "w":
            csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "test_acc"])

    # ------------------------------------------------------------------
    # Initialize live plotting using interactive mode and two subplots:
    plt.ion()   # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Subplot for Losses
    train_loss_line, = ax1.plot([], [], label="Training Loss", color="blue")
    val_loss_line,   = ax1.plot([], [], label="Validation Loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    # Subplot for Accuracies
    train_acc_line, = ax2.plot([], [], label="Training Accuracy", color="blue")
    test_acc_line,  = ax2.plot([], [], label="Validation Accuracy", color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    # ------------------------------------------------------------------

    # Initialize lists for live plotting
    train_losses = []
    val_losses = []
    train_accs = []
    test_accs = []

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = args.patience

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # Training step: capture training loss and accuracy for this epoch
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        lr_scheduler.step()
        # Evaluation step: obtain validation (test) accuracy and loss

        # CHANGED: Early stopping and checkpoint saving logic based on validation loss
        #if epoch > args.lr_warmup_epochs:
        #    if VAL_LOSS < best_val_loss:
        #        best_val_loss = VAL_LOSS
        #        epochs_no_improve = 0
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"checkpoint_{epoch}.pth"))
            print(f"Checkpoint saved at epoch {epoch} with loss {train_loss:.3f}")
        #    else:
        #        epochs_no_improve += 1
        #        print(f"Validation loss did not improve for {epochs_no_improve} epoch(s)")

        #if epochs_no_improve >= patience:
        #    print("Early stopping triggered")
        #    break
        # ------------------------------------------------------------------
        # Live plot update: update both subplots with new data
        current_epochs = list(range(args.start_epoch, epoch + 1))
        # Update Loss plot
        train_loss_line.set_data(current_epochs, train_losses)
        val_loss_line.set_data(current_epochs, val_losses)
        ax1.relim()
        ax1.autoscale_view()
        # Update Accuracy plot
        train_acc_line.set_data(current_epochs, train_accs)
        test_acc_line.set_data(current_epochs, test_accs)
        ax2.relim()
        ax2.autoscale_view()
        plt.draw()
        plt.pause(0.01)
        
        if do_write:
            csv_writer.writerow([epoch, train_loss, train_acc])
            csv_file.flush()
        # ------------------------------------------------------------------

    # Turn off interactive mode and display final plot
    plt.ioff()
    plt.savefig("training_plot.png")  # Save the plot as an image
    plt.show()

    if do_write:
        csv_file.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--norm-weight-decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--bias-weight-decay", default=None, type=float, help="weight decay for bias parameters of all layers (default: None, same value as --wd)")
    parser.add_argument("--transformer-embedding-decay", default=None, type=float, help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--patience", default=5, type=int, help="number of epochs to wait before early stopping")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true")
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model-ema-steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)")
    parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--resize-size", default=256, type=int, help="the resize size used for validation (default: 256)")
    parser.add_argument("--crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument("--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
