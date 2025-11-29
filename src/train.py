import argparse
import os
from typing import Any
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from tqdm import tqdm

from data_processing.jester_dataset import JesterDataset
from models.CNN3D import C3D


def setup_ddp():
    """Initialize DDP using environment variables set by torchrun."""
    dist.init_process_group(backend="nccl")
    
    # Get rank information from environment variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def get_model(model_type, sample_size, sample_duration, num_classes):
    """Create model based on model type."""
    if model_type == "c3d":
        return C3D(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_checkpoint(checkpoint_dir, model, optimizer, rank, model_type, sample_size, num_frames, num_classes):
    """Load latest checkpoint if available and validate configuration."""
    if not os.path.exists(checkpoint_dir):
        return 1, None
        
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        return 1, None
    
    latest_ckpt = max(checkpoint_files, key=lambda x: int(x.split("epoch")[1].split(".")[0]))
    ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
    
    # Load checkpoint to CPU first
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Validate checkpoint configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        
        # Check model type
        if config.get('model_type') != model_type:
            raise ValueError(
                f"Checkpoint model type mismatch! "
                f"Checkpoint: {config.get('model_type')}, Requested: {model_type}"
            )
        
        # Check model parameters
        if config.get('sample_size') != sample_size:
            raise ValueError(
                f"Checkpoint sample_size mismatch! "
                f"Checkpoint: {config.get('sample_size')}, Requested: {sample_size}"
            )
        
        if config.get('num_frames') != num_frames:
            raise ValueError(
                f"Checkpoint num_frames mismatch! "
                f"Checkpoint: {config.get('num_frames')}, Requested: {num_frames}"
            )
        
        if config.get('num_classes') != num_classes:
            raise ValueError(
                f"Checkpoint num_classes mismatch! "
                f"Checkpoint: {config.get('num_classes')}, Requested: {num_classes}"
            )
    else:
        if rank == 0:
            print("Warning: Checkpoint has no config validation. This is an old checkpoint format.")
    
    # Extract metadata
    start_epoch = checkpoint['epoch'] + 1
    wandb_run_id = checkpoint.get('wandb_run_id', None)
    
    # Load model and optimizer state if provided
    if model is not None and optimizer is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if rank == 0:
            print(f"Resumed from checkpoint: {ckpt_path} (starting epoch {start_epoch})")
            if wandb_run_id:
                print(f"Will continue wandb run: {wandb_run_id}")
    
    # Delete checkpoint from memory immediately
    del checkpoint
    torch.cuda.empty_cache()
    
    return start_epoch, wandb_run_id


def train_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs, rank):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Only show progress bar on rank 0
    iterator = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} - Training", leave=False) if rank == 0 else loader
    
    for videos, labels in iterator:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * videos.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device, epoch, num_epochs, rank):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Only show progress bar on rank 0
    iterator = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} - Validation", leave=False) if rank == 0 else loader

    with torch.no_grad():
        for videos, labels in iterator:
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * videos.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train gesture recognition model with DDP (use torchrun)")

    # Model parameters
    parser.add_argument("--model_type", type=str, default="c3d", choices=["c3d"])
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of DataLoader workers per GPU")
    
    args = parser.parse_args()
    
    # Setup DDP - reads environment variables set by torchrun
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    # Print info only on rank 0
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"{'='*60}")
        print(f"World Size (GPUs):       {world_size}")
        print(f"Batch size per GPU:      {args.batch_size}")
        print(f"Effective batch size:    {args.batch_size * world_size}")
        print(f"Workers per GPU:         {args.num_workers}")
        print(f"Total workers:           {args.num_workers * world_size}")
        print(f"Learning rate:           {args.learning_rate}")
        print(f"Number of epochs:        {args.num_epochs}")
        print(f"{'='*60}\n")
    
    # Fixed parameters
    train_csv = "dataset/modified/annotations/train.csv"
    val_csv = "dataset/modified/annotations/val.csv"
    root_dir = "dataset/modified/data"
    num_classes = 9
    sample_size = 128
    num_frames = 32
    
    # Create checkpoint directory (only rank 0)
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Model with DDP
    model = get_model(args.model_type, sample_size, num_frames, num_classes)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    # Load checkpoint (only once, with model and optimizer ready)
    start_epoch, wandb_run_id = load_checkpoint(
        args.checkpoint_dir, 
        model, 
        optimizer, 
        rank, 
        args.model_type, 
        sample_size, 
        num_frames, 
        num_classes
    )
    use_wandb = False
    if rank == 0:
        try:
            import wandb
            
            # If resuming, continue the same run
            if start_epoch > 1 and wandb_run_id:
                wandb.init(
                    project="hand-gesture-recognition",
                    id=wandb_run_id,
                    resume="allow",
                    config={
                        'batch_size': args.batch_size,
                        'num_epochs': args.num_epochs,
                        'learning_rate': args.learning_rate,
                        'model_type': args.model_type,
                        'num_classes': num_classes,
                        'sample_size': sample_size,
                        'num_frames': num_frames,
                        'world_size': world_size,
                        'effective_batch_size': args.batch_size * world_size
                    }
                )
                print(f"Continuing existing wandb run: {wandb_run_id}")
            else:
                # New run
                wandb.init(
                    project="hand-gesture-recognition",
                    config={
                        'batch_size': args.batch_size,
                        'num_epochs': args.num_epochs,
                        'learning_rate': args.learning_rate,
                        'model_type': args.model_type,
                        'num_classes': num_classes,
                        'sample_size': sample_size,
                        'num_frames': num_frames,
                        'world_size': world_size,
                        'effective_batch_size': args.batch_size * world_size
                    }
                )
                wandb_run_id = wandb.run.id
                print(f"Started new wandb run: {wandb_run_id}")
            
            use_wandb = True
        except ImportError:
            print("wandb not available, continuing without logging")
    
    # Datasets with DistributedSampler
    train_dataset = JesterDataset(csv_file=train_csv, root_dir=root_dir, num_frames=num_frames, train=True)
    val_dataset = JesterDataset(csv_file=val_csv, root_dir=root_dir, num_frames=num_frames, train=False)
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True,
        seed=42
    )
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Free up memory before training starts
    torch.cuda.empty_cache()
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs + 1):
        # Set epoch for DistributedSampler to ensure different shuffling each epoch
        train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.num_epochs, rank)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, args.num_epochs, rank)
        
        # Aggregate metrics across all GPUs
        train_loss_tensor = torch.tensor([train_loss], device=device)
        train_acc_tensor = torch.tensor([train_acc], device=device)
        val_loss_tensor = torch.tensor([val_loss], device=device)
        val_acc_tensor = torch.tensor([val_acc], device=device)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
        
        train_loss = train_loss_tensor.item() / world_size
        train_acc = train_acc_tensor.item() / world_size
        val_loss = val_loss_tensor.item() / world_size
        val_acc = val_acc_tensor.item() / world_size
        
        # Only rank 0 logs and saves
        if rank == 0:
            print(f"Epoch [{epoch}/{args.num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'val/loss': val_loss,
                    'val/accuracy': val_acc
                })
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model_type}_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': {
                    'model_type': args.model_type,
                    'sample_size': sample_size,
                    'num_frames': num_frames,
                    'num_classes': num_classes,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate
                },
                'wandb_run_id': wandb_run_id if use_wandb else None
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}\n")
    
    if rank == 0 and use_wandb:
        wandb.finish()
    
    cleanup_ddp()


if __name__ == "__main__":
    main()