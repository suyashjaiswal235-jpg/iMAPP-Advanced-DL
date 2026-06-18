import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from denoising_diffusion_pytorch import GaussianDiffusion, Unet


# Hyperparameters from the template.
LEARNING_RATE = 4e-4
BATCH_SIZE = 128
N_EPOCHS = 100
IMAGE_SIZE = 28
TIME_STEPS = 1000
SAMPLING_TIMESTEPS = 250
DIM = 32
DIM_MULTS = (1, 2, 5)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(data_dir: str, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    # ToTensor keeps MNIST pixel amplitudes in [0, 1], as requested in the assignment hint.
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    print("loading MNIST digits dataset")
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=mnist_transform,
        download=True,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=mnist_transform,
        download=True,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def build_diffusion(device: torch.device) -> GaussianDiffusion:
    model = Unet(
        dim=DIM,
        dim_mults=DIM_MULTS,
        flash_attn=False,
        channels=1,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=IMAGE_SIZE,
        timesteps=TIME_STEPS,
        sampling_timesteps=SAMPLING_TIMESTEPS,
    )
    return diffusion.to(device)


@torch.no_grad()
def save_samples(diffusion: GaussianDiffusion, output_dir: Path, epoch: int, batch_size: int) -> None:
    diffusion.eval()
    samples = diffusion.sample(batch_size=batch_size)
    samples = samples.clamp(0.0, 1.0)
    nrow = max(1, int(math.sqrt(batch_size)))
    save_image(samples, output_dir / f"samples_epoch_{epoch:03d}.png", nrow=nrow)


@torch.no_grad()
def evaluate_loss(diffusion: GaussianDiffusion, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    diffusion.eval()
    total_loss = 0.0
    n_batches = 0

    for images, _labels in loader:
        images = images.to(device)
        loss = diffusion(images)
        total_loss += float(loss.item())
        n_batches += 1
        if n_batches >= max_batches:
            break

    return total_loss / max(1, n_batches)


def save_checkpoint(
    diffusion: GaussianDiffusion,
    optimizer: optim.Optimizer,
    output_dir: Path,
    epoch: int,
    global_step: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": diffusion.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "image_size": IMAGE_SIZE,
        "timesteps": TIME_STEPS,
        "sampling_timesteps": SAMPLING_TIMESTEPS,
        "dim": DIM,
        "dim_mults": DIM_MULTS,
    }
    torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save(checkpoint, output_dir / "checkpoint_latest.pt")


def load_checkpoint(diffusion: GaussianDiffusion, optimizer: optim.Optimizer, path: str, device: torch.device) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    diffusion.model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint.get("global_step", 0))
    print(f"resumed from {path} at epoch {start_epoch}")
    return start_epoch, global_step


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = build_loaders(args.data_dir, args.batch_size, args.num_workers)
    diffusion = build_diffusion(device)
    optimizer = optim.AdamW(diffusion.parameters(), lr=args.learning_rate)

    start_epoch = 1
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(diffusion, optimizer, args.resume, device)

    for epoch in range(start_epoch, args.epochs + 1):
        diffusion.train()
        running_loss = 0.0

        for batch_idx, (images, _labels) in enumerate(train_loader, start=1):
            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = diffusion(images)
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            running_loss += float(loss.item())

            if batch_idx % args.log_every == 0 or batch_idx == len(train_loader):
                avg_loss = running_loss / batch_idx
                print(
                    f"epoch {epoch:03d}/{args.epochs:03d} "
                    f"batch {batch_idx:04d}/{len(train_loader):04d} "
                    f"train_loss={avg_loss:.4f}"
                )

        val_loss = evaluate_loss(diffusion, test_loader, device, args.eval_batches)
        print(f"epoch {epoch:03d} validation_loss={val_loss:.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_samples(diffusion, output_dir, epoch, args.num_samples)

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(diffusion, optimizer, output_dir, epoch, global_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPM on MNIST digits.")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="runs/mnist_ddpm")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
