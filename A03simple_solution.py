import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm.auto import tqdm


# This is a simple example of a diffusion model in 1D.


def build_parser():
    parser = argparse.ArgumentParser(description="Train a 1D DDPM on a two-Gaussian mixture.")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser


# We will keep these parameters fixed throughout.
# These parameters should give you an acceptable result, but feel free to play with them.
TIME_STEPS = 250
BETA = 0.02
N_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.8e-4


def make_dataset():
    """Generate a 1D data set from a mixture of two Gaussians."""
    data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
        torch.distributions.Categorical(torch.tensor([1.0, 2.0])),
        torch.distributions.Normal(torch.tensor([-4.0, 4.0]), torch.tensor([1.0, 1.0])),
    )
    dataset = data_distribution.sample(torch.Size([10000]))
    dataset_validation = data_distribution.sample(torch.Size([1000]))
    return dataset, dataset_validation


def make_schedule(device):
    betas = torch.full((TIME_STEPS,), BETA, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def make_network():
    # The network receives the noised sample x_t and the normalized time step t / T.
    return torch.nn.Sequential(
        torch.nn.Linear(2, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 1),
    )


def predict_noise(g, x_t, t_index):
    t_scaled = (t_index.float() + 1.0) / TIME_STEPS
    model_input = torch.stack((x_t, t_scaled), dim=1)
    return g(model_input).squeeze(1)


def ddpm_training_loss(g, x0, alpha_bars):
    t_index = torch.randint(0, TIME_STEPS, (x0.shape[0],), device=x0.device)
    epsilon = torch.randn_like(x0)

    alpha_bar_t = alpha_bars[t_index]
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * epsilon

    epsilon_pred = predict_noise(g, x_t, t_index)
    return torch.nn.functional.mse_loss(epsilon_pred, epsilon)


def train(g, dataset, dataset_validation, alpha_bars, n_epochs, batch_size):
    optimizer = torch.optim.Adam(g.parameters(), lr=LEARNING_RATE)
    train_losses = []
    validation_losses = []

    epochs = tqdm(range(n_epochs))
    for _ in epochs:
        g.train()
        indices = torch.randperm(dataset.shape[0], device=dataset.device)
        shuffled_dataset = dataset[indices]
        batch_losses = []

        for i in range(0, shuffled_dataset.shape[0] - batch_size + 1, batch_size):
            x0 = shuffled_dataset[i:i + batch_size]

            # Algorithm 1 from DDPM: choose t, add closed-form noise, predict that noise.
            loss = ddpm_training_loss(g, x0, alpha_bars)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        g.eval()
        with torch.no_grad():
            validation_loss = ddpm_training_loss(g, dataset_validation, alpha_bars)

        train_loss = float(np.mean(batch_losses))
        train_losses.append(train_loss)
        validation_losses.append(validation_loss.item())
        epochs.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{validation_loss.item():.4f}")

    return train_losses, validation_losses


def sample_reverse(g, count, betas, alphas, alpha_bars, device, history_steps=None):
    """
    Sample from the model by applying the reverse diffusion process.

    This implements algorithm 2 of the DDPM paper (https://arxiv.org/abs/2006.11239).
    """
    g.eval()
    x = torch.randn(count, device=device)

    if history_steps is None:
        history_steps = []
    history_steps = set(history_steps)
    history = {}
    if TIME_STEPS in history_steps:
        history[TIME_STEPS] = x.detach().cpu()

    with torch.no_grad():
        for t in reversed(range(TIME_STEPS)):
            t_index = torch.full((count,), t, device=device, dtype=torch.long)
            epsilon_pred = predict_noise(g, x, t_index)

            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]
            mean = (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * epsilon_pred) / torch.sqrt(alpha_t)

            if t > 0:
                z = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * z
            else:
                x = mean

            if t in history_steps:
                history[t] = x.detach().cpu()

    return x, history


def plot_training_distribution(dataset, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.kdeplot(dataset.cpu().numpy(), ax=ax, color="blue", linewidth=2)
    ax.set_title("Training data distribution")
    ax.set_xlabel("Sample value")
    ax.set_ylabel("Probability Density")
    fig.tight_layout()
    fig.savefig(output_dir / "training_distribution.png", dpi=180)
    plt.close(fig)


def plot_losses(train_losses, validation_losses, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(train_losses, label="Training loss")
    ax.plot(validation_losses, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Noise-prediction loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close(fig)


def plot_forward_diffusion(dataset, alpha_bars, device, output_dir):
    levels = [0, 25, 50, 100, 150, 250]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)

    x0 = dataset[:5000].to(device)
    for ax, t in zip(axes.ravel(), levels):
        if t == 0:
            x_t = x0
        else:
            alpha_bar_t = alpha_bars[t - 1]
            x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * torch.randn_like(x0)

        sns.histplot(x_t.detach().cpu().numpy(), ax=ax, bins=60, stat="density", color="steelblue")
        ax.set_title(f"Forward step t={t}")
        ax.set_xlim(-10, 10)

    fig.suptitle("Data transformed by increasing forward diffusion steps")
    fig.tight_layout()
    fig.savefig(output_dir / "forward_diffusion_steps.png", dpi=180)
    plt.close(fig)


def plot_reverse_diffusion(history, output_dir):
    levels = [250, 200, 150, 100, 50, 0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)

    for ax, t in zip(axes.ravel(), levels):
        samples = history[t].numpy()
        sns.histplot(samples, ax=ax, bins=60, stat="density", color="crimson")
        ax.set_title(f"Reverse state x_{t}")
        ax.set_xlim(-10, 10)

    fig.suptitle("Noise transformed into samples by the learned reverse process")
    fig.tight_layout()
    fig.savefig(output_dir / "reverse_diffusion_steps.png", dpi=180)
    plt.close(fig)


def plot_final_samples(dataset, samples, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    bins = np.linspace(-10, 10, 50)
    sns.kdeplot(dataset.cpu().numpy(), ax=ax, color="blue", label="True distribution", linewidth=2)
    sns.histplot(samples, ax=ax, bins=bins, color="red", label="Sampled distribution", stat="density", alpha=0.45)
    ax.legend()
    ax.set_xlabel("Sample value")
    ax.set_ylabel("Density")
    ax.set_title("True data distribution vs. generated samples")
    fig.tight_layout()
    fig.savefig(output_dir / "learned_distribution.png", dpi=180)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, dataset_validation = make_dataset()
    dataset = dataset.to(device)
    dataset_validation = dataset_validation.to(device)
    betas, alphas, alpha_bars = make_schedule(device)

    g = make_network().to(device)
    train_losses, validation_losses = train(
        g,
        dataset,
        dataset_validation,
        alpha_bars,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    history_steps = [250, 200, 150, 100, 50, 0]
    samples, history = sample_reverse(
        g,
        args.samples,
        betas,
        alphas,
        alpha_bars,
        device,
        history_steps=history_steps,
    )
    samples = samples.detach().cpu().numpy()

    plot_training_distribution(dataset, args.output_dir)
    plot_losses(train_losses, validation_losses, args.output_dir)
    plot_forward_diffusion(dataset, alpha_bars, device, args.output_dir)
    plot_reverse_diffusion(history, args.output_dir)
    plot_final_samples(dataset, samples, args.output_dir)

    print(f"Saved figures to {args.output_dir}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {validation_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
