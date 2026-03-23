import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
class CVAE(nn.Module):
    """
    Conditional VAE for tabular features.

    Encoder: [x, c] -> (mu, logvar)
    Reparam: z = mu + exp(0.5*logvar) * eps
    Decoder: [z, c] -> x_hat
    """
    def __init__(self, x_dim: int, c_dim: int, z_dim: int, hidden: int):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + c_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + c_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim), # linear output for MSE/Gaussian recon
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        xc = torch.cat([x, c], dim=1)
        h = self.enc(xc)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        eps = torch.randn_like(mu)
        sigma = torch.exp(0.5 * logvar)
        return mu + sigma * eps

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        zc = torch.cat([z, c], dim=1)
        return self.dec(zc)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar


def elbo_loss(x, x_hat, mu, logvar, beta: float):
    recon = ((x_hat - x) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1.0 + logvar - mu**2 - torch.exp(logvar)).sum(dim=1)).mean()
    total = recon + beta * kl
    return total, recon, kl


@torch.no_grad()
def evaluate_cvae(model, loader, device, beta: float):
    model.eval()
    tot = rec = kl = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        c = F.one_hot(y, num_classes=2).float()

        x_hat, mu, logvar = model(x, c)
        loss, r, k = elbo_loss(x, x_hat, mu, logvar, beta=beta)

        tot += loss.item()
        rec += r.item()
        kl += k.item()
        n += 1

    return {
        "loss": tot / n,
        "recon": rec / n,
        "kl": kl / n,
    }


def train_cvae_on_arrays(
    X,
    y,
    seed=42,
    z_dim=16,
    hidden=128,
    beta=0.5,
    lr=1e-3,
    epochs=200,
    batch_size=64,
    test_size=0.2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False
    )

    model = CVAE(
        x_dim=X.shape[1],
        c_dim=2,
        z_dim=z_dim,
        hidden=hidden
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            cb = F.one_hot(yb, num_classes=2).float()

            x_hat, mu, logvar = model(xb, cb)
            loss, _, _ = elbo_loss(xb, x_hat, mu, logvar, beta=beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

        val_metrics = evaluate_cvae(model, val_loader, device, beta=beta)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {
                "model_state": model.state_dict(),
                "scaler_mean": scaler.mean_.copy(),
                "scaler_scale": scaler.scale_.copy(),
                "z_dim": z_dim,
                "hidden": hidden,
                "x_dim": X.shape[1],
            }

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1:3d} | "
                f"val loss={val_metrics['loss']:.4f} "
                f"recon={val_metrics['recon']:.4f} "
                f"kl={val_metrics['kl']:.4f}"
            )

    return best_state


@torch.no_grad()
def sample_cvae_dataset(best_state, n0, n1, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CVAE(
        x_dim=best_state["x_dim"],
        c_dim=2,
        z_dim=best_state["z_dim"],
        hidden=best_state["hidden"],
    ).to(device)

    model.load_state_dict(best_state["model_state"])
    model.eval()

    mean = np.asarray(best_state["scaler_mean"], dtype=np.float32)
    scale = np.asarray(best_state["scaler_scale"], dtype=np.float32)

    def sample_class(n, y_label):
        z = torch.randn(n, model.z_dim, device=device)
        c = F.one_hot(
            torch.full((n,), y_label, dtype=torch.long, device=device),
            num_classes=2
        ).float()

        x_scaled = model.decode(z, c).cpu().numpy()
        x = x_scaled * scale + mean
        return x.astype(np.float32)

    X0 = sample_class(n0, 0)
    X1 = sample_class(n1, 1)

    X_syn = np.vstack([X0, X1])
    y_syn = np.concatenate([
        np.zeros(n0, dtype=int),
        np.ones(n1, dtype=int)
    ])

    return X_syn, y_syn