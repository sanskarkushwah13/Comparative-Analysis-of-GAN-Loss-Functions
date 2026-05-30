"""
train_gans.py
=============
Full training code for the IEEE Access paper:
  "A Comparative Analysis of GAN Loss Functions for Cross-Domain
   Image Synthesis: Natural Images, Remote Sensing, and Medical Imaging"

Supports 6 objectives × 3 datasets:
  Objectives : standard | lsgan | wgan | wgangp | hinge | hybrid
  Datasets   : cifar10  | eurosat | chexpert

Usage
-----
# Train WGAN-GP on CIFAR-10
python train_gans.py --loss wgangp --dataset cifar10 --epochs 200

# Train the Hybrid loss on CheXpert (3 seeds for robustness)
python train_gans.py --loss hybrid --dataset chexpert --epochs 200 --seeds 42 123 7

# Run the full benchmark (all 6 × 3 combinations)
python train_gans.py --all --epochs 200

Requirements
------------
pip install torch torchvision scipy tqdm matplotlib pillow
"""

import os, argparse, json, time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as torch_grad
from torchvision import datasets, transforms, utils as vutils
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. DCGAN Architecture (fixed for ALL experiments)
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Maps z (100-dim) → 3×64×64 image via 5 transposed-conv blocks.
    Architecture follows Radford et al. (2016) exactly.
    """
    def __init__(self, z_dim: int = 100, ngf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # z → 512×4×4
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            # 512×4×4 → 256×8×8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # 256×8×8 → 128×16×16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # 128×16×16 → 64×32×32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # 64×32×32 → 3×64×64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    """
    Maps 3×64×64 image → scalar score via 5 strided-conv blocks.
    Sigmoid is used ONLY for Standard GAN (use_sigmoid=True).
    """
    def __init__(self, ndf: int = 64, use_sigmoid: bool = False):
        super().__init__()
        layers = [
            # 3×64×64 → 64×32×32 (no BN on first layer)
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64×32×32 → 128×16×16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            # 128×16×16 → 256×8×8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            # 256×8×8 → 512×4×4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            # 512×4×4 → 1×1×1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        # expose penultimate features for IBFV computation
        self.features = nn.Sequential(*layers[:-2])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)

    def get_features(self, x):
        return self.features(x).view(x.size(0), -1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Loss Function Classes
# ─────────────────────────────────────────────────────────────────────────────

class StandardGANLoss:
    """Binary Cross-Entropy / Jensen-Shannon divergence."""
    def __init__(self, device):
        self.criterion = nn.BCELoss()
        self.device = device

    def d_loss(self, D, real, fake):
        b = real.size(0)
        real_label = torch.ones(b, 1, device=self.device)
        fake_label = torch.zeros(b, 1, device=self.device)
        loss = self.criterion(D(real), real_label) + \
               self.criterion(D(fake.detach()), fake_label)
        return loss

    def g_loss(self, D, fake):
        b = fake.size(0)
        real_label = torch.ones(b, 1, device=self.device)
        return self.criterion(D(fake), real_label)

    @property
    def n_critic(self): return 1


class LSGANLoss:
    """Least Squares GAN — Pearson chi^2 divergence."""
    def d_loss(self, D, real, fake):
        return (0.5 * ((D(real) - 1) ** 2).mean() +
                0.5 * (D(fake.detach()) ** 2).mean())

    def g_loss(self, D, fake):
        return 0.5 * ((D(fake) - 1) ** 2).mean()

    @property
    def n_critic(self): return 1


class WGANLoss:
    """Wasserstein GAN with weight clipping."""
    def __init__(self, clip_value: float = 0.01):
        self.clip_value = clip_value

    def d_loss(self, D, real, fake):
        return D(fake.detach()).mean() - D(real).mean()

    def g_loss(self, D, fake):
        return -D(fake).mean()

    def clip_weights(self, D):
        for p in D.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)

    @property
    def n_critic(self): return 5


class WGANGPLoss:
    """Wasserstein GAN with gradient penalty."""
    def __init__(self, device, lambda_gp: float = 10.0):
        self.device = device
        self.lambda_gp = lambda_gp

    def _gradient_penalty(self, D, real, fake):
        b = real.size(0)
        eps = torch.rand(b, 1, 1, 1, device=self.device)
        interp = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
        d_interp = D(interp)
        gradients = torch_grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True
        )[0]
        gp = ((gradients.norm(2, dim=[1, 2, 3]) - 1) ** 2).mean()
        return self.lambda_gp * gp

    def d_loss(self, D, real, fake):
        return (D(fake.detach()).mean() - D(real).mean() +
                self._gradient_penalty(D, real, fake))

    def g_loss(self, D, fake):
        return -D(fake).mean()

    @property
    def n_critic(self): return 5


class HingeLoss:
    """Hinge loss GAN (SVM-style margin)."""
    def d_loss(self, D, real, fake):
        return (torch.relu(1 - D(real)).mean() +
                torch.relu(1 + D(fake.detach())).mean())

    def g_loss(self, D, fake):
        return -D(fake).mean()

    @property
    def n_critic(self): return 1


class HybridLoss:
    """
    Hybrid loss: alpha * WGAN-GP + (1-alpha) * Hinge.
    Generator objective is shared by both components,
    so the gradient is alpha-independent (see paper proof).
    """
    def __init__(self, device, alpha: float = 0.5, lambda_gp: float = 10.0):
        self.alpha = alpha
        self.wgangp = WGANGPLoss(device, lambda_gp)
        self.hinge  = HingeLoss()

    def d_loss(self, D, real, fake):
        return (self.alpha  * self.wgangp.d_loss(D, real, fake) +
                (1 - self.alpha) * self.hinge.d_loss(D, real, fake))

    def g_loss(self, D, fake):
        return -D(fake).mean()   # same for both; alpha-independent

    @property
    def n_critic(self): return 5


def build_loss(name: str, device):
    """Factory for loss functions."""
    name = name.lower()
    if name == "standard": return StandardGANLoss(device)
    if name == "lsgan":    return LSGANLoss()
    if name == "wgan":     return WGANLoss()
    if name == "wgangp":   return WGANGPLoss(device)
    if name == "hinge":    return HingeLoss()
    if name == "hybrid":   return HybridLoss(device)
    raise ValueError(f"Unknown loss: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset Loaders
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loader(root: str = "./data", batch_size: int = 128):
    tf = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)


def get_eurosat_loader(root: str = "./data/EuroSAT", batch_size: int = 128):
    """
    EuroSAT RGB split.
    Download from: https://madm.dfki.de/files/sentinel/EuroSAT.zip
    and unzip into ./data/EuroSAT/  (ImageFolder structure).
    """
    tf = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.ImageFolder(root=root, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)


def get_chexpert_loader(root: str = "./data/CheXpert", batch_size: int = 128):
    """
    CheXpert frontal-view images.
    Download from: https://stanfordmlgroup.github.io/competitions/chexpert/
    Folder structure: root/{train,valid}/PatientXXXXX/studyY/...
    Uses grayscale→RGB replication and resizes to 64×64.
    """
    tf = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.Grayscale(num_output_channels=3),   # replicate channel
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.ImageFolder(root=os.path.join(root, "train"), transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)


def get_loader(name: str, batch_size: int = 128):
    name = name.lower()
    if name == "cifar10":  return get_cifar10_loader(batch_size=batch_size)
    if name == "eurosat":  return get_eurosat_loader(batch_size=batch_size)
    if name == "chexpert": return get_chexpert_loader(batch_size=batch_size)
    raise ValueError(f"Unknown dataset: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FID Computation (Inception-v3 pool-3 features)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """
    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r Sigma_g)^0.5)
    """
    from scipy.linalg import sqrtm
    mu_r, mu_g = real_feats.mean(0), fake_feats.mean(0)
    sig_r = np.cov(real_feats, rowvar=False)
    sig_g = np.cov(fake_feats, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = sqrtm(sig_r @ sig_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sig_r + sig_g - 2 * covmean)
    return float(fid)


def extract_inception_features(images: torch.Tensor, inception_model,
                                device) -> np.ndarray:
    """Extract 2048-dim pool-3 features from Inception-v3."""
    inception_model.eval()
    with torch.no_grad():
        feats = inception_model(images.to(device))
    return feats.cpu().numpy()


def get_inception_model(device):
    """Load Inception-v3 with pool-3 output (2048-dim)."""
    from torchvision.models import inception_v3
    model = inception_v3(pretrained=True, transform_input=False)
    # Replace final FC with identity to get pool-3 features
    model.fc = nn.Identity()
    model.aux_logits = False
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. IBFV (Intra-Batch Feature Variance)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ibfv(D: Discriminator, fake_images: torch.Tensor) -> float:
    """
    IBFV = mean ||phi(x_i) - phi_bar||^2   (penultimate discriminator layer)
    IBFV -> 0 indicates mode collapse.
    """
    feats = D.get_features(fake_images)          # (N, C)
    mean_feat = feats.mean(0, keepdim=True)       # (1, C)
    ibfv = ((feats - mean_feat) ** 2).sum(1).mean()
    return ibfv.item()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(
    loss_name: str,
    dataset_name: str,
    epochs: int = 200,
    batch_size: int = 128,
    z_dim: int = 100,
    fid_every: int = 10,
    fid_samples: int = 2000,
    seed: int = 42,
    out_dir: str = "./runs",
    device_str: str = "auto",
) -> dict:
    """
    Main training function. Returns a metrics dict with per-epoch logs.

    Parameters
    ----------
    loss_name    : one of standard | lsgan | wgan | wgangp | hinge | hybrid
    dataset_name : cifar10 | eurosat | chexpert
    epochs       : total training epochs
    batch_size   : mini-batch size (128 for all experiments in paper)
    z_dim        : latent dimension (100)
    fid_every    : evaluate FID every N epochs
    fid_samples  : number of samples for FID (2000 as in paper)
    seed         : random seed
    out_dir      : directory for checkpoints + samples
    device_str   : 'auto' | 'cuda' | 'cpu'
    """
    set_seed(seed)

    # ── device ────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[{loss_name}/{dataset_name}] device={device}, seed={seed}")

    # ── output dirs ───────────────────────────────────────────────────
    run_id = f"{loss_name}_{dataset_name}_seed{seed}"
    run_dir = Path(out_dir) / run_id
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)

    # ── models ────────────────────────────────────────────────────────
    use_sigmoid = (loss_name == "standard")
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator(use_sigmoid=use_sigmoid).to(device)

    # ── loss ──────────────────────────────────────────────────────────
    loss_fn = build_loss(loss_name, device)
    n_critic = loss_fn.n_critic

    # ── optimisers (TTUR for Wasserstein-based models) ─────────────────
    is_wass = loss_name in ("wgan", "wgangp", "hybrid")
    lr_G = 1e-4 if is_wass else 2e-4
    lr_D = 4e-4 if is_wass else 2e-4
    opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

    # ── data ──────────────────────────────────────────────────────────
    loader = get_loader(dataset_name, batch_size=batch_size)

    # ── inception model for FID ────────────────────────────────────────
    inception = get_inception_model(device)

    # fixed noise for sample visualisation
    fixed_z = torch.randn(64, z_dim, device=device)

    # ── metrics log ───────────────────────────────────────────────────
    metrics = {"epoch": [], "loss_G": [], "loss_D": [],
               "fid": [], "mode_var": [], "ibfv": []}
    best_fid = float("inf")

    # ── collect real features once for FID ────────────────────────────
    real_feats_buf = []
    for images, _ in loader:
        images_up = torch.nn.functional.interpolate(images, 299, mode="bilinear",
                                                     align_corners=False)
        real_feats_buf.append(extract_inception_features(images_up, inception, device))
        if sum(f.shape[0] for f in real_feats_buf) >= fid_samples:
            break
    real_feats = np.concatenate(real_feats_buf, 0)[:fid_samples]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EPOCH LOOP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        epoch_loss_G, epoch_loss_D, n_batches = 0.0, 0.0, 0

        data_iter = iter(loader)
        try:
            while True:
                # ── discriminator updates ──────────────────────────────
                for _ in range(n_critic):
                    real, _ = next(data_iter)
                    real = real.to(device)
                    z = torch.randn(real.size(0), z_dim, device=device)
                    fake = G(z)

                    opt_D.zero_grad()
                    ld = loss_fn.d_loss(D, real, fake)
                    ld.backward()
                    opt_D.step()

                    # weight clipping for vanilla WGAN
                    if loss_name == "wgan":
                        loss_fn.clip_weights(D)

                    epoch_loss_D += ld.item()

                # ── generator update ──────────────────────────────────
                z = torch.randn(batch_size, z_dim, device=device)
                fake = G(z)
                opt_G.zero_grad()
                lg = loss_fn.g_loss(D, fake)
                lg.backward()
                opt_G.step()

                epoch_loss_G += lg.item()
                n_batches += 1

        except StopIteration:
            pass

        avg_G = epoch_loss_G / max(n_batches, 1)
        avg_D = epoch_loss_D / max(n_batches * n_critic, 1)

        # ── evaluation every fid_every epochs ─────────────────────────
        if epoch % fid_every == 0 or epoch == 1:
            G.eval()
            fake_feats_buf, ibfv_buf = [], []
            with torch.no_grad():
                while sum(f.shape[0] for f in fake_feats_buf) < fid_samples:
                    z = torch.randn(min(batch_size, fid_samples), z_dim, device=device)
                    fake_batch = G(z)
                    # IBFV (penultimate discriminator features)
                    ibfv_buf.append(compute_ibfv(D, fake_batch))
                    fake_up = torch.nn.functional.interpolate(
                        fake_batch, 299, mode="bilinear", align_corners=False)
                    fake_feats_buf.append(
                        extract_inception_features(fake_up, inception, device))

            fake_feats = np.concatenate(fake_feats_buf, 0)[:fid_samples]
            fid_score = compute_fid(real_feats, fake_feats)
            mode_var = float(np.var(fake_feats, axis=0).mean())
            ibfv_score = float(np.mean(ibfv_buf))

            if fid_score < best_fid:
                best_fid = fid_score
                torch.save({"G": G.state_dict(), "D": D.state_dict()},
                           run_dir / "best_model.pt")

            metrics["epoch"].append(epoch)
            metrics["loss_G"].append(avg_G)
            metrics["loss_D"].append(avg_D)
            metrics["fid"].append(fid_score)
            metrics["mode_var"].append(mode_var)
            metrics["ibfv"].append(ibfv_score)

            print(f"  Ep {epoch:4d}/{epochs} | "
                  f"LG={avg_G:9.4f} LD={avg_D:9.4f} | "
                  f"FID={fid_score:7.2f} | ModeVar={mode_var:.4f} | "
                  f"IBFV={ibfv_score:.4f}")

            # save sample grid
            with torch.no_grad():
                samples = G(fixed_z).cpu()
            grid = vutils.make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
            vutils.save_image(grid, run_dir / "samples" / f"epoch_{epoch:04d}.png")

        # periodic checkpoint
        if epoch % 50 == 0:
            torch.save({"epoch": epoch,
                        "G": G.state_dict(), "D": D.state_dict(),
                        "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict()},
                       run_dir / f"ckpt_epoch{epoch}.pt")

    # ── save metrics JSON ──────────────────────────────────────────────
    metrics["best_fid"] = best_fid
    metrics["loss_name"] = loss_name
    metrics["dataset"] = dataset_name
    metrics["seed"] = seed
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{run_id}] Best FID = {best_fid:.2f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7. Plotting Utilities
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "standard": "#E63946",
    "lsgan":    "#F4A261",
    "wgan":     "#2A9D8F",
    "wgangp":   "#457B9D",
    "hinge":    "#6A4C93",
    "hybrid":   "#1D3557",
}
DISPLAY_NAMES = {
    "standard": "Standard GAN",
    "lsgan":    "LSGAN",
    "wgan":     "WGAN",
    "wgangp":   "WGAN-GP",
    "hinge":    "Hinge Loss",
    "hybrid":   "Hybrid (Ours)",
}


def plot_fid_curves(runs_dir: str, dataset: str, out_path: str):
    """Plot FID-vs-epoch curves for all 6 loss functions on one dataset."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for loss_name in ["standard", "lsgan", "wgan", "wgangp", "hinge", "hybrid"]:
        mfile = Path(runs_dir) / f"{loss_name}_{dataset}_seed42" / "metrics.json"
        if not mfile.exists():
            print(f"  [skip] {mfile} not found")
            continue
        with open(mfile) as f:
            m = json.load(f)
        ax.plot(m["epoch"], m["fid"],
                color=COLORS[loss_name], marker="o", ms=3, lw=1.5,
                label=DISPLAY_NAMES[loss_name])
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("FID $\\downarrow$", fontsize=11)
    ax.set_title(f"FID vs Epoch — {dataset.upper()}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_loss_curves(runs_dir: str, dataset: str, out_path: str, epochs: int = 100):
    """Plot generator and discriminator loss curves side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for loss_name in ["standard", "lsgan", "wgan", "wgangp", "hinge", "hybrid"]:
        mfile = Path(runs_dir) / f"{loss_name}_{dataset}_seed42" / "metrics.json"
        if not mfile.exists():
            continue
        with open(mfile) as f:
            m = json.load(f)
        ep = [e for e in m["epoch"] if e <= epochs]
        lg = m["loss_G"][:len(ep)]
        ld = m["loss_D"][:len(ep)]
        kw = dict(color=COLORS[loss_name], lw=1.5, label=DISPLAY_NAMES[loss_name])
        axes[0].plot(ep, lg, **kw)
        axes[1].plot(ep, ld, **kw)
    for ax, title in zip(axes, ["Generator Loss", "Discriminator Loss"]):
        ax.set_title(f"{title} — {dataset.upper()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

ALL_LOSSES   = ["standard", "lsgan", "wgan", "wgangp", "hinge", "hybrid"]
ALL_DATASETS = ["cifar10", "eurosat", "chexpert"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GAN Loss Function Benchmark — IEEE Access Paper")
    parser.add_argument("--loss",    choices=ALL_LOSSES,   default="hybrid")
    parser.add_argument("--dataset", choices=ALL_DATASETS, default="cifar10")
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--batch",   type=int, default=128)
    parser.add_argument("--z_dim",   type=int, default=100)
    parser.add_argument("--fid_every",  type=int, default=10)
    parser.add_argument("--fid_samples",type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--out_dir", type=str, default="./runs")
    parser.add_argument("--all", action="store_true",
                        help="Run the full 6×3 benchmark (all losses × all datasets)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.all:
        # Full benchmark — runs all 6 × 3 combinations with seed 42
        print("=" * 60)
        print(" Running full 6 × 3 GAN benchmark")
        print("=" * 60)
        results = {}
        for ds in ALL_DATASETS:
            for loss in ALL_LOSSES:
                key = f"{loss}/{ds}"
                try:
                    m = train(loss_name=loss, dataset_name=ds,
                              epochs=args.epochs, batch_size=args.batch,
                              z_dim=args.z_dim, fid_every=args.fid_every,
                              fid_samples=args.fid_samples, seed=42,
                              out_dir=args.out_dir, device_str=args.device)
                    results[key] = m["best_fid"]
                except Exception as e:
                    print(f"  [ERROR] {key}: {e}")
                    results[key] = None
        print("\n=== Cross-Dataset FID Summary ===")
        header = f"{'Loss':<14}" + "".join(f"{d:>12}" for d in ALL_DATASETS)
        print(header)
        for loss in ALL_LOSSES:
            row = f"{loss:<14}"
            for ds in ALL_DATASETS:
                v = results.get(f"{loss}/{ds}")
                row += f"{v:>12.2f}" if v is not None else f"{'---':>12}"
            print(row)
    else:
        # Single run (possibly multiple seeds)
        for seed in args.seeds:
            train(loss_name=args.loss, dataset_name=args.dataset,
                  epochs=args.epochs, batch_size=args.batch,
                  z_dim=args.z_dim, fid_every=args.fid_every,
                  fid_samples=args.fid_samples, seed=seed,
                  out_dir=args.out_dir, device_str=args.device)

        if len(args.seeds) > 1:
            # Robustness summary
            fids = []
            for seed in args.seeds:
                mfile = (Path(args.out_dir) /
                         f"{args.loss}_{args.dataset}_seed{seed}" / "metrics.json")
                with open(mfile) as f:
                    fids.append(json.load(f)["best_fid"])
            print(f"\nRobustness: FID = {np.mean(fids):.2f} ± {np.std(fids):.2f} "
                  f"over seeds {args.seeds}")
