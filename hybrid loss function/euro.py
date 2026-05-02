"""
eurosat_dcgan_hybrid.py
=======================
DCGAN for EuroSAT dataset with HYBRID loss only.
  • Hybrid loss: WGAN-GP + L1 pixel + VGG feature matching
  • FID scoring, checkpointing, best-model saving, CSV results
  • 150 Epochs (matches paper's EuroSAT training duration)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tvm
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from PIL import Image
import glob

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.expanduser("~/dataset/eurosat/EuroSAT_RGB")
# DATA_ROOT   = os.path.expanduser("~/dataset/euro")   # EuroSAT root folder
IMG_SIZE    = 64                                       # EuroSAT is 64×64
Z_DIM       = 100
NGF         = 64
NDF         = 64
BATCH_SIZE  = 64                                       # smaller than CIFAR-10
NUM_WORKERS = 2
NUM_CLASSES = 10

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ──────────────────────────────────────────────────────────────────────────────
# DATASET  — EuroSAT folder structure:
#   DATA_ROOT/
#     AnnualCrop/  Forest/  Highway/  ... (10 class folders)
#     each folder: 2000–3000 jpg/png files
# ──────────────────────────────────────────────────────────────────────────────

class EuroSATDataset(Dataset):
    """
    Loads EuroSAT from a flat directory of class sub-folders.
    Supports both .jpg and .png files.
    """
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples   = []
        self.labels    = []

        class_dirs = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        if len(class_dirs) == 0:
            raise RuntimeError(
                f"No class sub-folders found in {root}.\n"
                f"Expected structure:\n"
                f"  {root}/AnnualCrop/*.jpg\n"
                f"  {root}/Forest/*.jpg  ... etc."
            )

        print(f"  Found {len(class_dirs)} class folders: {class_dirs}")

        for label, cls in enumerate(class_dirs):
            cls_dir = os.path.join(root, cls)
            files   = (
                glob.glob(os.path.join(cls_dir, "*.jpg")) +
                glob.glob(os.path.join(cls_dir, "*.jpeg")) +
                glob.glob(os.path.join(cls_dir, "*.png")) +
                glob.glob(os.path.join(cls_dir, "*.tif"))
            )
            if len(files) == 0:
                print(f"  ⚠️  No images found in {cls_dir}")
                continue
            self.samples.extend(files)
            self.labels.extend([label] * len(files))
            print(f"    {cls:30s}: {len(files):5d} images")

        print(f"  Total EuroSAT images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_dataloader(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),          # light augmentation for satellite
        transforms.RandomVerticalFlip(),            # satellite images have no canonical orientation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = EuroSATDataset(root=DATA_ROOT, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# WEIGHT INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ──────────────────────────────────────────────────────────────────────────────
# GENERATOR  (z → 3×64×64 for EuroSAT)
# Note: one extra upsampling block compared to CIFAR-10 (32→64 needs 4 blocks)
# ──────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """z(100,1,1) → 512×4 → 256×8 → 128×16 → 64×32 → 3×64"""
    def __init__(self, z_dim=Z_DIM, ngf=NGF):
        super().__init__()
        self.net = nn.Sequential(
            # 100×1×1 → 512×4×4
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
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# DISCRIMINATOR  (3×64×64 → adv score)
# ──────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """3×64×64 → 64×32 → 128×16 → 256×8 → 512×4 → 1"""
    def __init__(self, ndf=NDF):
        super().__init__()
        self.backbone = nn.Sequential(
            # 3×64×64 → 64×32×32
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64×32×32 → 128×16×16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 128×16×16 → 256×8×8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 256×8×8 → 512×4×4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 512×4×4 → 1×1×1
        self.adv_head = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.apply(weights_init)

    def forward(self, x):
        feat      = self.backbone(x)
        adv_score = self.adv_head(feat).view(-1)
        return adv_score


# ──────────────────────────────────────────────────────────────────────────────
# HYBRID LOSS  (WGAN-GP + L1 pixel + VGG feature matching)
# ──────────────────────────────────────────────────────────────────────────────

def gradient_penalty(D, real, fake):
    batch  = real.size(0)
    alpha  = torch.rand(batch, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    d_out  = D(interp)
    grads  = torch.autograd.grad(
        outputs=d_out, inputs=interp,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True
    )[0]
    grads = grads.view(batch, -1)
    return torch.mean((grads.norm(2, dim=1) - 1) ** 2)


class VGGFeatureExtractor(nn.Module):
    """Mid-level VGG-16 features (relu3_3) for perceptual loss."""
    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:16])
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        # EuroSAT is already 64×64, upsample to 224 for VGG
        x = F.interpolate(x, size=(224, 224),
                          mode='bilinear', align_corners=False)
        x = (x + 1.0) / 2.0   # [-1,1] → [0,1]
        return self.features(x)

_vgg = None
def _get_vgg():
    global _vgg
    if _vgg is None:
        _vgg = VGGFeatureExtractor().to(device).eval()
    return _vgg


def hybrid_d(D, real, fake, lambda_gp=10):
    real_score = D(real)
    fake_score = D(fake.detach())
    d_adv      = -(torch.mean(real_score) - torch.mean(fake_score))
    gp         = gradient_penalty(D, real, fake)
    return d_adv + lambda_gp * gp


def hybrid_g(fake_score, real, fake, lambda_l1=10.0, lambda_fm=1.0):
    l_adv  = -torch.mean(fake_score)
    l_l1   = F.l1_loss(fake, real)
    vgg    = _get_vgg()
    with torch.no_grad():
        real_feat = vgg(real)
    fake_feat = vgg(fake)
    l_feat    = F.l1_loss(fake_feat, real_feat.detach())
    return l_adv + lambda_l1 * l_l1 + lambda_fm * l_feat


# ──────────────────────────────────────────────────────────────────────────────
# FID SCORE
# ──────────────────────────────────────────────────────────────────────────────

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights
        inc = tvm.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        inc.aux_logits = False
        self.features = nn.Sequential(
            inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3,
            inc.Conv2d_2b_3x3, nn.MaxPool2d(3, stride=2),
            inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3, nn.MaxPool2d(3, stride=2),
            inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
            inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c, inc.Mixed_6d, inc.Mixed_6e,
            inc.Mixed_7a, inc.Mixed_7b, inc.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = F.interpolate(x, size=(299, 299),
                          mode='bilinear', align_corners=False)
        x = (x + 1) / 2.0
        return self.features(x).view(x.size(0), -1)


def get_features(imgs, model, bs=64):
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        b = imgs[i:i+bs].to(device)
        with torch.no_grad():
            out.append(model(b).cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_fid(rf, ff):
    rf, ff     = rf.astype(np.float64), ff.astype(np.float64)
    mu_r, mu_f = np.mean(rf, 0), np.mean(ff, 0)
    sig_r = np.cov(rf, rowvar=False) + np.eye(rf.shape[1]) * 1e-6
    sig_f = np.cov(ff, rowvar=False) + np.eye(ff.shape[1]) * 1e-6
    diff       = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(abs((diff @ diff) + np.trace(sig_r + sig_f - 2.0 * covmean)))


def calc_fid(G, loader, inception, n=2000):
    print("    Computing FID...")
    real_imgs = []
    for imgs, _ in loader:
        real_imgs.append(imgs)
        if sum(x.size(0) for x in real_imgs) >= n:
            break
    real_imgs = torch.cat(real_imgs)[:n]

    G.eval()
    fake_imgs, done = [], 0
    while done < n:
        cur = min(BATCH_SIZE, n - done)
        z   = torch.randn(cur, Z_DIM, 1, 1, device=device)
        with torch.no_grad():
            fake_imgs.append(G(z).cpu())
        done += cur
    fake_imgs = torch.cat(fake_imgs)[:n]
    G.train()

    fid = compute_fid(get_features(real_imgs, inception),
                      get_features(fake_imgs, inception))
    print(f"    FID: {fid:.4f} ✅")
    return fid


# ──────────────────────────────────────────────────────────────────────────────
# SAVE UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def save_samples(G, epoch, out_dir):
    z = torch.randn(64, Z_DIM, 1, 1, device=device)
    with torch.no_grad():
        fake = G(z)
    grid   = make_grid(fake, normalize=True, nrow=8)
    folder = os.path.join(out_dir, "samples")
    os.makedirs(folder, exist_ok=True)
    save_image(grid, os.path.join(folder, f"epoch_{epoch+1:03d}.png"))


def save_final(G, D, out_dir):
    folder = os.path.join(out_dir, "models", "final")
    os.makedirs(folder, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(folder, "hybrid_G.pth"))
    torch.save(D.state_dict(), os.path.join(folder, "hybrid_D.pth"))
    print(f"  ✅ Model saved → {folder}")


def save_ckpt(G, D, og, od, epoch, gl, dl, out_dir, every=10):
    if (epoch + 1) % every != 0:
        return
    folder = os.path.join(out_dir, "checkpoints")
    os.makedirs(folder, exist_ok=True)
    torch.save({
        "epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
        "og": og.state_dict(), "od": od.state_dict(),
        "gl": gl, "dl": dl
    }, os.path.join(folder, f"epoch_{epoch+1:03d}.pth"))
    print(f"  💾 Checkpoint → epoch_{epoch+1:03d}.pth")


class BestSaver:
    def __init__(self, out_dir):
        self.best   = float("inf")
        self.folder = os.path.join(out_dir, "models", "best")
        os.makedirs(self.folder, exist_ok=True)

    def update(self, G, D, fid, epoch):
        # Save best by FID (lower = better) instead of G loss
        if fid is not None and fid < self.best:
            self.best = fid
            torch.save(G.state_dict(),
                       os.path.join(self.folder, "hybrid_G_best.pth"))
            torch.save(D.state_dict(),
                       os.path.join(self.folder, "hybrid_D_best.pth"))
            print(f"  🏆 Best FID {fid:.4f} at epoch {epoch+1}")


# ──────────────────────────────────────────────────────────────────────────────
# IBFV — Intra-Batch Feature Variance (paper's mode diversity metric)
# ──────────────────────────────────────────────────────────────────────────────

def compute_ibfv(D, fake):
    """
    IBFV = (1/N) * sum_i || phi(x_i) - phi_mean ||^2
    where phi = penultimate discriminator features (512×4×4 flattened)
    """
    D.eval()
    with torch.no_grad():
        feats = D.backbone(fake)                      # (N, 512, 4, 4)
        feats = feats.view(feats.size(0), -1)         # (N, 8192)
        mean  = feats.mean(dim=0, keepdim=True)
        ibfv  = ((feats - mean) ** 2).sum(dim=1).mean().item()
    D.train()
    return ibfv


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train(out_dir, epochs=150, inception=None, fid_every=10):

    print(f"\n{'='*62}")
    print(f"  DCGAN + EuroSAT | HYBRID LOSS | {epochs} Epochs")
    print(f"  Device: {device}")
    print(f"{'='*62}")

    loader = get_dataloader(BATCH_SIZE)

    G = Generator(z_dim=Z_DIM, ngf=NGF).to(device)
    D = Discriminator(ndf=NDF).to(device)

    print(f"  G params : {sum(p.numel() for p in G.parameters()):,}")
    print(f"  D params : {sum(p.numel() for p in D.parameters()):,}")

    # TTUR: standard for WGAN-GP
    opt_g    = optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.9))
    opt_d    = optim.Adam(D.parameters(), lr=0.0004, betas=(0.0, 0.9))
    n_critic = 5
    print("  Adam TTUR | lr_G=0.0001 | lr_D=0.0004 | n_critic=5")

    g_losses, d_losses = [], []
    ibfv_log           = []
    mode_var_log       = []
    fid_scores         = {}
    best_saver         = BestSaver(out_dir)

    for epoch in range(epochs):
        epoch_g, epoch_d = [], []

        for real, _ in loader:
            real  = real.to(device)
            batch = real.size(0)

            # ── Discriminator (n_critic steps) ───────────────────────────────
            for _ in range(n_critic):
                fake_gp = G(torch.randn(batch, Z_DIM, 1, 1, device=device))
                d_loss  = hybrid_d(D, real, fake_gp)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            epoch_d.append(d_loss.item())

            # ── Generator ────────────────────────────────────────────────────
            fake       = G(torch.randn(batch, Z_DIM, 1, 1, device=device))
            fake_score = D(fake)
            g_loss     = hybrid_g(fake_score, real, fake)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            epoch_g.append(g_loss.item())

        avg_g = float(np.mean(epoch_g))
        avg_d = float(np.mean(epoch_d))
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        # Mode variance (pixel-space)
        with torch.no_grad():
            sample_z    = torch.randn(64, Z_DIM, 1, 1, device=device)
            sample_fake = G(sample_z)
        mv   = float(np.var(sample_fake.detach().cpu().numpy()))
        mode_var_log.append(mv)

        # IBFV — discriminator feature diversity
        ibfv = compute_ibfv(D, sample_fake)
        ibfv_log.append(ibfv)

        save_samples(G, epoch, out_dir)
        save_ckpt(G, D, opt_g, opt_d, epoch, g_losses, d_losses, out_dir)

        # FID every fid_every epochs
        fid = None
        if inception is not None and (epoch + 1) % fid_every == 0:
            fid = calc_fid(G, loader, inception)
            fid_scores[epoch + 1] = fid

        best_saver.update(G, D, fid, epoch)

        fid_str = f" | FID: {fid:.4f}" if fid else ""
        print(f"  Epoch [{epoch+1:3d}/{epochs}] | "
              f"G: {avg_g:8.4f} | D: {avg_d:8.4f} | "
              f"ModeVar: {mv:.4f} | IBFV: {ibfv:.4f}{fid_str}")

    save_final(G, D, out_dir)
    return G, g_losses, d_losses, fid_scores, mode_var_log, ibfv_log


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(g_losses, d_losses, fid_scores, mode_var_log, ibfv_log, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss curves
    axes[0, 0].plot(g_losses, label="G Loss", color="teal")
    axes[0, 0].plot(d_losses, label="D Loss", color="orange")
    axes[0, 0].set_title("Hybrid Loss — Generator & Discriminator (EuroSAT)")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True)

    # FID curve
    if fid_scores:
        axes[0, 1].plot(list(fid_scores.keys()), list(fid_scores.values()),
                        marker='o', color="teal")
        axes[0, 1].set_title("FID Score — Hybrid Loss (Lower = Better)")
        axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("FID Score")
        axes[0, 1].grid(True)
    else:
        axes[0, 1].set_visible(False)

    # Mode Variance
    axes[1, 0].plot(mode_var_log, color="purple")
    axes[1, 0].set_title("Mode Variance (Pixel-Space)")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Variance")
    axes[1, 0].grid(True)

    # IBFV
    axes[1, 1].plot(ibfv_log, color="green")
    axes[1, 1].set_title("IBFV — Intra-Batch Feature Variance (Higher = More Diverse)")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("IBFV")
    axes[1, 1].grid(True)

    plt.suptitle("DCGAN EuroSAT — Hybrid Loss (150 Epochs)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "hybrid_loss_curves.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"✅ Plot → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.join(os.path.expanduser("~"), "project", "eurosat_hybrid")
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output : {out_dir}")
    print(f"📂 Dataset: {DATA_ROOT}")

    print("\nLoading Inception v3 for FID...")
    inception = InceptionFeatureExtractor().to(device).eval()
    print("✅ Inception loaded\n")

    epochs    = 150          # paper uses 150 epochs for EuroSAT
    fid_every = 10

    print(f"{'#'*62}")
    print(f"  DATASET      : EuroSAT (10 classes, ~27k images)")
    print(f"  ARCHITECTURE : DCGAN (ngf={NGF}, ndf={NDF})")
    print(f"  LOSS         : HYBRID (WGAN-GP + L1 + VGG perceptual)")
    print(f"  IMAGE SIZE   : {IMG_SIZE}×{IMG_SIZE}")
    print(f"  BATCH SIZE   : {BATCH_SIZE}")
    print(f"  EPOCHS       : {epochs}")
    print(f"  DEVICE       : {device}")
    print(f"{'#'*62}")

    G, g_losses, d_losses, fid_scores, mv_log, ibfv_log = train(
        out_dir=out_dir,
        epochs=epochs,
        inception=inception,
        fid_every=fid_every
    )

    # ── Final metrics ────────────────────────────────────────────────────────
    z = torch.randn(200, Z_DIM, 1, 1, device=device)
    with torch.no_grad():
        fake = G(z)

    final_fid  = list(fid_scores.values())[-1] if fid_scores else None
    final_mv   = mv_log[-1]
    final_ibfv = ibfv_log[-1]

    result = {
        "Dataset"       : "EuroSAT",
        "Architecture"  : "DCGAN",
        "Optimizer"     : "Adam TTUR",
        "Loss_Type"     : "hybrid",
        "Image_Size"    : f"{IMG_SIZE}x{IMG_SIZE}",
        "Batch_Size"    : BATCH_SIZE,
        "Epochs"        : epochs,
        "Final_G_Loss"  : round(g_losses[-1], 4),
        "Final_D_Loss"  : round(d_losses[-1], 4),
        "FID_Score"     : round(final_fid, 4) if final_fid else "N/A",
        "ModeVariance"  : round(final_mv, 4),
        "IBFV"          : round(final_ibfv, 4),
    }

    df       = pd.DataFrame([result])
    csv_path = os.path.join(out_dir, "hybrid_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results → {csv_path}")
    print("\n", df.to_string(index=False))

    plot_results(g_losses, d_losses, fid_scores, mv_log, ibfv_log, out_dir)

    print(f"\n{'#'*62}")
    print("  🎉 EuroSAT HYBRID EXPERIMENT COMPLETE!")
    print(f"  📁 Output: {out_dir}")
    print(f"{'#'*62}")
    print("""
  Files:
  ├── hybrid_results.csv
  ├── hybrid_loss_curves.png        ← 4-panel plot (loss, FID, ModeVar, IBFV)
  ├── samples/                      ← generated grids every epoch
  ├── models/
  │   ├── final/  hybrid_G.pth  hybrid_D.pth
  │   └── best/   hybrid_G_best.pth  hybrid_D_best.pth  (saved by FID)
  └── checkpoints/                  ← saved every 10 epochs
    """)