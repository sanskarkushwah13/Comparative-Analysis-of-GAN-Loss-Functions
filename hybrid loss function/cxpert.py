"""
chexpert_dcgan_hybrid.py
========================
DCGAN for CheXpert-v1.0-small dataset with HYBRID loss only.
  • Multi-label classification head on Discriminator (14 pathologies)
  • Uncertainty label policy: ignore / zeros / ones
  • Hybrid loss: WGAN-GP + L1 pixel + VGG feature matching
  • FID scoring, checkpointing, best-model saving, CSV results
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as tvm
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT   = os.path.expanduser("~/dataset/CheXpert-v1.0-small")
IMG_SIZE    = 64
Z_DIM       = 100
NGF         = 64
NDF         = 64
BATCH_SIZE  = 32
NUM_WORKERS = 2

PATHOLOGY_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]
NUM_CLASSES = len(PATHOLOGY_COLS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class CheXpertDataset(Dataset):
    def __init__(self, csv_path, data_root, transform=None,
                 uncertainty_policy="ignore", frontal_only=True):
        df = pd.read_csv(csv_path)
        df["Path"] = df["Path"].apply(
            lambda p: os.path.join(
                data_root,
                "/".join(p.replace("\\", "/").split("/")[1:])
            )
        )
        if frontal_only:
            df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

        for col in PATHOLOGY_COLS:
            if col not in df.columns:
                df[col] = 0.0

        if uncertainty_policy == "zeros":
            df[PATHOLOGY_COLS] = df[PATHOLOGY_COLS].replace(-1, 0)
        elif uncertainty_policy == "ones":
            df[PATHOLOGY_COLS] = df[PATHOLOGY_COLS].replace(-1, 1)
        elif uncertainty_policy == "ignore":
            df[PATHOLOGY_COLS] = df[PATHOLOGY_COLS].replace(-1, float("nan"))

        df[PATHOLOGY_COLS] = df[PATHOLOGY_COLS].fillna(0)
        self.df        = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img      = Image.open(row["Path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(
            row[PATHOLOGY_COLS].values.astype(np.float32),
            dtype=torch.float32
        )
        return img, labels


def get_dataloader(split="train", batch_size=BATCH_SIZE,
                   uncertainty_policy="ignore"):
    csv_path = os.path.join(DATA_ROOT,
                            "train.csv" if split == "train" else "valid.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV not found: {csv_path}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CheXpertDataset(
        csv_path=csv_path, data_root=DATA_ROOT,
        transform=transform, uncertainty_policy=uncertainty_policy,
        frontal_only=True
    )
    print(f"  CheXpert {split}: {len(dataset)} frontal images | "
          f"uncertainty_policy='{uncertainty_policy}'")
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(split == "train"),
                      num_workers=NUM_WORKERS,
                      drop_last=True, pin_memory=True)


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
# GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """z(100,1,1) → 512×4 → 256×8 → 128×16 → 64×32 → 3×64"""
    def __init__(self, z_dim=Z_DIM, ngf=NGF):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# DISCRIMINATOR
# ──────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    def __init__(self, ndf=NDF, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adv_head = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 8, num_classes)
        )
        self.apply(weights_init)

    def forward(self, x):
        feat      = self.backbone(x)
        adv_score = self.adv_head(feat).view(-1)
        cls_logit = self.cls_head(feat)
        return adv_score, cls_logit


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION LOSS
# ──────────────────────────────────────────────────────────────────────────────

def cls_loss(logits, labels):
    valid = ~torch.isnan(labels)
    if not valid.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.binary_cross_entropy_with_logits(logits[valid], labels[valid])


# ──────────────────────────────────────────────────────────────────────────────
# HYBRID LOSS  (WGAN-GP + L1 + VGG feature matching)
# ──────────────────────────────────────────────────────────────────────────────

def gradient_penalty(D, real, fake):
    batch  = real.size(0)
    alpha  = torch.rand(batch, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    d_out, _ = D(interp)
    grads    = torch.autograd.grad(
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
        x = (x + 1.0) / 2.0   # [-1,1] → [0,1]
        return self.features(x)

_vgg = None
def _get_vgg():
    global _vgg
    if _vgg is None:
        _vgg = VGGFeatureExtractor().to(device).eval()
    return _vgg


def hybrid_d(D, real, fake, real_labels, lambda_gp=10, lambda_cls=1.0):
    real_score, real_cls = D(real)
    fake_score, _        = D(fake.detach())
    d_adv = -(torch.mean(real_score) - torch.mean(fake_score))
    gp    = gradient_penalty(D, real, fake)
    d_cls = cls_loss(real_cls, real_labels)
    return d_adv + lambda_gp * gp + lambda_cls * d_cls


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
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = (x + 1) / 2.0
        return self.features(x).view(x.size(0), -1)


def get_features(imgs, model, bs=32):
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


def calc_fid(G, loader, inception, n=1000):
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
    folder = os.path.join(out_dir, "samples", "hybrid")
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
    folder = os.path.join(out_dir, "checkpoints", "hybrid")
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

    def update(self, G, D, g_loss, epoch):
        if g_loss < self.best:
            self.best = g_loss
            torch.save(G.state_dict(),
                       os.path.join(self.folder, "hybrid_G_best.pth"))
            torch.save(D.state_dict(),
                       os.path.join(self.folder, "hybrid_D_best.pth"))
            print(f"  🏆 Best epoch {epoch+1} | G: {g_loss:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train(out_dir, epochs=300, inception=None, fid_every=10,
          uncertainty_policy="ignore", lambda_cls=1.0):

    print(f"\n{'='*62}")
    print(f"  DCGAN + CheXpert | HYBRID LOSS | {epochs} Epochs")
    print(f"  Uncertainty policy : {uncertainty_policy}")
    print(f"  Device: {device}")
    print(f"{'='*62}")

    loader = get_dataloader("train", BATCH_SIZE, uncertainty_policy)

    G = Generator(z_dim=Z_DIM, ngf=NGF).to(device)
    D = Discriminator(ndf=NDF, num_classes=NUM_CLASSES).to(device)

    print(f"  G params : {sum(p.numel() for p in G.parameters()):,}")
    print(f"  D params : {sum(p.numel() for p in D.parameters()):,}")

    opt_g    = optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.9))
    opt_d    = optim.Adam(D.parameters(), lr=0.0004, betas=(0.0, 0.9))
    n_critic = 5
    print("  Adam TTUR | lr_G=0.0001 | lr_D=0.0004 | n_critic=5")

    g_losses, d_losses = [], []
    fid_scores         = {}
    best_saver         = BestSaver(out_dir)

    for epoch in range(epochs):
        epoch_g, epoch_d = [], []

        for real, labels in loader:
            real   = real.to(device)
            labels = labels.to(device)
            batch  = real.size(0)

            # ── Discriminator (n_critic steps) ───────────────────────────────
            for _ in range(n_critic):
                fake_gp = G(torch.randn(batch, Z_DIM, 1, 1, device=device))
                d_loss  = hybrid_d(D, real, fake_gp, labels,
                                   lambda_cls=lambda_cls)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            epoch_d.append(d_loss.item())

            # ── Generator ────────────────────────────────────────────────────
            fake          = G(torch.randn(batch, Z_DIM, 1, 1, device=device))
            fake_score, _ = D(fake)
            g_loss        = hybrid_g(fake_score, real, fake)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            epoch_g.append(g_loss.item())

        avg_g = float(np.mean(epoch_g))
        avg_d = float(np.mean(epoch_d))
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        save_samples(G, epoch, out_dir)
        save_ckpt(G, D, opt_g, opt_d, epoch, g_losses, d_losses, out_dir)
        best_saver.update(G, D, avg_g, epoch)

        fid = None
        if inception is not None and (epoch + 1) % fid_every == 0:
            fid = calc_fid(G, loader, inception)
            fid_scores[epoch + 1] = fid

        mv      = float(np.var(fake.detach().cpu().numpy()))
        fid_str = f" | FID: {fid:.4f}" if fid else ""
        print(f"  Epoch [{epoch+1:3d}/{epochs}] | "
              f"G: {avg_g:7.4f} | D: {avg_d:7.4f} | "
              f"ModeVar: {mv:.4f}{fid_str}")

    save_final(G, D, out_dir)
    return G, g_losses, d_losses, fid_scores


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(g_losses, d_losses, fid_scores, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(g_losses, label="G Loss", color="teal")
    axes[0].plot(d_losses, label="D Loss", color="orange")
    axes[0].set_title("Hybrid Loss — Generator & Discriminator (CheXpert)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    if fid_scores:
        axes[1].plot(list(fid_scores.keys()), list(fid_scores.values()),
                     marker='o', color="teal")
        axes[1].set_title("FID Score — Hybrid Loss (Lower = Better)")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("FID Score")
        axes[1].grid(True)
    else:
        axes[1].set_visible(False)

    plt.suptitle("DCGAN CheXpert — Hybrid Loss", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "hybrid_loss_curves.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"✅ Plot → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.join(os.path.expanduser("~"), "project", "chexpert_hybrid")
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output : {out_dir}")
    print(f"📂 Dataset: {DATA_ROOT}")

    print("\nLoading Inception v3 for FID...")
    inception = InceptionFeatureExtractor().to(device).eval()
    print("✅ Inception loaded\n")

    uncertainty_policy = "ignore"   # options: 'ignore' | 'zeros' | 'ones'
    epochs             = 300
    fid_every          = 10

    print(f"{'#'*62}")
    print(f"  DATASET      : CheXpert-v1.0-small (frontal views)")
    print(f"  ARCHITECTURE : DCGAN (ngf={NGF}, ndf={NDF}) + cls head")
    print(f"  LOSS         : HYBRID (WGAN-GP + L1 + VGG perceptual)")
    print(f"  CLASSES      : {NUM_CLASSES} pathology labels")
    print(f"  UNCERTAINTY  : policy='{uncertainty_policy}'")
    print(f"  EPOCHS       : {epochs}")
    print(f"  DEVICE       : {device}")
    print(f"{'#'*62}")

    G, g_losses, d_losses, fid_scores = train(
        out_dir=out_dir,
        epochs=epochs,
        inception=inception,
        fid_every=fid_every,
        uncertainty_policy=uncertainty_policy
    )

    # ── Final metrics ────────────────────────────────────────────────────────
    z = torch.randn(200, Z_DIM, 1, 1, device=device)
    with torch.no_grad():
        fake = G(z)
    var       = float(np.var(fake.detach().cpu().numpy()))
    final_fid = (list(fid_scores.values())[-1]
                 if fid_scores else None)

    result = {
        "Dataset"           : "CheXpert-v1.0-small",
        "Architecture"      : "DCGAN",
        "Optimizer"         : "Adam",
        "Loss_Type"         : "hybrid",
        "Uncertainty_Policy": uncertainty_policy,
        "Epochs"            : epochs,
        "Final_G_Loss"      : round(g_losses[-1], 4),
        "Final_D_Loss"      : round(d_losses[-1], 4),
        "FID_Score"         : round(final_fid, 4) if final_fid else "N/A",
        "ModeVariance"      : round(var, 4),
    }

    df       = pd.DataFrame([result])
    csv_path = os.path.join(out_dir, "hybrid_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results → {csv_path}")
    print("\n", df.to_string(index=False))

    plot_results(g_losses, d_losses, fid_scores, out_dir)

    print(f"\n{'#'*62}")
    print("  🎉 HYBRID EXPERIMENT COMPLETE!")
    print(f"  📁 Output: {out_dir}")
    print(f"{'#'*62}")
    print("""
  Files:
  ├── hybrid_results.csv
  ├── hybrid_loss_curves.png
  ├── samples/hybrid/
  ├── models/
  │   ├── final/  hybrid_G.pth  hybrid_D.pth
  │   └── best/   hybrid_G_best.pth  hybrid_D_best.pth
  └── checkpoints/hybrid/
    """)