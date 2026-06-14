"""
chexpert_dcgan.py  ── CORRECTED + PUBLICATION-GRADE METRICS
=============================================================
DCGAN for CheXpert-v1.0-small with:

  ARCHITECTURE
  ─────────────
  • Generator  : z(100) → 3×64 images
  • Discriminator: shared backbone + adversarial head + 14-class aux classifier

  LOSS VARIANTS (6)
  ─────────────────
  standard | lsgan | wgan | wgangp | hinge | hybrid

  METRICS (publication-grade)
  ────────────────────────────
  ✅ FID   — Fréchet Inception Distance
  ✅ IS    — Inception Score  (NEW)
  ✅ P&R   — Precision & Recall for GANs  (NEW, VERY IMPORTANT)
  ✅ SSIM  — Structural Similarity Index  (NEW, medical imaging)

  BUG FIXES vs original
  ──────────────────────
  [FIX-1] Double sigmoid in standard GAN — removed from loss fns
  [FIX-2] WGAN clip scope — only backbone params, moved to training loop
  [FIX-3] cls_loss masking — uses reduction='none' + 2-D mask correctly
  [FIX-4] WGAN-GP fake_gp — reuses single fake tensor, no double forward
  [FIX-5] Uncertainty 'ignore' — NaN fill separated from missing-col fill
  [FIX-6] CSV path — points to CheXpert-v1.0-small sub-folder correctly
  [FIX-7] BestSaver — tracks FID (lower=better) not G loss
  [FIX-8] calc_fid — preserves G.training mode correctly

Dataset layout expected:
  ~/dataset/CheXpert-v1.0-small/
      train.csv
      valid.csv
      train/   patient.../study.../view_frontal.jpg ...
      valid/   ...

Usage:
  python chexpert_dcgan.py
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
from skimage.metrics import structural_similarity as sk_ssim   # SSIM


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT   = os.path.join(os.path.expanduser("~"), "dataset", "CheXpert-v1.0-small")
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
NUM_CLASSES = len(PATHOLOGY_COLS)   # 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ──────────────────────────────────────────────────────────────────────────────
# DATASET   [FIX-5] uncertainty 'ignore' NaN fill separated from missing cols
#           [FIX-6] CSV path inside CheXpert-v1.0-small sub-folder
# ──────────────────────────────────────────────────────────────────────────────

class CheXpertDataset(Dataset):
    """
    Uncertainty policy
    ------------------
    'ignore' : -1 → NaN  (masked out of cls loss; NOT back-filled to 0)
    'zeros'  : -1 → 0
    'ones'   : -1 → 1
    """

    def __init__(self, csv_path, data_root, transform=None,
                 uncertainty_policy="ignore", frontal_only=True):
        df = pd.read_csv(csv_path)

        # Normalise CSV paths → absolute paths
        # CSV stores:  CheXpert-v1.0-small/train/patient.../...
        # We strip the first segment and join with data_root
        def _abs(p):
            parts = p.replace("\\", "/").split("/")
            # drop leading 'CheXpert-v1.0-small' segment if present
            if parts[0].lower().startswith("chexpert"):
                parts = parts[1:]
            return os.path.join(data_root, *parts)

        df["Path"] = df["Path"].apply(_abs)

        if frontal_only:
            df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

        # Fill columns that are entirely absent with 0
        for col in PATHOLOGY_COLS:
            if col not in df.columns:
                df[col] = 0.0

        # [FIX-5] Apply uncertainty policy FIRST
        if uncertainty_policy == "zeros":
            df[PATHOLOGY_COLS] = df[PATHOLOGY_COLS].replace(-1, 0)
        elif uncertainty_policy == "ones":
            df[PATHOLOGY_COLS] = df[PATHOLOGY_COLS].replace(-1, 1)
        elif uncertainty_policy == "ignore":
            df[PATHOLOGY_COLS] = df[PATHOLOGY_COLS].replace(-1, float("nan"))
            # Do NOT fillna here — NaN signals 'ignore' to cls_loss

        # Only fill NaN values in columns that were originally absent
        # (uncertainty NaNs must stay as NaN for the 'ignore' policy)
        if uncertainty_policy != "ignore":
            df[PATHOLOGY_COLS] = df[PATHOLOGY_COLS].fillna(0)

        self.df        = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = row["Path"]
        img      = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(
            row[PATHOLOGY_COLS].values.astype(np.float32),
            dtype=torch.float32
        )
        return img, labels


def get_dataloader(split="train", batch_size=BATCH_SIZE,
                   uncertainty_policy="ignore"):
    # [FIX-6] CSV lives inside DATA_ROOT (which already includes sub-folder)
    csv_name = "train.csv" if split == "train" else "valid.csv"
    csv_path = os.path.join(DATA_ROOT, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CheXpertDataset(
        csv_path=csv_path,
        data_root=DATA_ROOT,
        transform=transform,
        uncertainty_policy=uncertainty_policy,
        frontal_only=True
    )
    print(f"  CheXpert {split}: {len(dataset)} frontal images | "
          f"uncertainty_policy='{uncertainty_policy}'")

    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(split == "train"),
                      num_workers=NUM_WORKERS,
                      drop_last=True,
                      pin_memory=True)


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
# DISCRIMINATOR  (backbone + adv head + cls head)
# ──────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Backbone : 3×64 → 64×32 → 128×16 → 256×8 → 512×4
    adv_head : raw logit (NO built-in sigmoid — handled per-loss-fn)  [FIX-1]
    cls_head : 14 pathology logits
    """
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

        # [FIX-1] No sigmoid here — each loss function handles it explicitly
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

    def backbone_parameters(self):
        """Used by WGAN clip — backbone only."""
        return self.backbone.parameters()


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION LOSS  [FIX-3] correct 2-D masking
# ──────────────────────────────────────────────────────────────────────────────

def cls_loss(logits, labels):
    """
    Binary cross-entropy over 14 pathology labels.
    NaN entries (uncertainty_policy='ignore') are masked per element.
    [FIX-3] Uses reduction='none' so we mask in 2-D correctly.
    """
    valid = ~torch.isnan(labels)           # (B, 14)
    if not valid.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    # Element-wise loss, then mask and average
    loss_elem = F.binary_cross_entropy_with_logits(
        logits, labels.nan_to_num(0.0), reduction='none'
    )                                      # (B, 14)
    return loss_elem[valid].mean()


# ──────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS  [FIX-1] sigmoid handled here, not in D architecture
# ──────────────────────────────────────────────────────────────────────────────

bce = nn.BCEWithLogitsLoss()   # numerically stable — takes raw logits
mse = nn.MSELoss()


# ── 1. Standard GAN ──────────────────────────────────────────────────────────
def standard_d(D, real, fake, real_labels, lambda_cls=1.0):
    real_score, real_cls = D(real)
    fake_score, _        = D(fake.detach())
    # [FIX-1] BCEWithLogitsLoss takes raw scores (no manual sigmoid)
    d_adv = 0.5 * (
        bce(real_score, torch.ones_like(real_score)) +
        bce(fake_score, torch.zeros_like(fake_score))
    )
    return d_adv + lambda_cls * cls_loss(real_cls, real_labels)

def standard_g(fake_score):
    # [FIX-1] raw logit into BCEWithLogitsLoss
    return bce(fake_score, torch.ones_like(fake_score))


# ── 2. LSGAN ─────────────────────────────────────────────────────────────────
def lsgan_d(D, real, fake, real_labels, lambda_cls=1.0):
    real_score, real_cls = D(real)
    fake_score, _        = D(fake.detach())
    d_adv = 0.5 * (mse(real_score, torch.ones_like(real_score)) +
                   mse(fake_score, torch.zeros_like(fake_score)))
    return d_adv + lambda_cls * cls_loss(real_cls, real_labels)

def lsgan_g(fake_score):
    return 0.5 * mse(fake_score, torch.ones_like(fake_score))


# ── 3. WGAN  [FIX-2] clip moved to training loop, backbone-only ──────────────
def wgan_d(D, real, fake, real_labels, lambda_cls=1.0):
    real_score, real_cls = D(real)
    fake_score, _        = D(fake.detach())
    d_adv = -(torch.mean(real_score) - torch.mean(fake_score))
    # [FIX-2] NO clipping here — done in training loop on backbone params only
    return d_adv + lambda_cls * cls_loss(real_cls, real_labels)

def wgan_g(fake_score):
    return -torch.mean(fake_score)


# ── 4. WGAN-GP  [FIX-4] single fake tensor for GP ────────────────────────────
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
    return torch.mean((grads.view(batch, -1).norm(2, dim=1) - 1) ** 2)

def wgangp_d(D, real, fake, real_labels, lambda_gp=10, lambda_cls=1.0):
    # [FIX-4] fake is passed in from the training loop (single tensor)
    real_score, real_cls = D(real)
    fake_score, _        = D(fake.detach())
    d_adv = -(torch.mean(real_score) - torch.mean(fake_score))
    gp    = gradient_penalty(D, real, fake)
    return d_adv + lambda_gp * gp + lambda_cls * cls_loss(real_cls, real_labels)

def wgangp_g(fake_score):
    return -torch.mean(fake_score)


# ── 5. Hinge ──────────────────────────────────────────────────────────────────
def hinge_d(D, real, fake, real_labels, lambda_cls=1.0):
    real_score, real_cls = D(real)
    fake_score, _        = D(fake.detach())
    d_adv = (torch.mean(F.relu(1.0 - real_score)) +
             torch.mean(F.relu(1.0 + fake_score)))
    return d_adv + lambda_cls * cls_loss(real_cls, real_labels)

def hinge_g(fake_score):
    return -torch.mean(fake_score)


# ── 6. Hybrid (WGAN-GP + L1 pixel + VGG feature matching) ────────────────────
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:16])
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.features((x + 1.0) / 2.0)   # [-1,1] → [0,1]

_vgg = None
def _get_vgg():
    global _vgg
    if _vgg is None:
        _vgg = VGGFeatureExtractor().to(device).eval()
    return _vgg

def hybrid_d(D, real, fake, real_labels, lambda_gp=10, lambda_cls=1.0):
    return wgangp_d(D, real, fake, real_labels,
                    lambda_gp=lambda_gp, lambda_cls=lambda_cls)

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
# ✅ METRIC A — FID  (Fréchet Inception Distance)
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
        # Also expose softmax logits for IS
        self.classifier = nn.Sequential(
            inc.dropout,
            nn.Flatten(),
            inc.fc,
            nn.Softmax(dim=1)
        )

    def forward(self, x, return_logits=False):
        x = F.interpolate(x, (299, 299), mode='bilinear', align_corners=False)
        x = (x + 1.0) / 2.0                      # [-1,1] → [0,1]
        feat = self.features(x)                   # (B, 2048, 1, 1)
        feat_flat = feat.view(x.size(0), -1)      # (B, 2048)
        if return_logits:
            logits = self.classifier(feat)        # (B, 1000) softmax probs
            return feat_flat, logits
        return feat_flat


def _get_features_and_probs(imgs, inception, bs=32):
    """Returns (features [N,2048], probs [N,1000]) from a tensor of images."""
    inception.eval()
    feats, probs = [], []
    for i in range(0, len(imgs), bs):
        b = imgs[i:i+bs].to(device)
        with torch.no_grad():
            f, p = inception(b, return_logits=True)
        feats.append(f.cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(feats, 0), np.concatenate(probs, 0)


def compute_fid(rf, ff):
    rf, ff     = rf.astype(np.float64), ff.astype(np.float64)
    mu_r, mu_f = np.mean(rf, 0), np.mean(ff, 0)
    sig_r = np.cov(rf, rowvar=False) + np.eye(rf.shape[1]) * 1e-6
    sig_f = np.cov(ff, rowvar=False) + np.eye(ff.shape[1]) * 1e-6
    diff  = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(abs(diff @ diff + np.trace(sig_r + sig_f - 2.0 * covmean)))


# ──────────────────────────────────────────────────────────────────────────────
# ✅ METRIC B — Inception Score (IS)
#    IS = exp( E_x[ KL( p(y|x) || p(y) ) ] )
#    Higher IS = better (sharper + diverse images)
#    Reference: Salimans et al. 2016
# ──────────────────────────────────────────────────────────────────────────────

def compute_inception_score(probs, splits=10):
    """
    Args:
        probs  : np.ndarray (N, 1000) — softmax class probabilities
        splits : number of chunks for mean/std estimation
    Returns:
        is_mean, is_std
    """
    scores = []
    chunk  = len(probs) // splits
    for k in range(splits):
        part = probs[k * chunk : (k + 1) * chunk]
        p_y  = np.mean(part, axis=0, keepdims=True)       # marginal
        kl   = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


# ──────────────────────────────────────────────────────────────────────────────
# ✅ METRIC C — Precision & Recall for GANs
#    Precision = fraction of fake images that fall in real manifold  (fidelity)
#    Recall    = fraction of real images covered by fake manifold    (diversity)
#    Reference: Kynkäänniemi et al. 2019
# ──────────────────────────────────────────────────────────────────────────────

def _manifold_radii(feats, k=3):
    """
    For each feature vector, compute the distance to its k-th nearest neighbour
    (the 'radius' that defines the manifold around that point).
    Uses chunked matrix computation to avoid OOM on large N.
    """
    n = len(feats)
    feats_t = torch.tensor(feats, dtype=torch.float32)
    radii   = torch.zeros(n)
    chunk   = 512
    for i in range(0, n, chunk):
        batch = feats_t[i:i+chunk]                  # (C, D)
        dists = torch.cdist(batch, feats_t)          # (C, N)
        kth   = torch.topk(dists, k=k+1, largest=False).values[:, -1]
        radii[i:i+len(batch)] = kth
    return radii.numpy()


def compute_precision_recall(real_feats, fake_feats, k=3):
    """
    Args:
        real_feats : (N, D) real image features
        fake_feats : (M, D) generated image features
        k          : neighbourhood size
    Returns:
        precision, recall  (both in [0, 1])
    """
    real_r = _manifold_radii(real_feats, k)      # per-real-point radius
    fake_r = _manifold_radii(fake_feats, k)      # per-fake-point radius

    real_t = torch.tensor(real_feats, dtype=torch.float32)
    fake_t = torch.tensor(fake_feats, dtype=torch.float32)

    # Precision: for each fake point, is it inside ANY real ball?
    prec_in = 0
    chunk   = 512
    for i in range(0, len(fake_t), chunk):
        fb = fake_t[i:i+chunk]
        dists = torch.cdist(fb, real_t)            # (C, N_real)
        # inside real ball: dist ≤ real_radius
        inside = (dists <= torch.tensor(real_r, dtype=torch.float32)).any(dim=1)
        prec_in += inside.sum().item()
    precision = prec_in / len(fake_feats)

    # Recall: for each real point, is it inside ANY fake ball?
    rec_in = 0
    for i in range(0, len(real_t), chunk):
        rb = real_t[i:i+chunk]
        dists = torch.cdist(rb, fake_t)            # (C, N_fake)
        inside = (dists <= torch.tensor(fake_r, dtype=torch.float32)).any(dim=1)
        rec_in += inside.sum().item()
    recall = rec_in / len(real_feats)

    return float(precision), float(recall)


# ──────────────────────────────────────────────────────────────────────────────
# ✅ METRIC D — SSIM  (Structural Similarity Index)
#    Critical for medical imaging — measures structural fidelity
#    Computes pairwise SSIM between generated and nearest-real images
#    Range: [0, 1], higher is better
# ──────────────────────────────────────────────────────────────────────────────

def compute_ssim(real_imgs, fake_imgs, n_pairs=200):
    """
    Computes average SSIM between generated images and their nearest real
    neighbours (in pixel space).  Converts [-1,1] tensors → [0,1] numpy.

    Args:
        real_imgs  : (N, 3, H, W) tensor, values in [-1, 1]
        fake_imgs  : (M, 3, H, W) tensor, values in [-1, 1]
        n_pairs    : number of fake images to evaluate (speed/accuracy trade-off)
    Returns:
        ssim_mean, ssim_std
    """
    # Convert to (N, H, W, 3) numpy in [0, 1]
    def to_np(t):
        t = t.detach().cpu().float()
        t = (t * 0.5 + 0.5).clamp(0, 1)          # [-1,1] → [0,1]
        return t.permute(0, 2, 3, 1).numpy()       # (N, H, W, C)

    real_np = to_np(real_imgs)
    fake_np = to_np(fake_imgs[:n_pairs])

    scores = []
    # For each fake image, find the most similar real image and compute SSIM
    for fi in range(len(fake_np)):
        best_ssim = -1.0
        for ri in range(len(real_np)):
            s = sk_ssim(
                real_np[ri], fake_np[fi],
                data_range=1.0,
                channel_axis=-1
            )
            if s > best_ssim:
                best_ssim = s
        scores.append(best_ssim)

    return float(np.mean(scores)), float(np.std(scores))


def compute_ssim_fast(real_imgs, fake_imgs, n_pairs=200):
    """
    Faster SSIM variant: matches each fake to its nearest real by L2 distance
    in pixel space rather than brute-force SSIM search.  Good for large N.
    """
    def to_np_and_flat(t):
        t = t.detach().cpu().float()
        t = (t * 0.5 + 0.5).clamp(0, 1)
        np_t = t.permute(0, 2, 3, 1).numpy()           # (N, H, W, C)
        flat  = np_t.reshape(len(np_t), -1)             # (N, H*W*C)
        return np_t, flat

    real_np, real_flat = to_np_and_flat(real_imgs)
    fake_np, fake_flat = to_np_and_flat(fake_imgs[:n_pairs])

    scores = []
    chunk  = 64
    for i in range(0, len(fake_flat), chunk):
        fb = fake_flat[i:i+chunk]                        # (C, D)
        # L2 distance to all real images
        diff = real_flat[None, :, :] - fb[:, None, :]   # (C, N_real, D)
        dists = np.linalg.norm(diff, axis=-1)            # (C, N_real)
        nn_idx = np.argmin(dists, axis=1)                # (C,)
        for j, ri in enumerate(nn_idx):
            s = sk_ssim(
                real_np[ri], fake_np[i+j],
                data_range=1.0,
                channel_axis=-1
            )
            scores.append(s)

    return float(np.mean(scores)), float(np.std(scores))


# ──────────────────────────────────────────────────────────────────────────────
# COMBINED METRIC COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def calc_all_metrics(G, loader, inception, n=1000, n_ssim_pairs=200):
    """
    Computes FID, IS, Precision, Recall, SSIM in one pass.
    [FIX-8] Preserves G.training mode correctly.
    """
    print("    Computing metrics (FID / IS / P&R / SSIM)...")

    # ── Collect real images ───────────────────────────────────────────────────
    real_imgs = []
    for imgs, _ in loader:
        real_imgs.append(imgs)
        if sum(x.size(0) for x in real_imgs) >= n:
            break
    real_imgs = torch.cat(real_imgs)[:n]

    # ── Generate fake images ──────────────────────────────────────────────────
    was_training = G.training
    G.eval()
    fake_imgs, done = [], 0
    while done < n:
        cur = min(BATCH_SIZE, n - done)
        z   = torch.randn(cur, Z_DIM, 1, 1, device=device)
        with torch.no_grad():
            fake_imgs.append(G(z).cpu())
        done += cur
    G.train(was_training)           # [FIX-8] restore original mode
    fake_imgs = torch.cat(fake_imgs)[:n]

    # ── Inception features + probs ────────────────────────────────────────────
    real_feats, _          = _get_features_and_probs(real_imgs, inception)
    fake_feats, fake_probs = _get_features_and_probs(fake_imgs, inception)

    # ── FID ───────────────────────────────────────────────────────────────────
    fid = compute_fid(real_feats, fake_feats)

    # ── IS ────────────────────────────────────────────────────────────────────
    is_mean, is_std = compute_inception_score(fake_probs)

    # ── Precision & Recall ────────────────────────────────────────────────────
    # Use a subset for speed (P&R is O(N^2))
    pr_n = min(n, 2048)
    precision, recall = compute_precision_recall(
        real_feats[:pr_n], fake_feats[:pr_n], k=3
    )

    # ── SSIM ──────────────────────────────────────────────────────────────────
    # Use fast nearest-neighbour variant
    ssim_mean, ssim_std = compute_ssim_fast(real_imgs, fake_imgs,
                                            n_pairs=n_ssim_pairs)

    print(f"    FID: {fid:.4f} | IS: {is_mean:.3f}±{is_std:.3f} | "
          f"Prec: {precision:.4f} | Rec: {recall:.4f} | "
          f"SSIM: {ssim_mean:.4f}±{ssim_std:.4f}")

    return {
        "fid"      : round(fid,       4),
        "is_mean"  : round(is_mean,   4),
        "is_std"   : round(is_std,    4),
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "ssim_mean": round(ssim_mean, 4),
        "ssim_std" : round(ssim_std,  4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SAVE UTILITIES  [FIX-7] BestSaver now tracks FID
# ──────────────────────────────────────────────────────────────────────────────

def save_samples(G, epoch, loss_type, out_dir):
    z = torch.randn(64, Z_DIM, 1, 1, device=device)
    with torch.no_grad():
        fake = G(z)
    grid   = make_grid(fake, normalize=True, nrow=8)
    folder = os.path.join(out_dir, "samples", loss_type)
    os.makedirs(folder, exist_ok=True)
    save_image(grid, os.path.join(folder, f"epoch_{epoch+1:03d}.png"))


def save_final(G, D, loss_type, out_dir):
    folder = os.path.join(out_dir, "models", "final")
    os.makedirs(folder, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(folder, f"{loss_type}_G.pth"))
    torch.save(D.state_dict(), os.path.join(folder, f"{loss_type}_D.pth"))
    print(f"  Model saved → {folder}")


def save_ckpt(G, D, og, od, epoch, gl, dl, loss_type, out_dir, every=10):
    if (epoch + 1) % every != 0:
        return
    folder = os.path.join(out_dir, "checkpoints", loss_type)
    os.makedirs(folder, exist_ok=True)
    torch.save({
        "epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
        "og": og.state_dict(), "od": od.state_dict(),
        "gl": gl, "dl": dl
    }, os.path.join(folder, f"epoch_{epoch+1:03d}.pth"))
    print(f"  Checkpoint → epoch_{epoch+1:03d}.pth")


class BestSaver:
    """[FIX-7] Tracks lowest FID (better quality) instead of G loss."""
    def __init__(self, loss_type, out_dir):
        self.best_fid  = float("inf")
        self.loss_type = loss_type
        self.folder    = os.path.join(out_dir, "models", "best")
        os.makedirs(self.folder, exist_ok=True)

    def update(self, G, D, metrics, epoch):
        fid = metrics.get("fid", float("inf"))
        if fid < self.best_fid:
            self.best_fid = fid
            torch.save(G.state_dict(),
                       os.path.join(self.folder, f"{self.loss_type}_G_best.pth"))
            torch.save(D.state_dict(),
                       os.path.join(self.folder, f"{self.loss_type}_D_best.pth"))
            print(f"  Best model saved | epoch {epoch+1} | FID: {fid:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train(loss_type, out_dir, epochs=30, inception=None, metric_every=10,
          uncertainty_policy="ignore", lambda_cls=1.0):

    print(f"\n{'='*66}")
    print(f"  DCGAN + CheXpert | {loss_type.upper()} | {epochs} Epochs")
    print(f"  Uncertainty policy: {uncertainty_policy}  |  Device: {device}")
    print(f"{'='*66}")

    loader = get_dataloader("train", BATCH_SIZE, uncertainty_policy)

    # [FIX-1] No use_sigmoid flag — D never adds sigmoid internally
    G = Generator(z_dim=Z_DIM, ngf=NGF).to(device)
    D = Discriminator(ndf=NDF, num_classes=NUM_CLASSES).to(device)

    print(f"  G params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"  D params: {sum(p.numel() for p in D.parameters()):,}")

    if loss_type in ["wgan", "wgangp", "hybrid"]:
        opt_g    = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
        opt_d    = optim.Adam(D.parameters(), lr=4e-4, betas=(0.0, 0.9))
        n_critic = 5
        print("  TTUR Adam | lr_G=1e-4 | lr_D=4e-4 | n_critic=5")
    else:
        opt_g    = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_d    = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        n_critic = 1
        print("  Adam | lr=2e-4 | betas=(0.5, 0.999) | n_critic=1")

    g_losses, d_losses = [], []
    all_metrics        = {}          # epoch → dict of all metrics
    best_saver         = BestSaver(loss_type, out_dir)

    for epoch in range(epochs):
        epoch_g, epoch_d = [], []

        for real, labels in loader:
            real   = real.to(device)
            labels = labels.to(device)
            batch  = real.size(0)

            # ── Discriminator update ─────────────────────────────────────────
            for _ in range(n_critic):
                z    = torch.randn(batch, Z_DIM, 1, 1, device=device)
                fake = G(z).detach()

                opt_d.zero_grad()

                if loss_type == "standard":
                    d_loss = standard_d(D, real, fake, labels, lambda_cls)
                elif loss_type == "lsgan":
                    d_loss = lsgan_d(D, real, fake, labels, lambda_cls)
                elif loss_type == "wgan":
                    d_loss = wgan_d(D, real, fake, labels, lambda_cls)
                elif loss_type == "wgangp":
                    # [FIX-4] single fake tensor into wgangp_d
                    fake_for_gp = G(torch.randn(batch, Z_DIM, 1, 1, device=device))
                    d_loss = wgangp_d(D, real, fake_for_gp, labels, lambda_cls=lambda_cls)
                elif loss_type == "hinge":
                    d_loss = hinge_d(D, real, fake, labels, lambda_cls)
                elif loss_type == "hybrid":
                    fake_for_gp = G(torch.randn(batch, Z_DIM, 1, 1, device=device))
                    d_loss = hybrid_d(D, real, fake_for_gp, labels, lambda_cls=lambda_cls)

                d_loss.backward()
                opt_d.step()

                # [FIX-2] WGAN clip — backbone only, AFTER opt step
                if loss_type == "wgan":
                    for p in D.backbone_parameters():
                        p.data.clamp_(-0.01, 0.01)

            epoch_d.append(d_loss.item())

            # ── Generator update ─────────────────────────────────────────────
            opt_g.zero_grad()

            z           = torch.randn(batch, Z_DIM, 1, 1, device=device)
            fake        = G(z)
            fake_score, _ = D(fake)

            if loss_type == "standard":
                g_loss = standard_g(fake_score)
            elif loss_type == "lsgan":
                g_loss = lsgan_g(fake_score)
            elif loss_type == "wgan":
                g_loss = wgan_g(fake_score)
            elif loss_type == "wgangp":
                g_loss = wgangp_g(fake_score)
            elif loss_type == "hinge":
                g_loss = hinge_g(fake_score)
            elif loss_type == "hybrid":
                g_loss = hybrid_g(fake_score, real, fake)

            g_loss.backward()
            opt_g.step()
            epoch_g.append(g_loss.item())

        avg_g = float(np.mean(epoch_g))
        avg_d = float(np.mean(epoch_d))
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        save_samples(G, epoch, loss_type, out_dir)
        save_ckpt(G, D, opt_g, opt_d, epoch,
                  g_losses, d_losses, loss_type, out_dir)

        # ── Metrics every N epochs ────────────────────────────────────────────
        metrics = {}
        if inception is not None and (epoch + 1) % metric_every == 0:
            metrics = calc_all_metrics(G, loader, inception,
                                       n=1000, n_ssim_pairs=200)
            all_metrics[epoch + 1] = metrics
            best_saver.update(G, D, metrics, epoch)   # [FIX-7]

        mv      = float(np.var(fake.detach().cpu().numpy()))
        met_str = ""
        if metrics:
            met_str = (f" | FID:{metrics['fid']:.2f} IS:{metrics['is_mean']:.2f}"
                       f" Prec:{metrics['precision']:.3f} Rec:{metrics['recall']:.3f}"
                       f" SSIM:{metrics['ssim_mean']:.3f}")
        print(f"  Ep [{epoch+1:3d}/{epochs}] G:{avg_g:7.4f} D:{avg_d:7.4f} "
              f"Var:{mv:.4f}{met_str}")

    save_final(G, D, loss_type, out_dir)
    return G, g_losses, d_losses, all_metrics


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS  (loss curves + all 4 metrics)
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "standard": "red",   "lsgan"  : "green",
    "wgan"    : "purple","wgangp" : "blue",
    "hinge"   : "orange","hybrid" : "teal"
}


def _plot_metric(all_metrics_dict, key, title, ylabel, out_path,
                 lower_better=True):
    plt.figure(figsize=(10, 5))
    for name, m_dict in all_metrics_dict.items():
        if not m_dict:
            continue
        xs = sorted(m_dict.keys())
        ys = [m_dict[x][key] for x in xs if key in m_dict[x]]
        if ys:
            plt.plot(xs, ys, marker='o', label=name, color=COLORS[name])
    note = "(lower is better)" if lower_better else "(higher is better)"
    plt.title(f"{title} — {note}")
    plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()
    print(f"  Plot → {out_path}")


def plot_all(all_g, all_d, all_metrics_dict, out_dir):
    # Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for name, v in all_g.items():
        axes[0].plot(v, label=name, color=COLORS[name])
    axes[0].set_title("Generator Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(True)

    for name, v in all_d.items():
        axes[1].plot(v, label=name, color=COLORS[name])
    axes[1].set_title("Discriminator Loss"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss"); axes[1].legend(); axes[1].grid(True)

    plt.suptitle("DCGAN CheXpert — Loss Curves", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "chexpert_loss_curves.png")
    plt.savefig(p, dpi=300); plt.close(); print(f"  Plot → {p}")

    # FID
    _plot_metric(all_metrics_dict, "fid",
                 "FID Score", "FID (↓ better)",
                 os.path.join(out_dir, "chexpert_fid.png"),
                 lower_better=True)

    # IS
    _plot_metric(all_metrics_dict, "is_mean",
                 "Inception Score", "IS (↑ better)",
                 os.path.join(out_dir, "chexpert_is.png"),
                 lower_better=False)

    # Precision & Recall  (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for name, m_dict in all_metrics_dict.items():
        if not m_dict:
            continue
        xs = sorted(m_dict.keys())
        pr_y = [m_dict[x]["precision"] for x in xs if "precision" in m_dict[x]]
        rc_y = [m_dict[x]["recall"]    for x in xs if "recall"    in m_dict[x]]
        if pr_y:
            axes[0].plot(xs, pr_y, marker='o', label=name, color=COLORS[name])
            axes[1].plot(xs, rc_y, marker='s', label=name,
                         color=COLORS[name], linestyle='--')
    axes[0].set_title("Precision (fidelity, ↑ better)")
    axes[1].set_title("Recall (diversity, ↑ better)")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.legend(); ax.grid(True)
    plt.suptitle("GAN Precision & Recall — CheXpert", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "chexpert_precision_recall.png")
    plt.savefig(p, dpi=300); plt.close(); print(f"  Plot → {p}")

    # SSIM
    _plot_metric(all_metrics_dict, "ssim_mean",
                 "SSIM (medical structural similarity)", "SSIM (↑ better)",
                 os.path.join(out_dir, "chexpert_ssim.png"),
                 lower_better=False)


# ──────────────────────────────────────────────────────────────────────────────
# RUN EXPERIMENTS
# ──────────────────────────────────────────────────────────────────────────────

def run_experiments(epochs=300, metric_every=10,
                    uncertainty_policy="ignore"):

    out_dir = os.path.join(os.path.expanduser("~"), "project", "chexpert_output")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output : {out_dir}")
    print(f"Dataset: {DATA_ROOT}")

    print("\nLoading Inception v3...")
    inception = InceptionFeatureExtractor().to(device).eval()
    print("Inception loaded\n")

    experiments = ["standard", "lsgan", "wgan", "wgangp", "hinge", "hybrid"]

    print(f"{'#'*66}")
    print(f"  Dataset      : CheXpert-v1.0-small (frontal views)")
    print(f"  Architecture : DCGAN (ngf={NGF}, ndf={NDF}) + 14-class aux head")
    print(f"  Uncertainty  : policy='{uncertainty_policy}'")
    print(f"  Metrics      : FID | IS | Precision | Recall | SSIM")
    print(f"  Experiments  : {experiments}")
    print(f"  Epochs       : {epochs}  |  Metric every: {metric_every}")
    print(f"  Device       : {device}")
    print(f"{'#'*66}")

    results                        = []
    all_g, all_d, all_metrics_dict = {}, {}, {}

    for i, loss_type in enumerate(experiments):
        print(f"\n{'*'*66}")
        print(f"  Experiment {i+1}/{len(experiments)}: {loss_type.upper()}")
        print(f"{'*'*66}")

        model, g_curve, d_curve, m_dict = train(
            loss_type, out_dir,
            epochs=epochs,
            inception=inception,
            metric_every=metric_every,
            uncertainty_policy=uncertainty_policy
        )

        all_g[loss_type]           = g_curve
        all_d[loss_type]           = d_curve
        all_metrics_dict[loss_type] = m_dict

        # Final epoch metrics
        last_m = m_dict.get(epochs) or (
            m_dict[max(m_dict)] if m_dict else {})

        # Mode variance (pixel-level)
        z = torch.randn(200, Z_DIM, 1, 1, device=device)
        with torch.no_grad():
            fake = model(z)
        var = float(np.var(fake.detach().cpu().numpy()))

        results.append({
            "Exp_No"            : i + 1,
            "Dataset"           : "CheXpert-v1.0-small",
            "Architecture"      : "DCGAN",
            "Optimizer"         : "Adam",
            "Loss_Type"         : loss_type,
            "Uncertainty_Policy": uncertainty_policy,
            "Epochs"            : epochs,
            "Final_G_Loss"      : round(g_curve[-1], 4),
            "Final_D_Loss"      : round(d_curve[-1], 4),
            # All 5 metrics
            "FID"               : last_m.get("fid",       "N/A"),
            "IS_mean"           : last_m.get("is_mean",   "N/A"),
            "IS_std"            : last_m.get("is_std",    "N/A"),
            "Precision"         : last_m.get("precision", "N/A"),
            "Recall"            : last_m.get("recall",    "N/A"),
            "SSIM_mean"         : last_m.get("ssim_mean", "N/A"),
            "SSIM_std"          : last_m.get("ssim_std",  "N/A"),
            "ModeVariance"      : round(var, 4),
        })

        pd.DataFrame(results).to_csv(
            os.path.join(out_dir, "chexpert_results_partial.csv"), index=False)
        print("  Partial CSV saved")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "chexpert_results.csv"), index=False)
    print(f"\nFinal Results:\n{df.to_string(index=False)}")

    plot_all(all_g, all_d, all_metrics_dict, out_dir)

    print(f"\n{'#'*66}")
    print("  ALL 6 EXPERIMENTS COMPLETE")
    print(f"  Output: {out_dir}")
    print(f"{'#'*66}")
    print("""
  Files generated:
  ├── chexpert_results.csv          ← all metrics per experiment
  ├── chexpert_loss_curves.png
  ├── chexpert_fid.png
  ├── chexpert_is.png               ← Inception Score (NEW)
  ├── chexpert_precision_recall.png ← P&R (NEW, publication key)
  ├── chexpert_ssim.png             ← SSIM medical (NEW)
  ├── samples/
  │   ├── standard/ lsgan/ wgan/ wgangp/ hinge/ hybrid/
  ├── models/
  │   ├── final/   *_G.pth  *_D.pth
  │   └── best/    *_G_best.pth  *_D_best.pth   (best FID)
  └── checkpoints/
      └── standard/ lsgan/ wgan/ wgangp/ hinge/ hybrid/
    """)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_experiments(
        epochs=300,
        metric_every=10,
        uncertainty_policy="ignore"    # 'ignore' | 'zeros' | 'ones'
    )