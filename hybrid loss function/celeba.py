"""
train_gans_celeba.py
Hybrid Loss (L_alpha = alpha*L_WGAN-GP + (1-alpha)*L_Hinge) on CelebA
DCGAN backbone -- consistent with CIFAR-10 / EuroSAT / CheXpert protocol
FID pool: 2000 real + 2000 fake | Epochs: 100 | GPU: NVIDIA H100
"""

import os
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from scipy import linalg
from torchvision.models import inception_v3

# --------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="./data/celeba")
parser.add_argument("--out_dir", type=str, default="./outputs_celeba")
parser.add_argument("--image_size", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--nz", type=int, default=100)
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--n_critic", type=int, default=1)
parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid loss weight: alpha*WGAN-GP + (1-alpha)*Hinge")
parser.add_argument("--lambda_gp", type=float, default=10.0)
parser.add_argument("--fid_pool", type=int, default=2000)
parser.add_argument("--fid_every", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------
# Data: CelebA
# --------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# torchvision CelebA requires manual download flag=False if already present in data_root
dataset = datasets.CelebA(root=args.data_root, split="train", download=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=8, pin_memory=True, drop_last=True)

# --------------------------------------------------------------------------
# DCGAN Backbone (identical topology to CIFAR-10 / EuroSAT / CheXpert runs)
# --------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """No sigmoid at output -- raw critic score, used for both WGAN-GP and Hinge terms."""
    def __init__(self, ndf, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netG = Generator(args.nz, args.ngf).to(device)
netD = Discriminator(args.ndf).to(device)
netG.apply(weights_init)
netD.apply(weights_init)

optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# --------------------------------------------------------------------------
# Gradient Penalty (for WGAN-GP term)
# --------------------------------------------------------------------------
def gradient_penalty(D, real, fake, device):
    b = real.size(0)
    eps = torch.rand(b, 1, 1, 1, device=device).expand_as(real)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp = D(interp)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grads = grads.view(b, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# --------------------------------------------------------------------------
# Hybrid Loss: L_alpha = alpha * L_WGAN-GP + (1 - alpha) * L_Hinge
# --------------------------------------------------------------------------
def d_loss_hybrid(D, real, fake, alpha, lambda_gp, device):
    d_real = D(real)
    d_fake = D(fake.detach())

    # WGAN-GP critic loss
    loss_wgan = d_fake.mean() - d_real.mean()
    gp = gradient_penalty(D, real, fake.detach(), device)
    loss_wgan_gp = loss_wgan + lambda_gp * gp

    # Hinge discriminator loss
    loss_hinge = torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()

    return alpha * loss_wgan_gp + (1 - alpha) * loss_hinge


def g_loss_hybrid(D, fake, alpha):
    d_fake = D(fake)
    loss_wgan_g = -d_fake.mean()
    loss_hinge_g = -d_fake.mean()
    return alpha * loss_wgan_g + (1 - alpha) * loss_hinge_g


# --------------------------------------------------------------------------
# FID (Inception-v3 pool-3 features)
# --------------------------------------------------------------------------
inception = inception_v3(weights="IMAGENET1K_V1", transform_input=False).to(device)
inception.fc = nn.Identity()
inception.eval()

def get_inception_features(images):
    with torch.no_grad():
        images = torch.nn.functional.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = (images + 1) / 2  # [-1,1] -> [0,1]
        feats = inception(images)
    return feats.cpu().numpy()

def calculate_fid(real_feats, fake_feats):
    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r.dot(sigma_f), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)

def compute_fid(netG, dataloader, pool_size, device):
    netG.eval()
    real_batches, fake_batches = [], []
    collected = 0
    with torch.no_grad():
        for real, _ in dataloader:
            real = real.to(device)
            real_batches.append(get_inception_features(real))
            z = torch.randn(real.size(0), args.nz, 1, 1, device=device)
            fake = netG(z)
            fake_batches.append(get_inception_features(fake))
            collected += real.size(0)
            if collected >= pool_size:
                break
    netG.train()
    real_feats = np.concatenate(real_batches, axis=0)[:pool_size]
    fake_feats = np.concatenate(fake_batches, axis=0)[:pool_size]
    return calculate_fid(real_feats, fake_feats)


# --------------------------------------------------------------------------
# Training Loop
# --------------------------------------------------------------------------
csv_path = os.path.join(args.out_dir, "celeba_hybrid.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "d_loss", "g_loss", "fid", "time_sec"])

fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    d_loss_running, g_loss_running = 0.0, 0.0

    for i, (real, _) in enumerate(dataloader):
        real = real.to(device)
        b = real.size(0)

        # --- Train Discriminator ---
        for _ in range(args.n_critic):
            z = torch.randn(b, args.nz, 1, 1, device=device)
            fake = netG(z)
            optD.zero_grad()
            dloss = d_loss_hybrid(netD, real, fake, args.alpha, args.lambda_gp, device)
            dloss.backward()
            optD.step()

        # --- Train Generator ---
        z = torch.randn(b, args.nz, 1, 1, device=device)
        fake = netG(z)
        optG.zero_grad()
        gloss = g_loss_hybrid(netD, fake, args.alpha)
        gloss.backward()
        optG.step()

        d_loss_running += dloss.item()
        g_loss_running += gloss.item()

    d_loss_avg = d_loss_running / len(dataloader)
    g_loss_avg = g_loss_running / len(dataloader)

    fid_score = ""
    if epoch % args.fid_every == 0 or epoch == args.epochs:
        fid_score = compute_fid(netG, dataloader, args.fid_pool, device)
        print(f"[Epoch {epoch}/{args.epochs}] D_loss: {d_loss_avg:.4f} | G_loss: {g_loss_avg:.4f} | FID: {fid_score:.4f}")
    else:
        print(f"[Epoch {epoch}/{args.epochs}] D_loss: {d_loss_avg:.4f} | G_loss: {g_loss_avg:.4f}")

    elapsed = time.time() - t0
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, d_loss_avg, g_loss_avg, fid_score, elapsed])

    # Save sample grid
    with torch.no_grad():
        netG.eval()
        samples = netG(fixed_noise).detach().cpu()
        netG.train()
    vutils.save_image(samples, os.path.join(args.out_dir, "samples", f"epoch_{epoch:03d}.png"),
                       normalize=True, nrow=8)

    # Checkpoint every 10 epochs
    if epoch % 10 == 0 or epoch == args.epochs:
        torch.save(netG.state_dict(), os.path.join(args.out_dir, "checkpoints", f"netG_epoch{epoch}.pth"))
        torch.save(netD.state_dict(), os.path.join(args.out_dir, "checkpoints", f"netD_epoch{epoch}.pth"))

print("Training complete. Logs saved to:", csv_path)