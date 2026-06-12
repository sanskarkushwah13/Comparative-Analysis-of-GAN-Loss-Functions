import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import linalg
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ HPC EuroSAT path
DATA_ROOT = os.path.expanduser("~/dataset/eurosat/EuroSAT_RGB")

# =========================
# EUROSAT DATASET
# =========================

class EuroSATDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images    = []
        self.labels    = []

        classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        self.classes = classes

        for label, cls in enumerate(classes):
            folder = os.path.join(root, cls)
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    self.images.append(os.path.join(folder, fname))
                    self.labels.append(label)

        print(f"  Classes ({len(classes)}): {classes}")
        print(f"  Total Images: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(
            f"\n❌ EuroSAT not found: {DATA_ROOT}\n"
            f"Upload dataset:\n"
            f"  scp -r EuroSAT_RGB/ 2024mcsecs014@hpc2:~/dataset/eurosat/\n"
        )

    dataset = EuroSATDataset(root=DATA_ROOT, transform=transform)

    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True,
                      num_workers=1,      # HPC = 1
                      drop_last=True)     # ✅ WGAN-GP fix


# =========================
# DCGAN GENERATOR
# =========================

class Generator(nn.Module):
    """
    z(100,1,1) → 512x4 → 256x8 → 128x16 → 64x32 → 3x64
    """
    def __init__(self, z_dim=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(

            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)


# =========================
# DCGAN DISCRIMINATOR
# =========================

class Discriminator(nn.Module):
    """
    3x64 → 64x32 → 128x16 → 256x8 → 512x4 → 1
    """
    def __init__(self, ndf=64, use_sigmoid=False):
        super().__init__()
        layers = [
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

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())  # ✅ Only for Standard GAN
        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# =========================
# LOSS FUNCTIONS (ALL 5)
# =========================

bce = nn.BCELoss()
mse = nn.MSELoss()

# ── 1. Standard GAN ────────────────────────────────────────────────────────
def standard_d(D, real, fake):
    real_loss = bce(D(real), torch.ones(real.size(0)).to(device))
    fake_loss = bce(D(fake.detach()), torch.zeros(fake.size(0)).to(device))
    return (real_loss + fake_loss) / 2

def standard_g(fake_score):
    return bce(fake_score, torch.ones(fake_score.size(0)).to(device))

# ── 2. LSGAN ───────────────────────────────────────────────────────────────
def lsgan_d(D, real, fake):
    real_loss = 0.5 * mse(D(real), torch.ones(real.size(0)).to(device))
    fake_loss = 0.5 * mse(D(fake.detach()), torch.zeros(fake.size(0)).to(device))
    return real_loss + fake_loss

def lsgan_g(fake_score):
    return 0.5 * mse(fake_score, torch.ones(fake_score.size(0)).to(device))

# ── 3. WGAN ────────────────────────────────────────────────────────────────
def wgan_d(D, real, fake, clip=0.01):
    loss = -(torch.mean(D(real)) - torch.mean(D(fake.detach())))
    # Weight clipping
    for p in D.parameters():
        p.data.clamp_(-clip, clip)
    return loss

def wgan_g(fake_score):
    return -torch.mean(fake_score)

# ── 4. WGAN-GP ─────────────────────────────────────────────────────────────
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

def wgangp_d(D, real, fake, lambda_gp=10):
    return -(torch.mean(D(real)) - torch.mean(D(fake.detach()))) + \
            lambda_gp * gradient_penalty(D, real, fake)

def wgangp_g(fake_score):
    return -torch.mean(fake_score)

# ── 5. Hinge ───────────────────────────────────────────────────────────────
def hinge_d(real_score, fake_score):
    return torch.mean(torch.relu(1.0 - real_score)) + \
           torch.mean(torch.relu(1.0 + fake_score))

def hinge_g(fake_score):
    return -torch.mean(fake_score)


# =========================
# FID SCORE
# =========================

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights
        inc = torchvision.models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1
        )
        inc.aux_logits = False
        self.features  = nn.Sequential(
            inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3,
            inc.Conv2d_2b_3x3, nn.MaxPool2d(3, stride=2),
            inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
            inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c,
            inc.Mixed_6d, inc.Mixed_6e,
            inc.Mixed_7a, inc.Mixed_7b, inc.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(299, 299),
                                       mode='bilinear', align_corners=False)
        x = (x + 1) / 2.0
        return self.features(x).view(x.size(0), -1)


def get_features(imgs, model, bs=50):
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        b = imgs[i:i+bs].to(device)
        with torch.no_grad():
            out.append(model(b).cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_fid(rf, ff):
    rf = rf.astype(np.float64)
    ff = ff.astype(np.float64)
    mu_r,  mu_f  = np.mean(rf, 0), np.mean(ff, 0)
    sig_r, sig_f = np.cov(rf, rowvar=False), np.cov(ff, rowvar=False)
    eps          = 1e-6
    sig_r += np.eye(sig_r.shape[0]) * eps
    sig_f += np.eye(sig_f.shape[0]) * eps
    diff       = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = (diff @ diff) + np.trace(sig_r + sig_f - 2.0 * covmean)
    return float(abs(fid))


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
        cur = min(64, n - done)
        z   = torch.randn(cur, 100, 1, 1).to(device)
        with torch.no_grad():
            fake_imgs.append(G(z).cpu())
        done += cur
    fake_imgs = torch.cat(fake_imgs)[:n]
    G.train()

    fid = compute_fid(
        get_features(real_imgs, inception),
        get_features(fake_imgs, inception)
    )
    print(f"    FID: {fid:.4f} ✅")
    return fid


# =========================
# SAVE UTILS
# =========================

def save_samples(G, epoch, loss_type, out_dir):
    z      = torch.randn(64, 100, 1, 1).to(device)
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
    print(f"  ✅ Model saved → {folder}")


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
    print(f"  💾 Checkpoint → epoch_{epoch+1:03d}.pth")


class BestSaver:
    def __init__(self, loss_type, out_dir):
        self.best      = float("inf")
        self.loss_type = loss_type
        self.folder    = os.path.join(out_dir, "models", "best")
        os.makedirs(self.folder, exist_ok=True)

    def update(self, G, D, g_loss, epoch):
        if g_loss < self.best:
            self.best = g_loss
            torch.save(G.state_dict(),
                       os.path.join(self.folder,
                                    f"{self.loss_type}_G_best.pth"))
            torch.save(D.state_dict(),
                       os.path.join(self.folder,
                                    f"{self.loss_type}_D_best.pth"))
            print(f"  🏆 Best epoch {epoch+1} | G: {g_loss:.4f}")


# =========================
# TRAINING LOOP
# =========================

def train(loss_type, out_dir, epochs=50, inception=None, fid_every=10):

    print(f"\n{'='*60}")
    print(f"  DCGAN + EuroSAT | {loss_type.upper()} | {epochs} Epochs")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    loader = get_dataloader(batch_size=64)

    # ✅ Sigmoid only for Standard GAN
    use_sigmoid = (loss_type == "standard")
    G = Generator(z_dim=100, ngf=64).to(device)
    D = Discriminator(ndf=64, use_sigmoid=use_sigmoid).to(device)

    print(f"  G params : {sum(p.numel() for p in G.parameters()):,}")
    print(f"  D params : {sum(p.numel() for p in D.parameters()):,}")

    # ✅ Adam settings per loss type
    if loss_type in ["wgan", "wgangp"]:
        opt_g    = optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.9))
        opt_d    = optim.Adam(D.parameters(), lr=0.0004, betas=(0.0, 0.9))
        n_critic = 5
        print("  Adam TTUR | lr_G=0.0001 | lr_D=0.0004 | n_critic=5")
    else:
        opt_g    = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_d    = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        n_critic = 1
        print("  Adam | lr=0.0002 | betas=(0.5,0.999) | n_critic=1")

    g_losses, d_losses = [], []
    fid_scores         = {}
    best_saver         = BestSaver(loss_type, out_dir)

    for epoch in range(epochs):
        epoch_g, epoch_d = [], []

        for real, _ in loader:
            real  = real.to(device)
            batch = real.size(0)

            # ── Discriminator ──────────────────────────────────────────
            for _ in range(n_critic):
                z    = torch.randn(batch, 100, 1, 1).to(device)
                fake = G(z).detach()

                if loss_type == "standard":
                    d_loss = standard_d(D, real, fake)
                elif loss_type == "lsgan":
                    d_loss = lsgan_d(D, real, fake)
                elif loss_type == "wgan":
                    d_loss = wgan_d(D, real, fake)
                elif loss_type == "wgangp":
                    fake_gp = G(torch.randn(batch, 100, 1, 1).to(device))
                    d_loss  = wgangp_d(D, real, fake_gp)
                elif loss_type == "hinge":
                    d_loss  = hinge_d(D(real), D(fake))

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            epoch_d.append(d_loss.item())

            # ── Generator ──────────────────────────────────────────────
            z         = torch.randn(batch, 100, 1, 1).to(device)
            fake      = G(z)
            fake_score = D(fake)

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

            opt_g.zero_grad()
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
        best_saver.update(G, D, avg_g, epoch)

        fid = None
        if inception is not None and (epoch + 1) % fid_every == 0:
            fid = calc_fid(G, loader, inception, n=2000)
            fid_scores[epoch + 1] = fid

        mv      = float(np.var(fake.detach().cpu().numpy()))
        fid_str = f" | FID: {fid:.4f}" if fid else ""
        print(f"  Epoch [{epoch+1:3d}/{epochs}] | "
              f"G: {avg_g:7.4f} | D: {avg_d:7.4f} | "
              f"ModeVar: {mv:.4f}{fid_str}")

    save_final(G, D, loss_type, out_dir)
    return G, g_losses, d_losses, fid_scores


# =========================
# PLOTS
# =========================

def plot_all(all_g, all_d, all_fid, out_dir):
    colors = {
        "standard" : "red",
        "lsgan"    : "green",
        "wgan"     : "purple",
        "wgangp"   : "blue",
        "hinge"    : "orange"
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for name, v in all_g.items():
        axes[0].plot(v, label=name, color=colors[name])
    axes[0].set_title("Generator Loss — EuroSAT")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    for name, v in all_d.items():
        axes[1].plot(v, label=name, color=colors[name])
    axes[1].set_title("Discriminator Loss — EuroSAT")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("DCGAN EuroSAT — All 5 Loss Functions (50 Epochs)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "eurosat_loss_curves.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Loss plot → {path}")

    plt.figure(figsize=(10, 6))
    for name, scores in all_fid.items():
        if scores:
            plt.plot(list(scores.keys()), list(scores.values()),
                     marker='o', label=name, color=colors[name])
    plt.title("FID Score — All 5 Loss Functions (Lower = Better)")
    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.legend()
    plt.grid(True)
    path = os.path.join(out_dir, "eurosat_fid_scores.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ FID plot → {path}")


# =========================
# RUN EXPERIMENTS
# =========================

def run_experiments(epochs=150, fid_every=10):

    # Output folder
    out_dir = os.path.join(os.path.expanduser("~"), "project", "euro", "eurosat_output")
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output : {out_dir}")
    print(f"📂 Dataset: {DATA_ROOT}")

    # Inception for FID
    print("\nLoading Inception v3...")
    inception = InceptionFeatureExtractor().to(device).eval()
    print("✅ Inception loaded\n")

    # ✅ All 5 experiments
    experiments = ["standard", "lsgan", "wgan", "wgangp", "hinge"]

    print(f"{'#'*60}")
    print(f"  DATASET      : EuroSAT RGB")
    print(f"  ARCHITECTURE : DCGAN (ngf=64, ndf=64)")
    print(f"  OPTIMIZER    : Adam")
    print(f"  EXPERIMENTS  : {experiments}")
    print(f"  EPOCHS       : {epochs} each")
    print(f"  TOTAL EPOCHS : {len(experiments) * epochs}")
    print(f"  DEVICE       : {device}")
    print(f"{'#'*60}")

    results               = []
    all_g, all_d, all_fid = {}, {}, {}

    for i, loss_type in enumerate(experiments):
        print(f"\n{'*'*60}")
        print(f"  ▶  Experiment {i+1}/{len(experiments)}: {loss_type.upper()}")
        print(f"{'*'*60}")

        model, g_curve, d_curve, fid_scores = train(
            loss_type, out_dir,
            epochs=epochs,
            inception=inception,
            fid_every=fid_every
        )

        all_g[loss_type]   = g_curve
        all_d[loss_type]   = d_curve
        all_fid[loss_type] = fid_scores

        final_fid = fid_scores.get(epochs, None)
        if final_fid is None and fid_scores:
            final_fid = list(fid_scores.values())[-1]

        z = torch.randn(200, 100, 1, 1).to(device)
        with torch.no_grad():
            fake = model(z)
        var = float(np.var(fake.detach().cpu().numpy()))

        results.append({
            "Exp_No"       : i + 1,
            "Dataset"      : "EuroSAT_RGB",
            "Architecture" : "DCGAN",
            "Optimizer"    : "Adam",
            "Loss_Type"    : loss_type,
            "Epochs"       : epochs,
            "Final_G_Loss" : round(g_curve[-1], 4),
            "Final_D_Loss" : round(d_curve[-1], 4),
            "FID_Score"    : round(final_fid, 4) if final_fid else "N/A",
            "ModeVariance" : round(var, 4),
        })

        # Save partial CSV after each experiment
        pd.DataFrame(results).to_csv(
            os.path.join(out_dir, "eurosat_results_partial.csv"), index=False)

    # Final CSV
    df       = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "eurosat_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Final Results → {csv_path}")
    print("\n", df.to_string(index=False))

    plot_all(all_g, all_d, all_fid, out_dir)

    print(f"\n{'#'*60}")
    print("  🎉 ALL 5 EXPERIMENTS COMPLETE!")
    print(f"  📁 Output: {out_dir}")
    print(f"{'#'*60}")
    print(f"""
  Files:
  ├── eurosat_results.csv
  ├── eurosat_loss_curves.png
  ├── eurosat_fid_scores.png
  ├── samples/
  │   ├── standard/  lsgan/  wgan/  wgangp/  hinge/
  ├── models/
  │   ├── final/   *_G.pth  *_D.pth
  │   └── best/    *_G_best.pth
  └── checkpoints/
      └── standard/ lsgan/ wgan/ wgangp/ hinge/
    """)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    run_experiments(epochs=50, fid_every=10)
