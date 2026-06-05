import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "cifer_all_6.csv"  # change as needed

df = pd.read_csv(CSV_FILE)
df["FID"] = df["FID"].fillna(method="ffill")

experiments = df["Experiment"].unique()

# Generator Loss
plt.figure(figsize=(10,6))
for exp in experiments:
    temp = df[df["Experiment"] == exp]
    plt.plot(temp["Epoch"], temp["G_Loss"], label=exp)
plt.title("Generator Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Generator Loss")
plt.legend()
plt.grid(True)
plt.savefig("generator_loss.png", dpi=300)
plt.close()

# Discriminator Loss
plt.figure(figsize=(10,6))
for exp in experiments:
    temp = df[df["Experiment"] == exp]
    plt.plot(temp["Epoch"], temp["D_Loss"], label=exp)
plt.title("Discriminator Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Discriminator Loss")
plt.legend()
plt.grid(True)
plt.savefig("discriminator_loss.png", dpi=300)
plt.close()

# FID Score
plt.figure(figsize=(10,6))
for exp in experiments:
    temp = df[df["Experiment"] == exp].dropna(subset=["FID"])
    plt.plot(temp["Epoch"], temp["FID"], marker="o", label=exp)
plt.title("FID Score vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.legend()
plt.grid(True)
plt.savefig("fid_score.png", dpi=300)
plt.close()

# Mode Variance
plt.figure(figsize=(10,6))
for exp in experiments:
    temp = df[df["Experiment"] == exp]
    plt.plot(temp["Epoch"], temp["ModeVar"], label=exp)
plt.title("Mode Variance vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mode Variance")
plt.legend()
plt.grid(True)
plt.savefig("mode_variance.png", dpi=300)
plt.close()

# Best FID Comparison
best_fid = df.dropna(subset=["FID"]).groupby("Experiment")["FID"].min().reset_index()
best_fid = best_fid.sort_values("FID")

plt.figure(figsize=(10,6))
bars = plt.bar(best_fid["Experiment"], best_fid["FID"])
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, y, f"{y:.2f}", ha="center")
plt.title("Loss Function Comparison (Best FID)")
plt.xlabel("Loss Function")
plt.ylabel("Best FID")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("loss_function_comparison.png", dpi=300)
plt.close()

print("All graphs generated.")
