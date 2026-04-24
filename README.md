# Comparative Analysis of GAN Loss Functions
Comparative Analysis of GAN Loss FunctionsUsing DCGAN Architecture for Natural and Remote Sensing Image Synthesis

# 🧠 GAN Loss Function Comparison using DCGAN

## 📌 Overview

This project presents a systematic comparison of different Generative Adversarial Network (GAN) loss functions using a unified DCGAN architecture. The goal is to analyze how various loss functions affect training stability, image quality, and diversity.

The study evaluates five widely used GAN loss functions under identical experimental conditions.

---

## 🎯 Objectives

- Compare performance of GAN loss functions:
  - Standard GAN (BCE)
  - LSGAN (MSE)
  - WGAN
  - WGAN-GP
  - Hinge Loss

- Evaluate using:
  - Frechet Inception Distance (FID)
  - Mode Variance (diversity)
  - Training stability (loss curves)

- Perform cross-domain analysis:
  - CIFAR-10 (natural images)
  - EuroSAT (satellite images)

---

## 🏗️ Architecture

The experiments use a **Deep Convolutional GAN (DCGAN)** architecture:

- **Generator**:
  - Input: 100-dim noise vector
  - ConvTranspose layers → 64×64 RGB image

- **Discriminator**:
  - Input: 64×64 image
  - Conv layers → real/fake score

> The architecture is kept constant to ensure fair comparison across loss functions.

---

## ⚙️ Environment Setup

### Requirements

- Python 3.11
- PyTorch
- torchvision
- numpy
- scipy
- matplotlib
- pillow

### Installation

```bash
# Create virtual environment
python -m venv gan_env

# Activate environment (Windows)
gan_env\Scripts\activate

# Install dependencies
pip install torch torchvision numpy scipy matplotlib pillow