# 🧠 Unified DCGAN Framework for Multi-Dataset Image Generation

## 📌 Overview

This repository provides a **comprehensive DCGAN-based framework** for training and evaluating Generative Adversarial Networks across **multiple datasets and domains**:

* 🏥 Medical Imaging → **CheXpert**
* 🌍 Remote Sensing → **EuroSAT**
* 🧪 Benchmark Dataset → **CIFAR-10**

The framework is designed for:

* GAN research & experimentation
* Loss function comparison
* Multi-domain generalization
* HPC-based large-scale training

---

## 🚀 Key Features

### 🧩 Multi-Dataset Support

* CheXpert (Medical X-rays)
* EuroSAT (Satellite images)
* CIFAR-10 (Standard benchmark)

---

### ⚙️ GAN Variants Implemented

* Standard GAN
* LSGAN
* WGAN
* WGAN-GP
* Hinge Loss
* Hybrid GAN (only CheXpert)

---

### 🧠 Advanced Capabilities

#### CheXpert (Medical AI)

* Multi-label classification (14 diseases)
* Uncertainty handling (`-1 labels`)
* Auxiliary classifier GAN

#### General Features

* FID Score evaluation (InceptionV3)
* Mode collapse detection (variance)
* Automatic:

  * Checkpointing
  * Best model saving
  * Sample generation
  * CSV logging

---

## 📂 Project Structure

```id="proj123"
project/
│
├── chexpert_dcgan.py
├── cxpert_dcgan_hybrid.py
├── test_eurosat.py
├── test_cifar.py
│
├── dataset/
│   ├── CheXpert-v1.0-small/
│   ├── eurosat/
│   └── cifar-10-batches-py_small/
│
└── output/
    ├── samples/
    ├── models/
    ├── checkpoints/
    └── results.csv
```

---

## 📊 Datasets

### 1. 🏥 CheXpert (Medical)

```id="c1"
~/dataset/CheXpert-v1.0-small/
```

* 224K chest X-rays
* 14 pathology labels
* Multi-label classification

---

### 2. 🌍 EuroSAT (Satellite)

```id="c2"
~/dataset/eurosat/EuroSAT_RGB/
```

* Land-use classification dataset
* RGB satellite images

---

### 3. 🧪 CIFAR-10

```id="c3"
~/dataset/cifar-10-batches-py_small/
```

* 60,000 images
* 10 classes

---

## 📥 Dataset Setup (HPC)

### CheXpert

```bash id="d1"
kaggle datasets download -d ashery/chexpert
unzip chexpert.zip
```

### EuroSAT

```bash id="d2"
scp -r EuroSAT_RGB user@hpc:~/dataset/eurosat/
```

### CIFAR-10

```bash id="d3"
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

---

## ⚙️ Installation

```bash id="inst1"
pip install torch torchvision numpy pandas matplotlib pillow scipy
```

---

## 🏃 Training

### ▶ CheXpert (Full Research Model)

```bash id="run1"
python chexpert_dcgan.py
```

### ▶ EuroSAT

```bash id="run2"
python test_e_1.py
```

### ▶ CIFAR-10

```bash id="run3"
python test_c1.py
```

---

## ⚡ Training Configuration

| Parameter     | Value      |
| ------------- | ---------- |
| Image Size    | 64×64      |
| Latent Vector | 100        |
| Batch Size    | 32 / 64    |
| Optimizer     | Adam       |
| Device        | GPU (CUDA) |

---

## 📈 Evaluation Metrics

### 🔹 FID Score

Measures similarity between real and generated images.

### 🔹 Mode Variance

Detects mode collapse.

### 🔹 Loss Curves

* Generator Loss
* Discriminator Loss

---

## 🧠 Architecture

### Generator

* Input: Noise (z ∈ R¹⁰⁰)
* Output: 64×64 RGB image

### Discriminator

* CNN-based
* Outputs:

  * Real/Fake score
  * (CheXpert only) Multi-label classification

---

## 🧪 Experiments

### CheXpert

* 6 GAN variants
* Medical classification + generation

### EuroSAT & CIFAR

* 5 GAN variants
* Pure image generation

---

## 📁 Outputs

```id="out1"
output/
├── samples/
├── models/
│   ├── final/
│   └── best/
├── checkpoints/
├── *_results.csv
├── *_loss_curves.png
└── *_fid_scores.png
```

---

## 🔬 Research Contributions

* Multi-domain GAN evaluation
* Comparison of 5–6 loss functions
* Medical uncertainty label handling
* Integration of classification + generation
* HPC-ready training pipeline

---

## ⚠️ Important Notes

* All images resized to **64×64**
* CheXpert:

  * Uses **frontal images only**
  * Converts grayscale → RGB
* WGAN requires:

  * Weight clipping OR gradient penalty

---

## 🔧 Customization

Modify parameters inside code:

```python id="conf1"
IMG_SIZE = 64
Z_DIM = 100
BATCH_SIZE = 32
```

---

## 📌 Future Work

* Conditional GAN (cGAN)
* StyleGAN / Diffusion models
* Multi-modal medical datasets
* Higher resolution synthesis

---

## 👨‍💻 Author

Sanskar Kushwah
M.Tech Computer Science
NIT Srinagar

---

## 📜 License

Academic & research use only.

---

## ⭐ Support

If this helps your research:

* Star ⭐ the repo
* Cite in your paper
