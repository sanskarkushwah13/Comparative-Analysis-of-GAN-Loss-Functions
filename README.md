# Comparative Analysis of GAN Loss Functions

Comparative analysis of GAN loss functions using a DCGAN architecture for cross-domain image synthesis — natural images, remote sensing, and medical imaging.

IEEE Access paper: *A Comparative Analysis of GAN Loss Functions for Cross-Domain Image Synthesis* — Kushwah & Shaikh, NIT Srinagar (2026)

## Results

FID (Fréchet Inception Distance, lower is better) across three datasets. Each model was trained for 100 epochs (CIFAR-10), 150 epochs (EuroSAT), and 100 epochs (CheXpert), evaluated on 2,000 real + 2,000 generated images.

| Loss              | CIFAR-10 FID↓ | EuroSAT FID↓ | CheXpert FID↓ |
| ----------------- | ------------- | ------------ | ------------- |
| Standard GAN      | 363.74        | 40.90        | —¹            |
| LSGAN             | 161.46        | 39.91        | —¹            |
| WGAN              | 102.89        | 102.88       | 58.84²        |
| WGAN-GP           | **58.86**     | 95.55        | —¹            |
| Hinge Loss        | 185.02        | 48.88        | —¹            |
| **Hybrid (ours)** | 61.34         | 43.17        | **45.98**     |

¹ Not evaluated on CheXpert — *(fill in reason: compute budget / scope limited to best-performing baselines, etc.)*
² WGAN on CheXpert suffered critic loss divergence due to overly tight weight clipping (c = 0.01); reported value reflects pre-divergence checkpoint.

**Key findings:**
- No single loss function dominates across all three domains.
- WGAN-GP performs best on natural images (CIFAR-10).
- LSGAN performs best on structurally regular satellite imagery (EuroSAT).
- The proposed Hybrid Loss (WGAN-GP + Hinge) achieves the best FID on medical imaging (CheXpert) and the highest sample diversity (IBFV = 0.784) across all three datasets.

## Repository Structure

```
.
├── comparison_loss_function/   # Baseline loss function experiments (Standard GAN, LSGAN, WGAN, WGAN-GP, Hinge)
├── hybrid_loss_function/       # Proposed Hybrid Loss (WGAN-GP + Hinge) implementation and experiments
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt
python src/train.py --loss hybrid --dataset cifar10 --epochs 100
```

## Datasets

- **CIFAR-10**: auto-downloads via `torchvision`
- **EuroSAT**: [download link](https://madm.dfki.de/files/sentinel/EuroSAT.zip) → `data/EuroSAT/`
- **CheXpert**: [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/) → `data/CheXpert/` (Kaggle mirror used for prototyping: [link](https://www.kaggle.com/code/sanskarkushwah13/cxpertdataset/edit))

## Reproduce All Results


jikji
```bash
bash experiments/cifar10/run_all.sh
bash experiments/eurosat/run_all.sh
bash experiments/chexpert/run_all.sh
```

## Citation

```bibtex
@article{sanskarkushwah,
  title   = {A Comparative Analysis of GAN Loss Functions for Cross-Domain Image Synthesis},
  author  = {Kushwah, Sanskar and Shaikh, Tawseef Ayoub},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## Acknowledgements

Experiments were conducted on the NIT Srinagar HPC facility (NVIDIA H100, SLURM).