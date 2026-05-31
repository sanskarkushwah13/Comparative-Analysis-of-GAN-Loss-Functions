# Comparative Analysis of GAN Loss Functions

IEEE Access paper: *A Comparative Analysis of GAN Loss Functions for 
Cross-Domain Image Synthesis* — Kushwah & Shaikh, NIT Srinagar (2026)

## Results

| Loss | CIFAR-10 FID↓ | EuroSAT FID↓ | CheXpert FID↓ |
|---|---|---|---|
| Standard GAN | 363.74 | 40.90 | — |
| LSGAN | 161.46 | 39.91 | — |
| WGAN | 102.89 | 102.88 | 58.84 |
| WGAN-GP | 58.86 | 95.55 | — |
| Hinge Loss | 185.02 | 48.88 | — |
| **Hybrid (ours)** | **61.34** | **43.17** | **45.98** |

## Quickstart

```bash
pip install -r requirements.txt
python src/train.py --loss hybrid --dataset cifar10 --epochs 200
```

## Datasets
- CIFAR-10: auto-downloads via torchvision  
- EuroSAT: [download link](https://madm.dfki.de/files/sentinel/EuroSAT.zip) → `data/EuroSAT/`  
- CheXpert: [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/) → `data/CheXpert/`
kaggle link : https://www.kaggle.com/code/sanskarkushwah13/cxpertdataset/edit 

## Reproduce all results
```bash
bash experiments/cifar10/run_all.sh
bash experiments/eurosat/run_all.sh
```

## Citation
```bibtex
@article{kushwah2026gan,
  title={A Comparative Analysis of GAN Loss Functions...},
  author={ Sanskar and Shaikh, Tawseef Ayoub},
  journal={IEEE Access},
  year={2026}
}
```