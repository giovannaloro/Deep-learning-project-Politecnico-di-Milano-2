# Mars Terrain Semantic Segmentation

> Segmenting grayscale Mars surface images into 5 classes (Background, Soil, Bedrock, Sand, Big Rock) using a novel dual-path U-Net.

## Overview

This project was developed for the **AN2DL – Second Homework**. The task is pixel‑wise semantic segmentation of Mars terrain images (64×128 grayscale). We designed a dual U-Net architecture combining standard and dilated convolutions with squeeze‑and‑excitation attention, achieving strong performance despite severe class imbalance (Big Rock: only 0.13% of pixels).

## Key Achievements

- **Online segmentation score:** 69.06%
- **Mean IoU (without background):** 60.51% (improved from 41.76% baseline)
- **Architecture:** Dual‑path U-Net with SE attention
- **Key techniques:** Outlier removal, tiling, focal loss, geometric augmentation, ignoring background in loss

## Dataset

- 2,615 grayscale images (after outlier removal)
- 5 classes: Background, Soil, Bedrock, Sand, Big Rock
- 110 outlier images with artificial “gray alien” masks removed

## Methodology

1. **Data preprocessing**  
   - Outlier detection by visual inspection + mask equality check  
   - Tiling large images for memory efficiency and better generalization  
   - Train/validation split preserving original distribution

2. **Class imbalance analysis**  
   - GLCM texture analysis (contrast, correlation, energy, homogeneity) showed high variability for Sand/Background  
   - Oversampling underrepresented classes gave marginal improvement (IoU 43.51% vs 41.76%)

3. **Loss functions**  
   - Compared Sparse CE, Weighted CE, Focal, Dice, Dice+Weighted CE  
   - Focal loss (α=0.25, γ=2.0) gave best validation IoU (44.65%)  
   - **Breakthrough:** Excluding background class from loss significantly boosted performance

4. **Data augmentation**  
   - Geometric transforms (rotation, flip, zoom) + noise injection  
   - Best combination: Geometric + Noise (IoU 44.55%)

5. **Architecture exploration**  
   - U‑Net variants: gated skip connections, attention decoder, residual connections, deep supervision  
   - Deep supervision achieved IoU 42.80% (Score 45.34%)  
   - Final model: **dual‑path U‑Net** with standard + dilated convolutions, two down/up paths, SE attention in bottleneck, spatial dropout

## Final Model Details

| Component        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| Input            | 64×128 grayscale image                                                      |
| Path 1           | Standard convolutions (low‑scale features)                                  |
| Path 2           | Dilated convolutions (high‑scale, larger receptive field)                   |
| Bottleneck       | Squeeze‑and‑Excitation (SE) attention block                                 |
| Fusion           | Concatenate outputs of both paths → softmax                                 |
| Regularization   | Spatial dropout throughout                                                   |
| Loss             | Sparse categorical cross‑entropy (background class excluded)                |

## Results

| Model                        | Mean IoU (%) | Online Score (%) |
|------------------------------|--------------|------------------|
| Baseline U‑Net (no oversampling) | 41.76      | -                |
| + Oversampling               | 43.51        | -                |
| + Focal loss (α=0.25,γ=2.0)  | 44.65        | -                |
| + Deep supervision           | 42.80        | 45.34            |
| **Final dual‑path + SE**     | **60.51**    | **69.06**        |

## Repository Structure


## How to Run

1. Clone the repository  
2. Install dependencies: `pip install -r requirements.txt`  
3. Download Mars dataset and place under `data/`  
4. Run preprocessing: `python preprocess.py --remove_outliers`  
5. Train final model: `python train.py --model dual_unet --ignore_background`

## Dependencies

- Python 3.8+
- TensorFlow / Keras
- OpenCV, scikit‑image (GLCM)
- NumPy, pandas, matplotlib

## Authors

- Mehrshad Alipoor  
- Giovanni Annaloro  
- Maurizio Tirabassi  

## License

Academic use only – see report for details.
