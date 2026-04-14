# PathRadX: From Pathology to Radiology

[![Paper](https://img.shields.io/badge/Paper-Springer-blue.svg)](https://link.springer.com/book/10.1007/978-3-032-07845-2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This is the official repository for the paper **"From Pathology to Radiology: Evaluating the Applicability of Pathology Foundation Models"**. 

## 📌 Introduction

Foundation models demonstrate strong adaptability across diverse tasks, but their applicability across different medical imaging modalities has not been well studied.
This repository introduces **PathRadX**, a cross-modal framework designed to evaluate the applicability of pathology foundation models to various classification tasks in radiology images. 

Our framework integrates modality adaptation and task-specific classification strategies for efficient cross-modal adaptation, demonstrating that pathology pre-trained models possess strong generalization capabilities for radiology imaging tasks.

**Authors:** [Hyun Yang](https://github.com/YH-Paradise), Sumin Jung, and [Jin Tae Kwak](https://kwaklab.net/)

---

## 🧠 Methodology

To fairly assess the generalization ability of the pathology foundation model (we use **UNI**), all pre-trained weights are frozen.
PathRadX addresses the modality gap (e.g., RGB pathology images vs. grayscale radiology images) and task differences through two main components:

### 1. Modality Adaptation
* **Channel Manipulation (CM):** Applies a 3x3 convolution layer to convert single-channel grayscale images into a three-channel format matching the foundation model's expected input.
* **Pseudo Coloring (PC):** Utilizes the RdPu colormap to map grayscale intensity values into a color space that mimics the patterns of H&E stained pathology images.

### 2. Task-Specific Classification
* **Single Instance Learning (SIL):** Resizes radiology images to 224x224 pixels and feeds them directly into the image encoder to extract features.
* **Multiple Instance Learning (MIL):** Crops images into multiple overlapping 224x224 patches, processes each independently, and aggregates the features using attention-based MIL (AB-MIL).

<p align="center">
  <img src="images/pathradx_architecture.png" alt="PathRadX Methodology Pipeline" width="60%">
  <br>
  <em><b>Fig. 1: Overview of the proposed PathRadX framework.</b></em>
</p>

---

## 📂 Datasets

We evaluated the PathRadX framework on three diverse radiology datasets:
* **[MURA](https://stanfordmlgroup.github.io/competitions/mura/):** Musculoskeletal radiographs (40,005 images).
* **[RSNA (RSNA Pneumonia)](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge):** Chest X-ray images for pneumonia detection (26,684 images).
* **[MedFMC](https://www.nature.com/articles/s41597-023-02460-0):** Chest X-ray images for thoracic abnormality classification (4,848 images).

---

## 🚀 Getting Started

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/YH-Paradise/PathRadX.git
cd PathRadX

# Create a virtual environment and install dependencies
conda create -n pathradx python=3.10 -y
conda activate pathradx
pip install -r requirements.txt
```

### Training & Evaluation
*(Add brief instructions on how to run your code, e.g., how to run CM + MIL or PC + SIL)*
```bash
# Example command for running PathRadX with Channel Manipulation and MIL
python main.py --dataset mura --adaptation cm --classification mil
```

---

## 📊 Results

We compared PathRadX against three radiology-specific foundation models: SAM-Med2D, MedCLIP, and BiomedCLIP.

### Accuracy
| Models     |   MURA    | RSNA<sub>P</sub> | MedFMC<sub>C</sub> |
|:-----------|:---------:|:----------------:|:------------------:|
| UNI        | **0.773** |    **0.837**     |       0.730        |
| SAM-Med2D  |   0.709   |      0.797       |       0.742        |
| MedCLIP    |   0.750   |      0.819       |     **0.763**      |
| BiomedCLIP |   0.753   |      0.815       |       0.753        |
*Table 1. Performance evaluation based on Accuracy.*

### F1
| Models     |   MURA    | RSNA<sub>P</sub> | MedFMC<sub>C</sub> |
|:-----------|:---------:|:----------------:|:------------------:|
| UNI        | **0.777** |    **0.576**     |       0.812        |
| SAM-Med2D  |   0.726   |      0.333       |     **0.852**      |
| MedCLIP    |   0.696   |      0.488       |       0.848        |
| BiomedCLIP |   0.700   |      0.417       |       0.851        |
*Table 2. Performance evaluation based on F1*

### AUC
| Models     |   MURA    | RSNA<sub>P</sub> | MedFMC<sub>C</sub> |
|:-----------|:---------:|:----------------:|:------------------:|
| UNI        | **0.861** |    **0.863**     |       0.768        |
| SAM-Med2D  |   0.788   |      0.809       |       0.757        |
| MedCLIP    |   0.844   |      0.834       |     **0.784**      |
| BiomedCLIP |   0.835   |      0.820       |       0.762        |
*Table 3. Performance evaluation based on AUC*

* Leveraging the UNI model with AB-MIL and channel-manipulation achieved the highest accuracy, F1 score, and AUC scores for the MURA and RSNAP datasets.
* Overall, our proposed method demonstrated more balanced and reliable classification performance, particularly on highly imbalanced datasets.

<p align="center">
  <img src="images/confusion_matrix_revised_3.png" alt="PathRadX Methodology Pipeline" width="70%">
  <br>
  <em><b>Fig. 1: Overview of the proposed PathRadX framework.</b></em>
</p>

---

## 🔬 Ablation Study
| SIL | MIL | CM | PC |  MURA F1  | RSNA<sub>P</sub> F1 | MedFMC<sub>C</sub> F1 |
|:---:|:---:|:--:|:--:|:---------:|:-------------------:|:---------------------:|
|  ✓  |     | ✓  |    |   0.752   |        0.357        |         0.870         |
|  ✓  |     |    | ✓  |   0.744   |        0.348        |       **0.873**       |
|     |  ✓  | ✓  |    | **0.777** |      **0.576**      |         0.812         |
|     |  ✓  |    | ✓  |   0.753   |        0.540        |         0.852         |
*Table 4. A comparison of UNI’s performance on F1 score, using different strategies, single instance learning (SIL), multiple
instance learning (MIL), channel manipulation (CM), and pseudo-coloring (PC).*

## 📝 Citation

If you find this code or our framework useful in your research, please consider citing:

```bibtex
@inproceedings{yang2025pathradx,
  title={From Pathology to Radiology: Evaluating the Applicability of Pathology Foundation Models},
  author={Yang, Hyun and Jung, Sumin and Kwak, Jin Tae},
  booktitle={MICCAI},
  year={2025}
}
```

---

## 🤝 Acknowledgements

* This work was supported by a grant of the National Research Foundation of Korea (NRF) (No. RS-2025-00558322) and the Ministry of Health and Welfare of Korea (No. RS-2023-00266130).
* Affiliations: School of Electrical and Electronic Engineering, Korea University & DeepClue Inc..