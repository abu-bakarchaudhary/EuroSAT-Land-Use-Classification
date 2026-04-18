# 🛰️ EuroSAT Land Use Classification

A machine learning project for classifying satellite imagery into 10 land use categories using **only OpenCV and NumPy** - no deep learning frameworks required!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Methodology](#methodology)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Code Documentation](#code-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project implements a **custom Euclidean distance-based classifier** for satellite image classification using handcrafted computer vision features. Achieves **~69% accuracy** on the EuroSAT dataset without using any machine learning libraries.

### Key Highlights:
- ✅ **No ML Libraries**: Pure OpenCV + NumPy implementation
- ✅ **30+ Features**: Color, texture, edge, and intensity features
- ✅ **10 Land Classes**: Annual crops, forests, highways, residential areas, etc.
- ✅ **Complete Pipeline**: Training, testing, and visualization
- ✅ **Comprehensive Visualizations**: Boxplots, scatter plots, confidence analysis

### Live Demo:
🌐 **[View Project Documentation](https://yourusername.github.io/eurosat-classification/)**

---

## 🌍 Dataset

**EuroSAT Dataset**: RGB satellite images from Sentinel-2 satellite
- **Image Size**: 64×64 pixels
- **Classes**: 10 land use categories
- **Train/Test Split**: Custom split
- **Total Images**: 2,500 training + 500 test

### Land Use Classes:
1. 🌾 **Annual Crop** - Seasonal agricultural land
2. 🌲 **Forest** - Dense tree coverage
3. 🌿 **Herbaceous Vegetation** - Grasslands and meadows
4. 🛣️ **Highway** - Road infrastructure
5. 🏭 **Industrial** - Manufacturing and industrial zones
6. 🐄 **Pasture** - Grazing land for livestock
7. 🍇 **Permanent Crop** - Orchards and vineyards
8. 🏘️ **Residential** - Housing and urban areas
9. 🌊 **River** - Inland water bodies
10. 🌊 **Sea/Lake** - Large water bodies

**Dataset Source**: [EuroSAT: Land Use and Land Cover Classification](https://github.com/phelber/EuroSAT)

---

## ✨ Features

### Extracted Features (30+ dimensions):

#### 1. **Color Features (HSV)** - 12 features
- Mean, standard deviation, 25th percentile, 75th percentile for each of H, S, V channels
- Captures color distribution and variation

#### 2. **Texture Features** - 5 features
- Laplacian variance (roughness measure)
- Sobel gradient means (horizontal & vertical)
- Sobel gradient standard deviations (horizontal & vertical)

#### 3. **Edge Features** - 5 features
- Canny edge density
- Edge orientation histogram (4 bins)
- Captures structural patterns and directionality

#### 4. **Intensity Statistics** - 4 features
- Mean, standard deviation
- 10th percentile, 90th percentile
- Brightness distribution analysis

#### 5. **Local Binary Pattern** - 1 feature
- Simplified LBP for micro-texture analysis
- 8-neighbor comparison

#### 6. **Classification Method**
- Custom k-Nearest Neighbors (k=9)
- Euclidean distance metric
- Z-score normalization
- Majority voting for prediction

---

## 🚀 Installation

### Prerequisites:
```bash
