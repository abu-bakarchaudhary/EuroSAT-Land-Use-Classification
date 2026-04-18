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

#### 3. **Download EuroSAT Dataset:**
- Download from [official source](https://github.com/phelber/EuroSAT)
- Extract and organize as:

#### 4. **Update file paths:**
Edit the dataset paths in the Python files:
```python
# In train_model_manual.py and test_model_manual.py
TRAIN_PATH = r"F:\assignment\DIP\Lecture\EuroSAT Dataset\Train"
TEST_PATH = r"F:\assignment\DIP\Lecture\EuroSAT Dataset\Test"
```

---

## 💻 Usage

### Complete Workflow:

#### **Step 1: Train the Model**
```bash
python train_model_manual.py
```

**What it does:**
- Loads all training images
- Extracts 30+ features per image
- Computes normalization parameters (mean, std)
- Saves model to `Euclidean_distance.pkl`

**Expected Output:**

---

#### **Step 2: Test the Model**
```bash
python test_model_manual.py
```

**What it does:**
- Loads trained model
- Evaluates on test set
- Prints per-class accuracy
- Displays confusion matrix

**Expected Output:**

---

#### **Step 3: Generate Visualizations**
```bash
python visualization.py
```

**What it does:**
- Creates 4 publication-quality plots (300 DPI)
- Analyzes prediction confidence
- Shows class-wise performance

**Generated Files:**
1. `boxplot_distances.png` - Distance distribution per class
2. `scatter_predictions.png` - Prediction scatter plot
3. `accuracy_barplot.png` - Per-class accuracy bars
4. `confidence_distribution.png` - Confidence analysis

---

#### **Step 4 (Optional): View Training Statistics**
```bash
python satellite_logic.py
```

**What it does:**
- Displays average feature values per class
- Helps understand class characteristics

---

## 📊 Results

### Overall Performance:
- **Overall Accuracy**: 69.00%
- **Average Class Accuracy**: 68.20%
- **Total Test Samples**: 500 images
- **Best Performing Class**: Sea/Lake (90%)
- **Most Challenging Class**: Highway (40%)

### Detailed Per-Class Results:

| Class                    | Accuracy | Correct/Total | Performance |
|--------------------------|----------|---------------|-------------|
| 🌾 Annual Crop           | 68.00%   | 34/50         | Good        |
| 🌲 Forest                | 82.00%   | 41/50         | Excellent   |
| 🌿 Herbaceous Vegetation | 56.00%   | 28/50         | Moderate    |
| 🛣️ Highway               | 40.00%   | 20/50         | Challenging |
| 🏭 Industrial            | 78.00%   | 39/50         | Good        |
| 🐄 Pasture               | 88.00%   | 44/50         | Excellent   |
| 🍇 Permanent Crop        | 58.00%   | 29/50         | Moderate    |
| 🏘️ Residential           | 78.00%   | 39/50         | Good        |
| 🌊 River                 | 60.00%   | 30/50         | Moderate    |
| 🌊 Sea/Lake              | 90.00%   | 45/50         | Excellent   |

### Confusion Matrix:

### Key Observations:

#### ✅ **Strengths:**
- **Water Bodies** (Sea/Lake, River): High accuracy due to distinct color/texture
- **Pasture & Forest**: Well-separated in feature space
- **Industrial & Residential**: Good structural edge detection

#### ⚠️ **Challenges:**
- **Highway**: Often confused with Industrial (similar edge patterns) and Residential (urban context)
- **Herbaceous Vegetation**: Confused with Forest (similar green color but different texture)
- **Permanent Crop**: Confused with Pasture (both agricultural, similar green tones)

---

## 🧠 Methodology

### Algorithmic Workflow:

### Mathematical Foundation:

#### **1. Feature Normalization:**

#### **2. Euclidean Distance:**

#### **3. k-NN Classification:**

---

## 📈 Visualization

The project generates **4 comprehensive visualizations** for result analysis:

### 1. **Boxplot - Distance Distribution**
**File:** `boxplot_distances.png`

**Description:**
- Shows distribution of average distances to k-nearest neighbors for each class
- Lower distances → More confident predictions → Tighter clusters in feature space
- Box shows IQR (25th-75th percentile)
- Whiskers extend to 1.5×IQR
- Red diamond = mean distance

**Interpretation:**
- **Low variance + low mean**: Well-defined class (e.g., SeaLake)
- **High variance**: Class overlaps with others (e.g., Highway)

---

### 2. **Scatter Plot - Predictions vs True Labels**
**File:** `scatter_predictions.png`

**Description:**
- X-axis: True class (ground truth)
- Y-axis: Predicted class
- Point size: Confidence level (larger = more confident)
- Green circles: Correct predictions
- Red X's: Incorrect predictions
- Blue diagonal line: Perfect prediction

**Interpretation:**
- Points on diagonal = correct
- Points off diagonal = misclassification
- Clusters off-diagonal reveal systematic confusions

---

### 3. **Bar Plot - Per-Class Accuracy**
**File:** `accuracy_barplot.png`

**Description:**
- Color-coded performance bars
  - 🟢 Green: ≥70% (Good)
  - 🟠 Orange: 50-70% (Moderate)
  - 🔴 Red: <50% (Challenging)
- Exact percentage labeled on each bar
- Reference lines at 50% and 70%

**Interpretation:**
- Quickly identify strong and weak classes
- Overall accuracy shown in text box

---

### 4. **Confidence Distribution**
**File:** `confidence_distribution.png`

**Description:**
- **Left panel**: Histogram of confidence scores
  - Green: Correct predictions
  - Red: Incorrect predictions
- **Right panel**: Cumulative distribution
  - Shows percentage of predictions below each confidence threshold

**Interpretation:**
- Correct predictions should cluster at high confidence
- Incorrect predictions at low confidence suggest uncertainty
- Overlap indicates inherent ambiguity in dataset

---

## 📁 Project Structure

---

## 📚 Code Documentation

### **satellite_features.py**

```python
def extract_features(img_path: str) -> np.ndarray
```
**Purpose:** Extract 30-dimensional feature vector from satellite image

**Parameters:**
- `img_path`: Path to image file (jpg, png, tif)

**Returns:**
- NumPy array of shape (30,) containing features
- Returns `None` if image cannot be loaded

**Feature Breakdown:**
- Features 0-11: HSV color statistics (mean, std, 25th, 75th percentile for H, S, V)
- Features 12-16: Texture (Laplacian variance, Sobel means/stds)
- Features 17-21: Edges (Canny density, 4-bin orientation histogram)
- Features 22-25: Intensity (mean, std, 10th, 90th percentile)
- Feature 26: Local Binary Pattern

**Example:**
```python
features = extract_features("path/to/image.jpg")
# Output: array([45.2, 89.3, 23.1, ..., 4.5]) shape (30,)
```

---

### **train_model_manual.py**

```python
def build_training_data(train_path: str) -> tuple
```
**Purpose:** Load training images and extract features

**Parameters:**
- `train_path`: Path to training folder containing class subfolders

**Returns:**
- `X_train`: Normalized feature matrix (n_samples × 30)
- `y_train`: Class labels (n_samples,)
- `mean`: Feature means for normalization (30,)
- `std`: Feature standard deviations (30,)

---

```python
def Euclidean_distance(train_path: str, E: int = 9) -> tuple
```
**Purpose:** Train the classifier and save model

**Parameters:**
- `train_path`: Path to training data
- `E`: Number of nearest neighbors (default: 9)

**Saves:**
- `Euclidean_distance.pkl`: Pickled model with all training data

---

### **test_model_manual.py**

```python
def load_model() -> tuple
```
**Purpose:** Load trained model from disk

**Returns:**
- `X_train`: Training features
- `y_train`: Training labels
- `mean`: Normalization mean
- `std`: Normalization std
- `E`: k value

---

```python
def Euclidean_distance(X_train, y_train, features, mean, std, E) -> int
```
**Purpose:** Predict class for given features

**Parameters:**
- `X_train`: Training feature matrix
- `y_train`: Training labels
- `features`: Test sample features (30,)
- `mean`, `std`: Normalization parameters
- `E`: Number of neighbors

**Returns:**
- Predicted class index (0-9)

**Algorithm:**
1. Normalize test features
2. Calculate Euclidean distance to all training samples
3. Find E nearest neighbors
4. Return most common class among neighbors

---

```python
def evaluate_model(test_path: str) -> None
```
**Purpose:** Evaluate model on test set and print results

**Prints:**
- Per-class accuracy table
- Overall accuracy
- Confusion matrix

---

### **visualization.py**

```python
def get_predictions_and_confidences(test_path: str) -> tuple
```
**Purpose:** Collect all predictions with metadata

**Returns:**
- `predictions`: Predicted classes (n_test,)
- `true_labels`: Ground truth (n_test,)
- `confidences`: Confidence scores 0-1 (n_test,)
- `class_distances`: Dict of average distances per class

---

```python
def plot_boxplot(class_distances: dict, output_file: str) -> None
```
**Purpose:** Create boxplot of prediction distances

**Parameters:**
- `class_distances`: Dict with lists of distances per class
- `output_file`: Output PNG filename

---

```python
def plot_confusion_scatterplot(predictions, true_labels, confidences, output_file: str) -> None
```
**Purpose:** Create scatter plot of predictions vs truth

---

```python
def plot_class_accuracy_bar(predictions, true_labels, output_file: str) -> None
```
**Purpose:** Create bar chart of per-class accuracy

---

```python
def plot_confidence_distribution(confidences, predictions, true_labels, output_file: str) -> None
```
**Purpose:** Create histogram and CDF of confidence scores

---

```python
def generate_all_visualizations(test_path: str) -> None
```
**Purpose:** Main function to generate all 4 visualizations

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Improvement:
1. **Feature Engineering**
   - Add frequency domain features (FFT)
   - Implement GLCM texture features
   - Experiment with color histograms

2. **Classification**
   - Try different k values
   - Implement distance weighting
   - Add ensemble methods

3. **Visualization**
   - Add ROC curves
   - Implement t-SNE feature space visualization
   - Create interactive plots

4. **Performance**
   - Optimize feature extraction speed
   - Add multiprocessing support
   - Reduce memory footprint

### How to Contribute:

1. **Fork the repository**
2. **Create a feature branch**
```bash
   git checkout -b feature/AmazingFeature
```
3. **Make your changes**
4. **Test thoroughly**
5. **Commit your changes**
```bash
   git commit -m 'Add some AmazingFeature'
```
6. **Push to the branch**
```bash
   git push origin feature/AmazingFeature
```
7. **Open a Pull Request**

### Code Style:
- Follow PEP 8 guidelines
- Add docstrings to new functions
- Include comments for complex logic
- Update README if adding new features

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**AbuBakar Chaudhary**

- 🎓 **Education**: Computer Engineering Student, NUST CEME (Batch DE-45, 2023-2027)
- 💼 **GitHub**: [@abu-bakarchaudhary](https://github.com/abu-bakarchaudhary)
- 💼 **LinkedIn**: [AbuBakar Chaudhary](https://linkedin.com/in/abubakar-chaudhary)
- 🌐 **Portfolio**: [abu-bakarchaudhary.github.io/my-portfolio](https://abu-bakarchaudhary.github.io/my-portfolio)
- 📧 **Email**: abubakar.chaudhary@example.com
- 🏫 **Institution**: National University of Sciences and Technology (NUST), Pakistan

### About Me:
6th semester Computer Engineering student passionate about Computer Vision, Machine Learning, and Software Development. Experienced in Python, OpenCV, Flutter, and digital image processing. Currently building expertise in AI/ML while freelancing on Fiverr and Upwork.

### Other Projects:
- 🤖 **Nustified Synapse** - Anonymous professor review platform (Flutter + Firebase)
- 📊 **AI Behavioral Trading Intelligence** - FYP on trader behavior analysis
- 🔬 **Retinal Optic Nerve Head Segmentation** - Medical image processing
- 🎮 **Unity Car Racing Game** - Game development project

---

## 🙏 Acknowledgments

### Dataset:
- **Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019)**  
  *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*  
  IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(6), 2217-2226.  
  [Paper Link](https://ieeexplore.ieee.org/document/8736785)

### Libraries:
- **OpenCV**: [Open Source Computer Vision Library](https://opencv.org/)
  - Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
  
- **NumPy**: [Fundamental package for scientific computing](https://numpy.org/)
  - Harris, C.R., et al. (2020). *Array programming with NumPy*. Nature, 585, 357-362.

- **Matplotlib**: [Comprehensive library for creating static, animated, and interactive visualizations](https://matplotlib.org/)
  - Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment*. Computing in Science & Engineering, 9(3), 90-95.

### Institution:
- **National University of Sciences and Technology (NUST)**
- **College of Electrical and Mechanical Engineering (CEME)**
- Course: Digital Image Processing (6th Semester)
- Instructor: [Instructor Name]

### Special Thanks:
- NUST CEME faculty for academic guidance
- EuroSAT dataset creators for open-source satellite imagery
- OpenCV community for comprehensive documentation
- Stack Overflow community for troubleshooting support

---

## 📚 References & Further Reading

### Academic Papers:
1. **EuroSAT Dataset**  
   Helber, P., et al. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE JSTARS*, 12(6), 2217-2226.

2. **k-Nearest Neighbors**  
   Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.

3. **Feature Extraction**  
   Haralick, R. M., et al. (1973). Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics*, 3(6), 610-621.

### Tutorials:
- [OpenCV Python Tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

### Related Projects:
- [Satellite Image Classification with CNNs](https://github.com/phelber/EuroSAT)
- [OpenCV Feature Extraction Examples](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)

---

## 📸 Sample Outputs

### Training Process:
```bash
$ python train_model_manual.py

Building training dataset...
  AnnualCrop: 250 samples
  Forest: 250 samples
  HerbaceousVegetation: 250 samples
  Highway: 250 samples
  Industrial: 250 samples
  Pasture: 250 samples
  PermanentCrop: 250 samples
  Residential: 250 samples
  River: 250 samples
  SeaLake: 250 samples

Preparing classifier (E=9)...
Total training samples: 2500
Feature dimensions: 30
Model saved successfully!
```

### Testing Process:
```bash
$ python test_model_manual.py

Evaluating manual Euclidean_distance(E=9) on test set...

======================================================================
PER-CLASS ACCURACY
======================================================================
Class Name                | Accuracy (%)    | Correct/Total
----------------------------------------------------------------------
AnnualCrop                |          68.00% |   34/50  
Forest                    |          82.00% |   41/50  
HerbaceousVegetation      |          56.00% |   28/50  
Highway                   |          40.00% |   20/50  
Industrial                |          78.00% |   39/50  
Pasture                   |          88.00% |   44/50  
PermanentCrop             |          58.00% |   29/50  
Residential               |          78.00% |   39/50  
River                     |          60.00% |   30/50  
SeaLake                   |          90.00% |   45/50  
----------------------------------------------------------------------
OVERALL ACCURACY          |          69.00% |  345/500 
AVERAGE CLASS ACCURACY    |          68.20% |
```

### Visualization Process:
```bash
$ python visualization.py

======================================================================
GENERATING VISUALIZATIONS
======================================================================
Collecting predictions for visualization...

Total predictions: 500
Correct: 345
Accuracy: 69.00%

Generating plots...

✓ Boxplot saved: boxplot_distances.png
✓ Scatter plot saved: scatter_predictions.png
✓ Accuracy bar plot saved: accuracy_barplot.png
✓ Confidence distribution saved: confidence_distribution.png

======================================================================
ALL VISUALIZATIONS GENERATED SUCCESSFULLY!
======================================================================

Generated files:
  1. boxplot_distances.png       - Distance distribution per class
  2. scatter_predictions.png     - Prediction scatter plot
  3. accuracy_barplot.png        - Per-class accuracy bars
  4. confidence_distribution.png - Confidence score analysis
======================================================================
```

---

## 🔧 Troubleshooting

### Common Issues:

#### 1. **ModuleNotFoundError: No module named 'cv2'**
```bash
pip install opencv-python
```

#### 2. **FileNotFoundError: Dataset path not found**
- Check that dataset is downloaded and extracted
- Update paths in Python files to match your directory structure
- Use raw strings: `r"C:\path\to\dataset"`

#### 3. **Low accuracy (<50%)**
- Ensure dataset is organized correctly (Train/Test folders with class subfolders)
- Check that images are loading properly (verify with satellite_logic.py)
- Try different k values (5, 7, 11, 13)

#### 4. **Visualization errors**
```bash
pip install matplotlib
```

#### 5. **Memory issues with large datasets**
- Process in batches
- Reduce image resolution before feature extraction
- Use generators instead of loading all at once

---

## 🎓 Educational Value

This project is ideal for:

### Students:
- Understanding classical machine learning without black-box frameworks
- Learning computer vision feature extraction techniques
- Practicing data science workflow (train/test/visualize)
- Implementing algorithms from scratch

### Educators:
- Teaching traditional ML before deep learning
- Demonstrating the importance of feature engineering
- Showing limitations of handcrafted features
- Comparing with CNN-based approaches

### Researchers:
- Baseline for comparing new satellite classification methods
- Feature extraction module for other remote sensing tasks
- Understanding which features matter for land use classification

---

## 📈 Future Enhancements

### Short-term:
- [ ] Add cross-validation for hyperparameter tuning
- [ ] Implement feature selection (PCA, variance thresholding)
- [ ] Add command-line arguments for easy experimentation
- [ ] Create Docker container for reproducibility

### Medium-term:
- [ ] Web interface for single-image prediction
- [ ] Add more texture features (GLCM, Gabor filters)
- [ ] Implement ensemble methods (voting, bagging)
- [ ] Compare with SVM, Random Forest

### Long-term:
- [ ] Integrate with deep learning (CNN) for comparison
- [ ] Real-time classification from satellite API
- [ ] Mobile app deployment (Flutter + TensorFlow Lite)
- [ ] Multi-label classification (multiple land types per image)

---

## 💡 Tips for Best Results

1. **Data Quality**
   - Ensure images are properly labeled
   - Remove corrupted or ambiguous images
   - Balance class distribution if possible

2. **Feature Engineering**
   - Experiment with different color spaces (LAB, YCrCb)
   - Try multi-scale texture analysis
   - Consider domain-specific features

3. **Hyperparameter Tuning**
   - Test k values: 3, 5, 7, 9, 11, 13
   - Try different distance metrics (Manhattan, Minkowski)
   - Adjust Canny edge thresholds

4. **Validation**
   - Use stratified k-fold cross-validation
   - Monitor per-class performance, not just overall
   - Check for overfitting with train vs test accuracy

---

## 🌟 Star History

If you found this project helpful, please give it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/eurosat-classification&type=Date)](https://star-history.com/#yourusername/eurosat-classification&Date)

---

## 📞 Contact & Support

### Need Help?
- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/eurosat-classification/issues)
- 💬 **Questions**: [Start a discussion](https://github.com/yourusername/eurosat-classification/discussions)
- 📧 **Email**: abubakar.chaudhary@example.com

### Connect:
- **LinkedIn**: Share your results and tag me!
- **Twitter**: Tweet your visualizations with #EuroSATClassification
- **GitHub**: Star and fork to support the project

---

<p align="center">
  <img src="https://img.shields.io/github/stars/yourusername/eurosat-classification?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/yourusername/eurosat-classification?style=social" alt="GitHub forks">
  <img src="https://img.shields.io/github/watchers/yourusername/eurosat-classification?style=social" alt="GitHub watchers">
</p>

<p align="center">
  <b>⭐ If you found this project helpful, please give it a star! ⭐</b>
</p>

<p align="center">
  Made with ❤️ and ☕ by <a href="https://github.com/abu-bakarchaudhary">AbuBakar Chaudhary</a>
</p>

<p align="center">
  <i>"Traditional ML still has a place in modern computer vision"</i>
</p>

---

**Last Updated**: April 2026  
**Version**: 1.0.0  
**Status**: Active Development
