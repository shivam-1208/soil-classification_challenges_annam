

Soil Image Classifier with DINOv2 and One-Class SVM
This project provides a pipeline for soil image classification using a DINOv2 Vision Transformer backbone for feature extraction and a One-Class SVM for anomaly-based soil detection. The solution is designed for Kaggle-style competitions and outputs a submission-ready CSV.

Features
Modern Vision Backbone: Uses DINOv2 (ViT) via timm for robust feature extraction.

Unsupervised Outlier Detection: Employs One-Class SVM to distinguish soil from non-soil images.

Efficient Pipeline: Handles data preprocessing, feature extraction, model training, prediction, and submission file generation.

Metrics Calculation: Easily extendable to compute accuracy, precision, recall, F1, confusion matrix, and ROC-AUC if ground-truth labels are available.

File Structure
text
.
├── soil_classifier_dinov2.py      # Main pipeline script
├── submission.csv                 # Generated Kaggle submission
├── soil_competition-2025/
│   ├── train_labels.csv           # Training labels
│   ├── train/                     # Training images
│   ├── test.csv                   # Test image IDs
│   ├── test/                      # Test images
│   └── test_labels.csv            # (Optional) Test labels for metrics
Usage
Install Requirements

bash
pip install torch torchvision timm scikit-learn pandas numpy pillow tqdm
Prepare Data

Place train_labels.csv, train/, test.csv, and test/ in the soil_competition-2025/ directory.

Run the Classifier

bash
python soil_classifier_dinov2.py
Submission

The script generates submission.csv in the correct Kaggle format.

Metrics (Optional)

If you have ground-truth test labels (test_labels.csv), add the metrics code block (provided in responses above) to evaluate your predictions.

Pipeline Overview
Load DINOv2 Model:
Loads a pretrained DINOv2 Vision Transformer and removes the classification head to use as a feature extractor.

Image Preprocessing:
Resizes images to 224x224, normalizes, and converts to tensors.

Custom Dataset Loader:
Efficiently loads images and IDs for both training and testing.

Feature Extraction:
Extracts high-dimensional embeddings for all images.

One-Class SVM Training:
Trains on soil images to detect outliers (non-soil).

Prediction:
Predicts soil presence in test images (1 for soil, 0 for not soil).

Submission File Generation:
Outputs a CSV file ready for Kaggle submission.

Example: Metrics Calculation
If you have ground-truth test labels, you can compute all key metrics:

python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
# ... (see previous assistant response for full code)
Notes
The default SVM configuration (nu=0.05) assumes at least 95% of the training data are inliers (soil).

For best results, ensure your training data is clean and representative.

For competitions, only submit the generated submission.csv unless otherwise instructed.

License
This project is released under the MIT License.

Happy Competing! 🚀