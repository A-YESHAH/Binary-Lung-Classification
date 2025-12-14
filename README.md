🫁 PneumoScan AI

Explainable Deep Learning System for Pneumonia Detection from Chest X-Ray Images

📌 Overview

PneumoScan AI is a deep learning–based system designed to automatically detect pneumonia from chest X-ray images while providing visual explanations for model predictions. The system uses transfer learning with DenseNet121 and integrates Grad-CAM to highlight lung regions that influence the diagnosis, improving transparency and trust in medical AI systems.

This project was developed as part of the Artificial Intelligence (CSC325) course at Bahria University.

🎯 Objectives

Detect pneumonia from chest X-ray images using deep learning

Provide explainable predictions using Grad-CAM heatmaps

Handle class imbalance in medical datasets

Deploy the trained model through a web-based interface

🧠 Model & Methodology

Model Architecture: DenseNet121 (pretrained on ImageNet)

Learning Strategy: Transfer learning with two-phase training

Phase 1: Freeze base model layers

Phase 2: Fine-tune the last 50 layers

Loss Function: Binary Cross-Entropy (with class weights)

Optimizer: Adam

Explainability: Grad-CAM for visual interpretation

📂 Dataset

Source: Kaggle – Chest X-ray Pneumonia Dataset

Total Images: 5,863

Classes:

Normal

Pneumonia

Image Size: 256 × 256

Type: Pediatric chest X-ray images

Dataset link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

🔧 Preprocessing & Augmentation

Image resizing and normalization

Data augmentation techniques:

Rotation

Zoom

Width/height shift

Horizontal flip

📊 Results

Test Accuracy: 92%

Precision & Recall: High performance for both classes

AUC (ROC): 0.97

Grad-CAM heatmaps successfully highlight lung regions relevant to pneumonia detection, improving interpretability.

🌐 Web Deployment

The trained model is deployed using a web-based interface that allows users to:

Upload chest X-ray images

View pneumonia probability

Visualize Grad-CAM heatmaps

(Frontend built using HTML/CSS/JavaScript and backend using Flask/Django)

🛠️ Technologies Used

Programming Language: Python

Frameworks: TensorFlow, Keras

Libraries: NumPy, OpenCV, Matplotlib

Model: DenseNet121

Explainability: Grad-CAM

Web Framework: Flask / Django

👥 Team

Mahvil

Ayesha Niazi

⚠️ Limitations

Dataset is limited to pediatric patients

Class imbalance between Normal and Pneumonia images

DenseNet121 requires high computational resources

🚀 Future Improvements

Train on multi-age and multi-center datasets

Use lightweight models (EfficientNet, MobileNet)

Add uncertainty estimation for ambiguous cases

Integrate clinical metadata for improved diagnosis

📜 License

This project is for academic and educational purposes only.
