# MNIST Classification: ANN vs Logistic Regression vs SVM

This study compares the performance of a fully connected Artificial Neural Network (ANN) with traditional machine learning models (Logistic Regression and SVM) on the MNIST handwritten digit dataset.

## Features
- Fully connected ANN with ReLU + Softmax
- Cross-entropy loss
- Accuracy/Loss visualization
- Confusion matrix comparison
- Baselines: Logistic Regression, SVM

## Results
- Logistic Regression: ~92%
- SVM: ~95%
- ANN: ~97%

## How to Run
```bash
pip install -r requirements.txt
python mnist_comparison.py