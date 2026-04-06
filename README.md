# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Logistic Regression. The dataset is highly imbalanced, so under-sampling is applied to create a balanced training set before model training.

## Overview

- **Dataset:** Credit card transactions with PCA-transformed features and a binary `Class` label (`0` = legitimate, `1` = fraud)
- **Problem:** Highly imbalanced dataset (~492 fraud vs ~284K legitimate transactions)
- **Approach:** Under-sample the majority class to match the minority class size, then train a Logistic Regression classifier

## Workflow

1. **Data Loading & Exploration** – Load CSV, inspect shape, check for missing values, and analyze class distribution
2. **Under-Sampling** – Sample 492 legitimate transactions to match the 492 fraud cases and concatenate into a balanced dataset
3. **Feature/Target Split** – Separate features (`X`) from the target label (`Y` = `Class`)
4. **Train/Test Split** – 80/20 stratified split with `random_state=2`
5. **Preprocessing** – Standard scaling with `StandardScaler`
6. **Model Training** – Logistic Regression (`solver='liblinear'`, `max_iter=1000`)
7. **Evaluation** – Accuracy score on both training and test sets

## Requirements
```bash
pip install numpy pandas scikit-learn
```

## Usage

1. Place your dataset CSV (e.g., `creditcard.csv`) in the working directory
2. Update the file path in the notebook:
```python
   credit_card_data = pd.read_csv('creditcard.csv')
```
3. Run all cells in `credit_card_fraud_detection.ipynb`

## Results

The model reports accuracy on both training and test data. Since the dataset is balanced via under-sampling before training, accuracy is a reasonable metric here — though precision, recall, and F1-score would be worth adding for production use.

## Notes

- Under-sampling discards a large portion of legitimate transaction data. For better generalization, consider SMOTE (over-sampling) or ensemble methods like Random Forest / XGBoost.
- The original dataset used here is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
