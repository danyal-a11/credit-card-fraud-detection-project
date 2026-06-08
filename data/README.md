# Data Directory

## Raw Data

The Kaggle Credit Card Fraud Detection dataset (`creditcard.csv`) is excluded from this repository due to size.

To reproduce the project:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in this directory
3. Run notebook `02_data_preparation.ipynb` to generate the processed files

## Processed Data

After running notebook `02_data_preparation.ipynb`, this directory will contain:

- `processed/x_test_scaled.csv` — Scaled test features (56,746 samples)
- `processed/y_test.csv` — Test labels (95 fraud cases)

## Dataset Details

- 284,807 transactions over 2 days
- 492 fraud cases (0.173% fraud rate)
- 30 features: Time, Amount, V1–V28 (PCA-anonymized)
- Target: Class (0 = legitimate, 1 = fraud)