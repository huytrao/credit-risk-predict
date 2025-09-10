<<<<<<< HEAD
# Home Credit Default Risk Project

## Project Introduction

This is a complete machine learning project for the Kaggle Home Credit Default Risk competition. Many people find it difficult to obtain loans due to a lack of or no credit history, and these populations are often exploited by untrustworthy lenders.

Home Credit Group is committed to expanding financial inclusion for unbanked populations by providing a positive and safe lending experience. To ensure these underserved populations have a positive lending experience, Home Credit uses various alternative data (including telecommunications and transaction information) to predict customers' repayment ability.

## Project Goal

Predict each applicant's ability to repay the loan, using AUC (Area Under the ROC Curve) as the evaluation metric.

## Dataset Description

1.  **application_train.csv / application_test.csv**: Main table, containing static data for all applications
2.  **bureau.csv**: Historical credit records of clients from other financial institutions
3.  **bureau_balance.csv**: Monthly balance of historical credits from the credit bureau
4.  **POS_CASH_balance.csv**: Monthly balance snapshots of POS and cash loans
5.  **credit_card_balance.csv**: Monthly balance snapshots of credit cards
6.  **previous_application.csv**: Historical application records of clients at Home Credit
7.  **installments_payments.csv**: Repayment records for historical credits

## Project Structure

```
├── data/                    # Data directory
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   └── features/           # Feature engineered data
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
├── src/                    # Source code
│   ├── data/              # Data processing module
│   ├── features/          # Feature engineering module
│   ├── models/            # Model related module
│   └── utils/             # Utility functions
├── models/                 # Trained models
├── submissions/           # Submission files
├── requirements.txt       # Dependencies
└── README.md
```

## Installation and Usage

1.  Clone the project:
    ```bash
    git clone <repository-url>
    cd home-credit-default-risk
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download the dataset to the `data/raw/` directory

4.  Run notebooks for data analysis and model training

## Tech Stack

-   Python 3.11+
-   pandas, numpy, scikit-learn
-   LightGBM, XGBoost, CatBoost
-   matplotlib, seaborn, plotly
-   jupyter notebook

## Evaluation Metric

Use AUC (Area Under the ROC Curve) as the main evaluation metric.
=======
---
title: Finance Risk Check
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---

# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).
>>>>>>> 01d757a6cd21488668cce877c070f8e5c5717a25
