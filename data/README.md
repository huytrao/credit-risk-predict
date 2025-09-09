# Data Directory Description

This directory is used to store the data files for the Home Credit Default Risk competition.

## Directory Structure

data/
├── raw/ # Original data files
│ ├── application_train.csv
│ ├── application_test.csv
│ ├── bureau.csv
│ ├── bureau_balance.csv
│ ├── previous_application.csv
│ ├── POS_CASH_balance.csv
│ ├── credit_card_balance.csv
│ └── installments_payments.csv
├── processed/ # Processed data files
│ ├── train_features.csv
│ ├── test_features.csv
│ ├── feature_info.csv
│ └── feature_importance.csv
└── features/ # Intermediate files for feature engineering

markdown
Copy code

## Data Download

### Method 1: Direct Download from Kaggle

1. Visit the Kaggle competition page: https://www.kaggle.com/c/home-credit-default-risk  
2. Log in to your Kaggle account  
3. Go to the "Data" tab  
4. Download all data files into the `data/raw/` directory  

### Method 2: Using the Kaggle API

1. Install the Kaggle API:
   
pip install kaggle
Configure API key:

Download kaggle.json from your Kaggle account settings

Place the file in ~/.kaggle/kaggle.json

Set permissions: chmod 600 ~/.kaggle/kaggle.json

Download the data:

bash
Copy code
cd data/raw
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip
rm home-credit-default-risk.zip
Data Files Description
Main Files
application_train.csv / application_test.csv

Main application table, containing static data on all loan applications

Training set includes TARGET variable (1 = default, 0 = non-default)

bureau.csv

Credit history of clients at other financial institutions

Data from credit bureau

bureau_balance.csv

Monthly balance records of bureau credit history

previous_application.csv

Historical application records at Home Credit

POS_CASH_balance.csv

Monthly balance snapshots of POS and cash loans

credit_card_balance.csv

Monthly balance snapshots of credit cards

installments_payments.csv

Repayment records of previous credits

Data Size Reference
File Name	Size (approx.)	Rows (approx.)
application_train.csv	50MB	300K
application_test.csv	35MB	200K
bureau.csv	100MB	1.7M
bureau_balance.csv	300MB	27M
previous_application.csv	150MB	1.7M
POS_CASH_balance.csv	100MB	10M
credit_card_balance.csv	200MB	3.8M
installments_payments.csv	100MB	13M

Notes
Storage Space: Ensure at least 2GB free space for raw data

Memory Requirement: At least 8GB RAM recommended for processing

Processing Time: Full feature engineering may take 10–30 minutes

Data Quality: Raw data contains missing and abnormal values, handled automatically by project code

Quick Start
After downloading the data, you can start the project with the following commands:

bash
Copy code
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline quickly
python quick_start.py

# Or run step by step using Jupyter notebooks
jupyter notebook notebooks/01_data_exploration.ipynb
Troubleshooting
Data Download Issues
Ensure Kaggle competition rules have been accepted

Check network connection

Verify Kaggle API configuration

Memory Issues
Use memory optimization function reduce_memory_usage()

Process large files in batches

Close other applications to free memory

File Path Issues
Ensure data files are in the correct directory

Check file names match those in the code

Use relative paths instead of absolute paths