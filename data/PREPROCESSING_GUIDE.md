# Data Preprocessing Module - Testing Guide

## Overview

The `data_preprocessing.py` module has been fully implemented with comprehensive functionality for preparing network intrusion detection datasets.

## Features Implemented

### ✅ 1. Dataset Loading (`load_dataset`)

- Automatically discovers and loads all CSV files from `data/raw/`
- Supports both CICIDS 2017 and NSL-KDD formats
- Concatenates multiple files into a single DataFrame
- Reports detailed statistics: rows, columns, memory usage, missing values

### ✅ 2. Missing Value Handling (`handle_missing_values`)

- Drops columns with >30% missing data
- Replaces numeric missing values with median
- Replaces categorical missing values with mode
- Handles infinite values by replacing with finite max/min
- Comprehensive logging of all operations

### ✅ 3. Encoding and Label Separation (`encode_and_label`)

- Cleans column names (removes spaces, special characters)
- Auto-detects label column (supports: Label, Class, Attack, etc.)
- Standardizes labels to binary: BENIGN/normal → 0, attacks → 1
- Encodes categorical features using LabelEncoder
- Separates feature matrix (X) and label vector (y)
- Displays label distribution before and after encoding

### ✅ 4. Feature Normalization (`normalize_features`)

- Applies StandardScaler for zero mean and unit variance
- Shows before/after statistics for verification
- Returns both normalized data and fitted scaler

### ✅ 5. Class Balancing (`apply_smote`)

- Uses SMOTE to balance minority attack classes
- Random state = 42 for reproducibility
- Displays class distribution before and after balancing
- Handles errors gracefully (proceeds without SMOTE if it fails)

### ✅ 6. Data Persistence (`save_processed_data`)

- Saves preprocessed data to `data/processed/cleaned_data.csv`
- Combines features and labels
- Reports file size and statistics

### ✅ 7. Complete Pipeline (`main`)

- Executes all steps in sequence
- Comprehensive error handling
- Detailed logging at each step
- Final summary with dataset shape and next steps

### ✅ 8. Object-Oriented Interface (`DataPreprocessor` class)

- Convenient wrapper for all functionality
- Maintains state throughout pipeline
- Easy-to-use interface for integration

## How to Use

### Method 1: Function-based (Recommended)

```python
# Simply run the script
python data_preprocessing.py
```

### Method 2: Import as module

```python
from data_preprocessing import DataPreprocessor

# Create preprocessor instance
preprocessor = DataPreprocessor(data_dir='../data/raw')

# Run complete pipeline
X, y, scaler = preprocessor.preprocess_pipeline(apply_balancing=True)

# Save results
preprocessor.save_data()
```

### Method 3: Use individual functions

```python
from data_preprocessing import (
    load_dataset,
    handle_missing_values,
    encode_and_label,
    normalize_features,
    apply_smote,
    save_processed_data
)

# Load data
df = load_dataset()

# Clean data
df_clean = handle_missing_values(df)

# Encode
X, y, label_col = encode_and_label(df_clean)

# Normalize
X_norm, scaler = normalize_features(X)

# Balance
X_bal, y_bal = apply_smote(X_norm, y)

# Save
save_processed_data(X_bal, y_bal)
```

## Expected Output

When you run the script, you'll see detailed logs like:

```
============================================================
  NIDS-ML DATA PREPROCESSING PIPELINE
============================================================

============================================================
STEP 1: LOADING DATASET
============================================================
Found 8 CSV file(s) in data/raw
Loading: Monday-WorkingHours.pcap_ISCX.csv
  ✓ Loaded 529,918 rows, 79 columns
...
------------------------------------------------------------
DATASET SUMMARY
------------------------------------------------------------
Total Rows: 2,830,743
Total Columns: 79
Memory Usage: 1,654.32 MB
Duplicate Rows: 23,456
Columns with Missing Values: 5
------------------------------------------------------------

============================================================
STEP 2: HANDLING MISSING VALUES
============================================================
No columns exceed 30.0% missing data threshold
Replaced 1,234 missing values in 'Flow_Bytes/s' with median: 12345.67
...

============================================================
STEP 3: ENCODING AND LABEL SEPARATION
============================================================
Identified label column: 'Label'

Original Label Distribution:
  BENIGN: 2,273,097 (80.29%)
  DDoS: 128,027 (4.52%)
  PortScan: 158,930 (5.62%)
  ...

Binary Label Distribution:
  BENIGN (0): 2,273,097 (80.29%)
  ATTACK (1): 557,646 (19.71%)

Feature Matrix (X): (2,830,743, 78)
Label Vector (y): (2,830,743,)

============================================================
STEP 4: FEATURE NORMALIZATION
============================================================
Sample statistics BEFORE normalization:
  Flow_Duration: mean=12345.6789, std=98765.4321
  ...
Sample statistics AFTER normalization:
  Flow_Duration: mean=0.0000, std=1.0000
  ...

============================================================
STEP 5: CLASS BALANCING WITH SMOTE
============================================================
Class distribution BEFORE SMOTE:
  BENIGN (0): 2,273,097 (80.29%)
  ATTACK (1): 557,646 (19.71%)

Applying SMOTE...

Class distribution AFTER SMOTE:
  BENIGN (0): 2,273,097 (50.00%)
  ATTACK (1): 2,273,097 (50.00%)

Total samples: 2,830,743 → 4,546,194 (+1,715,451)

============================================================
STEP 6: SAVING PROCESSED DATA
============================================================
✓ Saved processed data to: data/processed/cleaned_data.csv
  Rows: 4,546,194
  Columns: 79 (78 features + 1 label)
  File size: 2,345.67 MB

============================================================
PREPROCESSING COMPLETED SUCCESSFULLY! ✓
============================================================
Final Dataset Shape: 4,546,194 rows × 78 features
Output File: data/processed/cleaned_data.csv
Next Step: Feature Selection (Part 3)
============================================================
```

## Requirements

Make sure you have placed your dataset files in `data/raw/`:

- For CICIDS 2017: Place all `.csv` files
- For NSL-KDD: Place `KDDTrain+.txt` or converted CSV files

## Next Steps

After preprocessing is complete:

1. ✅ Preprocessed data saved to `data/processed/cleaned_data.csv`
2. ✅ Logs saved to `logs/preprocessing.log`
3. ➡️ Move to **Part 3: Feature Selection Module**

## Logging

All operations are logged to:

- Console (real-time)
- `logs/preprocessing.log` (persistent)

Check the log file for detailed information about each step.

## Error Handling

The module includes comprehensive error handling:

- File not found errors
- Missing column errors
- SMOTE failures (graceful degradation)
- Memory issues
- Invalid data types

## Key Improvements from Original

1. ✅ Modular function-based design
2. ✅ Comprehensive logging at every step
3. ✅ Handles both CICIDS 2017 and NSL-KDD
4. ✅ Auto-detects label column
5. ✅ Proper infinite value handling
6. ✅ Duplicate removal
7. ✅ Column name cleaning
8. ✅ Binary label standardization
9. ✅ Both functional and OOP interfaces
10. ✅ Detailed statistics and validation

## Testing

To test with a small dataset:

```python
# Create a test CSV in data/raw/
import pandas as pd
import numpy as np

# Generate sample data
test_data = pd.DataFrame({
    'Feature1': np.random.randn(1000),
    'Feature2': np.random.randn(1000),
    'Feature3': np.random.choice(['A', 'B', 'C'], 1000),
    'Label': np.random.choice(['BENIGN', 'DDoS', 'PortScan'], 1000, p=[0.8, 0.1, 0.1])
})

test_data.to_csv('data/raw/test_data.csv', index=False)

# Run preprocessing
python data_preprocessing.py
```

## Author

Priyanshu Kumar
Date: November 10, 2025
Project: NIDS-ML Final Year CSE Project
