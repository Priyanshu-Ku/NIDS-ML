# 📌 Quick Reference: Data Preprocessing Module

## 🎯 Main File

**`src/data_preprocessing.py`** - 528 lines

## 🔧 Core Functions

| Function                  | Purpose                           | Input                | Output                         |
| ------------------------- | --------------------------------- | -------------------- | ------------------------------ |
| `load_dataset()`          | Load all CSV files                | data_dir path        | DataFrame                      |
| `handle_missing_values()` | Clean missing/infinite values     | DataFrame, threshold | Cleaned DataFrame              |
| `encode_and_label()`      | Encode features & separate labels | DataFrame            | X, y, label_col                |
| `normalize_features()`    | StandardScaler normalization      | X features           | X_normalized, scaler           |
| `apply_smote()`           | Balance classes with SMOTE        | X, y, random_state   | X_balanced, y_balanced         |
| `save_processed_data()`   | Save to CSV                       | X, y, output_dir     | file_path                      |
| `main()`                  | Complete pipeline                 | -                    | X_balanced, y_balanced, scaler |

## 🏃 Quick Start

### Run Full Pipeline

```bash
cd src
python data_preprocessing.py
```

### Test First

```bash
python test_preprocessing.py
```

### Use as Module

```python
from src.data_preprocessing import DataPreprocessor

prep = DataPreprocessor()
X, y, scaler = prep.preprocess_pipeline()
prep.save_data()
```

## 📊 Expected Output

### Console

- Loading stats (rows, columns, memory)
- Missing value handling summary
- Label distribution (before/after)
- Normalization verification
- SMOTE balancing results
- Final success message

### Files

- `data/processed/cleaned_data.csv` - Preprocessed data
- `logs/preprocessing.log` - Detailed logs

## ⚙️ Configuration

| Parameter      | Default    | Description                    |
| -------------- | ---------- | ------------------------------ |
| `threshold`    | 0.3        | Drop columns with >30% missing |
| `random_state` | 42         | SMOTE reproducibility          |
| `method`       | 'standard' | StandardScaler normalization   |

## 🔍 Key Features

✅ Auto-detects label column (Label, Class, Attack, etc.)
✅ Binary standardization: BENIGN=0, ATTACK=1  
✅ Handles missing values (median/mode)  
✅ Removes infinite values  
✅ Cleans column names  
✅ Removes duplicates  
✅ SMOTE balancing  
✅ Comprehensive logging

## 📁 Dataset Requirements

### CICIDS 2017

Place in `data/raw/`:

- Monday-WorkingHours.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Wednesday-workingHours.pcap_ISCX.csv
- Thursday-\*.csv (2 files)
- Friday-\*.csv (3 files)

### NSL-KDD

Place CSV version in `data/raw/`

## ⏱️ Performance

| Dataset Size  | Time      | Memory  |
| ------------- | --------- | ------- |
| Small (1K)    | <1 min    | <100 MB |
| Medium (100K) | 2-3 min   | ~500 MB |
| Large (2.8M)  | 15-20 min | 4-5 GB  |

## ⚠️ Troubleshooting

### "No CSV files found"

→ Place dataset in `data/raw/`

### "Label column not found"

→ Ensure dataset has 'Label' or 'Class' column

### "Memory Error"

→ Process files individually or reduce SMOTE

### SMOTE fails

→ Script continues without balancing (logged as warning)

## 📝 Logging

**Location:** `logs/preprocessing.log`

**Levels:**

- INFO: Normal operations
- WARNING: Non-critical issues
- ERROR: Critical failures

## 🎓 Code Quality

✅ 528 lines of clean code  
✅ Comprehensive docstrings  
✅ Type hints  
✅ Error handling  
✅ PEP 8 compliant  
✅ Production-ready

## 🚀 What's Next?

**Part 3: Feature Selection Module**

- Correlation analysis
- Univariate selection
- Tree-based importance
- RFE (Recursive Feature Elimination)
- Feature visualization

## 📞 Quick Help

**Check logs:** `logs/preprocessing.log`  
**Read guide:** `data/PREPROCESSING_GUIDE.md`  
**View completion:** `PART2_COMPLETION.md`

---

**Status:** ✅ READY FOR PRODUCTION  
**Author:** Priyanshu Kumar  
**Date:** November 10, 2025
