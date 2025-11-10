# Data Directory

This directory contains all datasets for the NIDS-ML project.

## Directory Structure:

```
data/
├── raw/                    # Original unprocessed datasets
│   ├── CICIDS2017/        # CICIDS 2017 dataset files
│   └── NSL-KDD/           # NSL-KDD dataset (fallback)
├── processed/              # Preprocessed and cleaned data
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
└── README.md              # This file
```

## Dataset Information

### CICIDS 2017 (Preferred)

- **Download**: https://www.unb.ca/cic/datasets/ids-2017.html
- **Description**: Realistic network intrusion dataset
- **Features**: 80+ network traffic features
- **Attack Types**: DDoS, PortScan, Brute Force, Web Attacks, Infiltration, Botnet
- **Size**: ~2.8 million records

**Files to Download:**

- Monday-WorkingHours.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Wednesday-WorkingHours.pcap_ISCX.csv
- Thursday-WorkingHours.pcap_ISCX.csv
- Friday-WorkingHours.pcap_ISCX.csv

### NSL-KDD (Fallback)

- **Download**: https://www.unb.ca/cic/datasets/nsl.html
- **Description**: Improved version of KDD Cup 99
- **Features**: 41 network features
- **Attack Categories**: DoS, Probe, R2L, U2R
- **Size**: 125,973 training + 22,544 test records

## Setup Instructions

1. **Download the dataset** from the links above
2. **Extract CSV files** to `data/raw/` folder
3. **Run preprocessing**:
   ```bash
   cd src
   python data_preprocessing.py
   ```
4. **Processed data** will be saved to `data/processed/`

## Notes

- Large CSV files are NOT tracked by git (see `.gitignore`)
- Preprocessed numpy arrays are also ignored to save space
- Share datasets separately or download independently
