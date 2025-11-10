"""
Configuration file for NIDS-ML Project

This file contains all configuration parameters and constants used throughout the project.

Author: [Your Name]
Date: November 10, 2025
"""

import os
from pathlib import Path

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory Paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'
SRC_DIR = BASE_DIR / 'src'

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
DATASET_CONFIG = {
    'type': 'CICIDS2017',  # Options: 'CICIDS2017', 'NSL-KDD'
    'file_path': RAW_DATA_DIR / 'dataset.csv',
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'handle_missing': 'drop',  # Options: 'drop', 'mean', 'median', 'mode'
    'remove_duplicates': True,
    'encoding_method': 'label',  # Options: 'label', 'onehot'
    'scaling_method': 'standard',  # Options: 'standard', 'minmax'
    'apply_smote': True,
    'smote_sampling_strategy': 'auto',
    'smote_k_neighbors': 5
}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {
    'method': 'tree_based',  # Options: 'correlation', 'univariate', 'rfe', 'tree_based', 'mi'
    'n_features': 20,
    'correlation_threshold': 0.95,
    'use_pca': False,
    'pca_variance': 0.95
}

# Model Training Configuration
MODEL_TRAINING_CONFIG = {
    'models_to_train': [
        'Random Forest',
        'XGBoost',
        'SVM',
        'KNN',
        'Logistic Regression',
        'Naive Bayes'
    ],
    'perform_tuning': True,
    'cv_folds': 5,
    'n_jobs': -1,
    'verbose': 1
}

# Hyperparameter Grids
PARAM_GRIDS = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'linear']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
}

# Deep Learning Configuration
DL_CONFIG = {
    'lstm_units': [128, 64],
    'cnn_filters': [64, 32],
    'dropout_rate': 0.3,
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.2,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}

# Real-time Detection Configuration
REALTIME_CONFIG = {
    'interface': None,  # None = all interfaces, or specify like 'eth0', 'wlan0'
    'filter_rule': None,  # BPF filter, e.g., 'tcp port 80'
    'packet_buffer_size': 1000,
    'detection_threshold': 0.85,
    'log_intrusions': True,
    'alert_enabled': True
}

# SHAP Configuration
SHAP_CONFIG = {
    'explainer_type': 'tree',  # Options: 'tree', 'kernel', 'linear', 'deep'
    'background_samples': 100,
    'plot_format': 'png',
    'max_display': 20
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'host': 'localhost',
    'port': 8501,
    'theme': 'light',  # Options: 'light', 'dark'
    'refresh_interval': 5  # seconds
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'files': {
        'preprocessing': LOGS_DIR / 'preprocessing.log',
        'feature_selection': LOGS_DIR / 'feature_selection.log',
        'model_training': LOGS_DIR / 'model_training.log',
        'explainability': LOGS_DIR / 'explainability.log',
        'realtime_detection': LOGS_DIR / 'realtime_detection.log'
    }
}

# Feature Names (CICIDS 2017)
CICIDS_FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
    'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
    'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
    'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# Attack Labels (CICIDS 2017)
ATTACK_LABELS = [
    'BENIGN',
    'DDoS',
    'PortScan',
    'Bot',
    'Infiltration',
    'Web Attack – Brute Force',
    'Web Attack – XSS',
    'Web Attack – Sql Injection',
    'FTP-Patator',
    'SSH-Patator',
    'DoS slowloris',
    'DoS Slowhttptest',
    'DoS Hulk',
    'DoS GoldenEye',
    'Heartbleed'
]

# Color Scheme for Visualizations
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17becf',
    'benign': '#2ca02c',
    'attack': '#d62728'
}

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
