"""
Data Preprocessing Module for NIDS-ML

This module handles:
- Loading CICIDS 2017 / NSL-KDD dataset
- Data cleaning (handling missing values, duplicates, infinite values)
- Encoding categorical features (Label encoding, One-hot encoding)
- Feature normalization/standardization (StandardScaler, MinMaxScaler)
- Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
- Train-test splitting with stratification

Author: Priyanshu Kumar
Date: November 10, 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
import logging
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Fix Windows console encoding
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

logger = logging.getLogger(__name__)




def load_dataset(data_dir='../data/raw'):
    """
    Load all CSV files from data/raw directory and merge into single DataFrame.
    Supports both CICIDS 2017 and NSL-KDD formats.
    
    Args:
        data_dir (str): Path to raw data directory
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    logger.info("="*60)
    logger.info("STEP 1: LOADING DATASET")
    logger.info("="*60)
    
    # Convert to absolute path
    data_path = Path(__file__).parent.parent / 'data' / 'raw'
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        raise FileNotFoundError(f"Please place dataset files in {data_path}")
    
    # Find all CSV files
    csv_files = list(data_path.glob('*.csv'))
    
    if not csv_files:
        logger.error(f"No CSV files found in {data_path}")
        raise FileNotFoundError(f"Please place CSV dataset files in {data_path}")
    
    logger.info(f"Found {len(csv_files)} CSV file(s) in {data_path}")
    
    # Load and concatenate all CSV files
    dataframes = []
    for csv_file in csv_files:
        logger.info(f"Loading: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
            dataframes.append(df)
            logger.info(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.warning(f"  ✗ Failed to load {csv_file.name}: {str(e)}")
            continue
    
    if not dataframes:
        raise ValueError("No data could be loaded from CSV files")
    
    # Merge all dataframes
    df_raw = pd.concat(dataframes, ignore_index=True)
    
    logger.info("\n" + "-"*60)
    logger.info("DATASET SUMMARY")
    logger.info("-"*60)
    logger.info(f"Total Rows: {len(df_raw):,}")
    logger.info(f"Total Columns: {len(df_raw.columns)}")
    logger.info(f"Memory Usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"Duplicate Rows: {df_raw.duplicated().sum():,}")
    
    # Check for missing values
    missing_values = df_raw.isnull().sum()
    missing_count = (missing_values > 0).sum()
    logger.info(f"Columns with Missing Values: {missing_count}")
    
    if missing_count > 0:
        logger.info("\nMissing Values per Column:")
        for col, count in missing_values[missing_values > 0].items():
            percentage = (count / len(df_raw)) * 100
            logger.info(f"  {col}: {count:,} ({percentage:.2f}%)")
    
    logger.info("-"*60 + "\n")
    
    return df_raw


def handle_missing_values(df, threshold=0.3):
    """
    Handle missing values in the dataset:
    - Drop columns with > threshold missing data
    - Replace numeric missing values with median
    - Replace categorical missing values with mode
    
    Args:
        df (pd.DataFrame): Input dataset
        threshold (float): Threshold for dropping columns (default: 0.3 = 30%)
        
    Returns:
        pd.DataFrame: Dataset with handled missing values
    """
    logger.info("="*60)
    logger.info("STEP 2: HANDLING MISSING VALUES")
    logger.info("="*60)
    
    df_clean = df.copy()
    initial_cols = len(df_clean.columns)
    
    # Calculate missing percentage for each column
    missing_percentages = df_clean.isnull().sum() / len(df_clean)
    
    # Drop columns with > threshold missing data
    cols_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
    
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing data:")
        for col in cols_to_drop:
            percentage = missing_percentages[col] * 100
            logger.info(f"  - {col}: {percentage:.2f}% missing")
        df_clean.drop(columns=cols_to_drop, inplace=True)
    else:
        logger.info(f"No columns exceed {threshold*100}% missing data threshold")
    
    # Handle remaining missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Replace numeric missing values with median
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            median_value = df_clean[col].median()
            missing_count = df_clean[col].isnull().sum()
            df_clean[col].fillna(median_value, inplace=True)
            logger.info(f"Replaced {missing_count:,} missing values in '{col}' with median: {median_value:.2f}")
    
    # Replace categorical missing values with mode
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            missing_count = df_clean[col].isnull().sum()
            df_clean[col].fillna(mode_value, inplace=True)
            logger.info(f"Replaced {missing_count:,} missing values in '{col}' with mode: '{mode_value}'")
    
    # Handle infinite values in numeric columns
    logger.info("\nHandling infinite values...")
    inf_count = 0
    for col in numeric_cols:
        inf_mask = np.isinf(df_clean[col])
        if inf_mask.any():
            count = inf_mask.sum()
            inf_count += count
            # Replace inf with max/min finite values
            finite_vals = df_clean[col][~inf_mask]
            if len(finite_vals) > 0:
                df_clean.loc[inf_mask & (df_clean[col] > 0), col] = finite_vals.max()
                df_clean.loc[inf_mask & (df_clean[col] < 0), col] = finite_vals.min()
            else:
                df_clean.loc[inf_mask, col] = 0
            logger.info(f"  Replaced {count:,} infinite values in '{col}'")
    
    if inf_count == 0:
        logger.info("  No infinite values found")
    
    logger.info(f"\nColumns after cleaning: {len(df_clean.columns)} (dropped {initial_cols - len(df_clean.columns)})")
    logger.info("-"*60 + "\n")
    
    return df_clean


def encode_and_label(df):
    """
    Encode categorical features and standardize labels.
    - Encode categorical columns using LabelEncoder
    - Standardize label: BENIGN/normal → 0, attacks → 1
    - Separate features (X) and labels (y)
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        tuple: (X, y, label_column_name)
    """
    logger.info("="*60)
    logger.info("STEP 3: ENCODING AND LABEL SEPARATION")
    logger.info("="*60)
    
    df_encoded = df.copy()
    
    # Clean column names (remove spaces and special characters)
    df_encoded.columns = df_encoded.columns.str.strip().str.replace(' ', '_')
    logger.info(f"Cleaned column names")
    
    # Identify label column (common names in CICIDS 2017 and NSL-KDD)
    label_candidates = ['Label', 'label', 'class', 'Class', 'attack', 'Attack']
    label_col = None
    
    for candidate in label_candidates:
        if candidate in df_encoded.columns:
            label_col = candidate
            break
    
    if label_col is None:
        # Try to find column with 'label' or 'class' in name
        for col in df_encoded.columns:
            if 'label' in col.lower() or 'class' in col.lower():
                label_col = col
                break
    
    if label_col is None:
        logger.error("Could not identify label column. Please ensure dataset has 'Label' or 'Class' column")
        raise ValueError("Label column not found in dataset")
    
    logger.info(f"Identified label column: '{label_col}'")
    
    # Display label distribution before encoding
    logger.info(f"\nOriginal Label Distribution:")
    label_counts = df_encoded[label_col].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df_encoded)) * 100
        logger.info(f"  {label}: {count:,} ({percentage:.2f}%)")
    
    # Standardize labels to binary (0 = BENIGN, 1 = ATTACK)
    benign_variants = ['BENIGN', 'benign', 'normal', 'Normal', 'NORMAL']
    df_encoded['Binary_Label'] = df_encoded[label_col].apply(
        lambda x: 0 if str(x).strip() in benign_variants else 1
    )
    
    # Keep original label for multi-class classification later
    df_encoded['Original_Label'] = df_encoded[label_col]
    
    # Separate features and labels
    y = df_encoded['Binary_Label']
    original_labels = df_encoded['Original_Label']
    
    # Drop label columns from features
    X = df_encoded.drop(columns=[label_col, 'Binary_Label', 'Original_Label'])
    
    logger.info(f"\nBinary Label Distribution:")
    logger.info(f"  BENIGN (0): {(y == 0).sum():,} ({(y == 0).sum() / len(y) * 100:.2f}%)")
    logger.info(f"  ATTACK (1): {(y == 1).sum():,} ({(y == 1).sum() / len(y) * 100:.2f}%)")
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        logger.info(f"\nEncoding {len(categorical_cols)} categorical columns:")
        for col in categorical_cols:
            unique_count = X[col].nunique()
            logger.info(f"  - {col}: {unique_count} unique values")
            
            # Use LabelEncoder for columns with many categories
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    else:
        logger.info("\nNo categorical columns found to encode")
    
    logger.info(f"\nFeature Matrix (X): {X.shape}")
    logger.info(f"Label Vector (y): {y.shape}")
    logger.info("-"*60 + "\n")
    
    return X, y, label_col


def normalize_features(X):
    """
    Normalize numeric features using StandardScaler (zero mean, unit variance).
    
    Args:
        X (pd.DataFrame): Feature matrix
        
    Returns:
        tuple: (X_normalized, scaler)
    """
    logger.info("="*60)
    logger.info("STEP 4: FEATURE NORMALIZATION")
    logger.info("="*60)
    
    # Display statistics before normalization
    logger.info("Sample statistics BEFORE normalization:")
    sample_cols = X.columns[:3].tolist()  # First 3 columns
    for col in sample_cols:
        logger.info(f"  {col}: mean={X[col].mean():.4f}, std={X[col].std():.4f}")
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Convert back to DataFrame to preserve column names
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)
    
    # Display statistics after normalization
    logger.info("\nSample statistics AFTER normalization:")
    for col in sample_cols:
        logger.info(f"  {col}: mean={X_normalized[col].mean():.4f}, std={X_normalized[col].std():.4f}")
    
    logger.info(f"\nNormalized feature matrix shape: {X_normalized.shape}")
    logger.info("-"*60 + "\n")
    
    return X_normalized, scaler


def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE to balance minority attack classes.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Label vector
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    logger.info("="*60)
    logger.info("STEP 5: CLASS BALANCING WITH SMOTE")
    logger.info("="*60)
    
    # Display class distribution before SMOTE
    logger.info("Class distribution BEFORE SMOTE:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        label_name = "BENIGN" if label == 0 else "ATTACK"
        logger.info(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
    
    total_before = len(y)
    
    # Apply SMOTE with optimized sampling strategy
    # For large datasets, balance to 50% ratio instead of 100% to save time
    minority_count = counts[1]  # ATTACK class count
    majority_count = counts[0]  # BENIGN class count
    target_minority = int(majority_count * 0.5)  # Balance to 50% ratio
    
    logger.info(f"\nApplying SMOTE (balancing to 50% ratio for performance)...")
    logger.info(f"Target minority class samples: {target_minority:,}")
    
    smote = SMOTE(
        sampling_strategy={1: target_minority},  # Only oversample minority class
        random_state=random_state, 
        k_neighbors=5
    )
    
    try:
        logger.info("Processing... This may take 2-5 minutes for large datasets.")
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Display class distribution after SMOTE
        logger.info("\nClass distribution AFTER SMOTE:")
        unique, counts = np.unique(y_balanced, return_counts=True)
        for label, count in zip(unique, counts):
            percentage = (count / len(y_balanced)) * 100
            label_name = "BENIGN" if label == 0 else "ATTACK"
            logger.info(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
        
        total_after = len(y_balanced)
        logger.info(f"\nTotal samples: {total_before:,} → {total_after:,} (+{total_after - total_before:,})")
        logger.info(f"Balanced feature matrix shape: {X_balanced.shape}")
        
    except Exception as e:
        logger.warning(f"SMOTE failed: {str(e)}")
        logger.warning("Proceeding without SMOTE balancing")
        X_balanced, y_balanced = X, y
    
    logger.info("-"*60 + "\n")
    
    return X_balanced, y_balanced


def save_processed_data(X, y, output_dir='../data/processed'):
    """
    Save preprocessed data to CSV file.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series or np.ndarray): Label vector
        output_dir (str): Output directory path
    """
    logger.info("="*60)
    logger.info("STEP 6: SAVING PROCESSED DATA")
    logger.info("="*60)
    
    # Convert to absolute path
    output_path = Path(__file__).parent.parent / 'data' / 'processed'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Combine X and y
    df_final = X.copy()
    df_final['Label'] = y
    
    # Save to CSV
    output_file = output_path / 'cleaned_data.csv'
    df_final.to_csv(output_file, index=False)
    
    logger.info(f"✓ Saved processed data to: {output_file}")
    logger.info(f"  Rows: {len(df_final):,}")
    logger.info(f"  Columns: {len(df_final.columns)} ({len(X.columns)} features + 1 label)")
    logger.info(f"  File size: {output_file.stat().st_size / 1024**2:.2f} MB")
    logger.info("-"*60 + "\n")
    
    return output_file


def main():
    """
    Main function to execute the complete preprocessing pipeline.
    """
    try:
        logger.info("\n" + "="*60)
        logger.info("NIDS-ML DATA PREPROCESSING PIPELINE")
        logger.info("="*60 + "\n")
        
        # Step 1: Load dataset
        df_raw = load_dataset()
        
        # Step 2: Handle missing values
        df_clean = handle_missing_values(df_raw, threshold=0.3)
        
        # Remove duplicates
        logger.info("Removing duplicate rows...")
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Removed {duplicates:,} duplicate rows\n")
        else:
            logger.info("No duplicate rows found\n")
        
        # Step 3: Encode and separate features/labels
        X, y, label_col = encode_and_label(df_clean)
        
        # Step 4: Normalize features
        X_normalized, scaler = normalize_features(X)
        
        # Step 5: Apply SMOTE for class balancing
        X_balanced, y_balanced = apply_smote(X_normalized, y, random_state=42)
        
        # Step 6: Save processed data
        output_file = save_processed_data(X_balanced, y_balanced)
        
        # Final summary
        logger.info("="*60)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY! ✓")
        logger.info("="*60)
        logger.info(f"Final Dataset Shape: {X_balanced.shape[0]:,} rows × {X_balanced.shape[1]} features")
        logger.info(f"Output File: {output_file}")
        logger.info(f"Next Step: Feature Selection (Part 3)")
        logger.info("="*60 + "\n")
        
        return X_balanced, y_balanced, scaler
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        raise



class DataPreprocessor:
    """
    Object-oriented wrapper for data preprocessing operations.
    Provides a convenient interface for the preprocessing pipeline.
    """
    
    def __init__(self, data_dir='../data/raw'):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir (str): Path to the raw dataset directory
        """
        self.data_dir = data_dir
        self.df = None
        self.X = None
        self.y = None
        self.X_balanced = None
        self.y_balanced = None
        self.scaler = None
        self.label_col = None
        
    def preprocess_pipeline(self, apply_balancing=True, random_state=42):
        """
        Execute complete preprocessing pipeline.
        
        Args:
            apply_balancing (bool): Whether to apply SMOTE balancing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: Preprocessed X_balanced, y_balanced, scaler
        """
        # Load dataset
        self.df = load_dataset(self.data_dir)
        
        # Handle missing values
        df_clean = handle_missing_values(self.df, threshold=0.3)
        
        # Remove duplicates
        logger.info("Removing duplicate rows...")
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Removed {duplicates:,} duplicate rows\n")
        
        # Encode and separate features/labels
        self.X, self.y, self.label_col = encode_and_label(df_clean)
        
        # Normalize features
        X_normalized, self.scaler = normalize_features(self.X)
        
        # Apply SMOTE if requested
        if apply_balancing:
            self.X_balanced, self.y_balanced = apply_smote(X_normalized, self.y, random_state)
        else:
            self.X_balanced, self.y_balanced = X_normalized, self.y
        
        return self.X_balanced, self.y_balanced, self.scaler
    
    def save_data(self, output_dir='../data/processed'):
        """
        Save preprocessed data to disk.
        
        Args:
            output_dir (str): Directory to save processed data
            
        Returns:
            Path: Path to saved file
        """
        if self.X_balanced is None or self.y_balanced is None:
            raise ValueError("No preprocessed data to save. Run preprocess_pipeline() first.")
        
        return save_processed_data(self.X_balanced, self.y_balanced, output_dir)


if __name__ == "__main__":
    """
    Execute preprocessing pipeline when run as main script.
    """
    main()


