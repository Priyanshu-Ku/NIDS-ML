"""
Utility Functions for NIDS-ML Project

This module contains common utility functions used across the project.

Author: [Your Name]
Date: November 10, 2025
"""

import logging
import os
import json
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file (str): Path to log file
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        

def save_json(data: Dict, filepath: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data (dict): Data to save
        filepath (str): Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)


def load_json(filepath: str) -> Dict:
    """
    Load JSON file.
    
    Args:
        filepath (str): JSON file path
        
    Returns:
        dict: Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object using pickle.
    
    Args:
        obj: Object to save
        filepath (str): Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load pickled object.
    
    Args:
        filepath (str): Pickle file path
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_dataset_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print comprehensive dataset information.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        name (str): Dataset name
    """
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print(f"{'='*60}\n")


def calculate_class_distribution(y: np.ndarray) -> Dict[Any, int]:
    """
    Calculate class distribution.
    
    Args:
        y (np.ndarray): Labels
        
    Returns:
        dict: Class distribution
    """
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


def print_class_distribution(y: np.ndarray, name: str = "Labels") -> None:
    """
    Print class distribution.
    
    Args:
        y (np.ndarray): Labels
        name (str): Dataset name
    """
    dist = calculate_class_distribution(y)
    print(f"\n{name} Distribution:")
    print("-" * 40)
    total = sum(dist.values())
    for label, count in sorted(dist.items()):
        percentage = (count / total) * 100
        print(f"{label}: {count:,} ({percentage:.2f}%)")
    print("-" * 40)


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


if __name__ == "__main__":
    # Test utility functions
    logger = setup_logging()
    logger.info("Utility module loaded successfully")
