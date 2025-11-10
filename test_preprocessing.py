"""
Quick Test Script for Data Preprocessing Module

This script creates a synthetic test dataset and runs the preprocessing pipeline
to verify everything works correctly before using real CICIDS 2017 data.

Usage:
    python test_preprocessing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import main, DataPreprocessor

def create_test_dataset():
    """Create a synthetic test dataset mimicking CICIDS 2017 structure."""
    
    print("\n" + "="*60)
    print("CREATING SYNTHETIC TEST DATASET")
    print("="*60 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create test data (1000 samples)
    n_samples = 1000
    
    # Generate numeric features (similar to CICIDS 2017)
    data = {
        'Flow_Duration': np.random.randint(0, 1000000, n_samples),
        'Total_Fwd_Packets': np.random.randint(1, 100, n_samples),
        'Total_Backward_Packets': np.random.randint(1, 100, n_samples),
        'Total_Length_of_Fwd_Packets': np.random.randint(0, 50000, n_samples),
        'Total_Length_of_Bwd_Packets': np.random.randint(0, 50000, n_samples),
        'Flow_Bytes/s': np.random.uniform(0, 100000, n_samples),
        'Flow_Packets/s': np.random.uniform(0, 1000, n_samples),
        'Flow_IAT_Mean': np.random.uniform(0, 10000, n_samples),
        'Flow_IAT_Std': np.random.uniform(0, 5000, n_samples),
        'Fwd_IAT_Mean': np.random.uniform(0, 10000, n_samples),
        'Bwd_IAT_Mean': np.random.uniform(0, 10000, n_samples),
        'Fwd_PSH_Flags': np.random.randint(0, 5, n_samples),
        'Bwd_PSH_Flags': np.random.randint(0, 5, n_samples),
        'Fwd_Header_Length': np.random.randint(20, 60, n_samples),
        'Bwd_Header_Length': np.random.randint(20, 60, n_samples),
        'Min_Packet_Length': np.random.randint(0, 100, n_samples),
        'Max_Packet_Length': np.random.randint(100, 1500, n_samples),
        'Packet_Length_Mean': np.random.uniform(100, 800, n_samples),
        'Packet_Length_Std': np.random.uniform(0, 500, n_samples),
        'FIN_Flag_Count': np.random.randint(0, 3, n_samples),
        'SYN_Flag_Count': np.random.randint(0, 3, n_samples),
        'RST_Flag_Count': np.random.randint(0, 3, n_samples),
        'ACK_Flag_Count': np.random.randint(0, 50, n_samples),
        'URG_Flag_Count': np.random.randint(0, 2, n_samples),
        'Average_Packet_Size': np.random.uniform(100, 800, n_samples),
        'Init_Win_bytes_forward': np.random.randint(0, 65535, n_samples),
        'Init_Win_bytes_backward': np.random.randint(0, 65535, n_samples),
        'Active_Mean': np.random.uniform(0, 10000, n_samples),
        'Active_Std': np.random.uniform(0, 5000, n_samples),
        'Idle_Mean': np.random.uniform(0, 10000, n_samples),
        'Idle_Std': np.random.uniform(0, 5000, n_samples),
    }
    
    # Generate labels (80% BENIGN, 10% DDoS, 5% PortScan, 5% Bot)
    labels = np.random.choice(
        ['BENIGN', 'DDoS', 'PortScan', 'Bot'],
        n_samples,
        p=[0.80, 0.10, 0.05, 0.05]
    )
    
    data['Label'] = labels
    
    # Add some missing values (5% randomly)
    df = pd.DataFrame(data)
    mask = np.random.random(df.shape) < 0.05
    df = df.mask(mask)
    
    # Add some duplicates (2%)
    n_duplicates = int(n_samples * 0.02)
    duplicate_indices = np.random.choice(df.index, n_duplicates, replace=False)
    df_duplicates = df.loc[duplicate_indices]
    df = pd.concat([df, df_duplicates], ignore_index=True)
    
    # Save to data/raw/
    output_dir = Path(__file__).parent / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'test_dataset.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Created test dataset: {output_file}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Missing values: {df.isnull().sum().sum():,}")
    print(f"  Duplicates: {df.duplicated().sum():,}")
    print(f"\nLabel Distribution:")
    for label, count in df['Label'].value_counts().items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    print()
    
    return output_file


def cleanup_test_data():
    """Remove test dataset after testing."""
    test_file = Path(__file__).parent / 'data' / 'raw' / 'test_dataset.csv'
    if test_file.exists():
        test_file.unlink()
        print(f"✓ Cleaned up test file: {test_file}\n")


if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("TESTING DATA PREPROCESSING MODULE")
        print("="*60)
        
        # Step 1: Create test dataset
        test_file = create_test_dataset()
        
        # Step 2: Run preprocessing
        print("="*60)
        print("RUNNING PREPROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Run main preprocessing function
        X_balanced, y_balanced, scaler = main()
        
        # Verify results
        print("\n" + "="*60)
        print("VERIFICATION")
        print("="*60)
        print(f"✓ Preprocessing completed successfully!")
        print(f"✓ Final dataset shape: {X_balanced.shape}")
        print(f"✓ Features: {X_balanced.shape[1]}")
        print(f"✓ Samples: {X_balanced.shape[0]:,}")
        print(f"✓ Label distribution balanced: {len(set(y_balanced))}")
        print(f"✓ Scaler fitted: {scaler is not None}")
        
        # Check output file
        output_file = Path(__file__).parent / 'data' / 'processed' / 'cleaned_data.csv'
        if output_file.exists():
            print(f"✓ Output file created: {output_file}")
            print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY! ✓")
        print("="*60)
        print("\nNext Steps:")
        print("1. Place your real CICIDS 2017 CSV files in data/raw/")
        print("2. Run: python src/data_preprocessing.py")
        print("3. Proceed to Part 3: Feature Selection")
        print("="*60 + "\n")
        
        # Cleanup
        response = input("Remove test files? (y/n): ").lower()
        if response == 'y':
            cleanup_test_data()
            processed_file = Path(__file__).parent / 'data' / 'processed' / 'cleaned_data.csv'
            if processed_file.exists():
                processed_file.unlink()
                print(f"✓ Cleaned up: {processed_file}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
