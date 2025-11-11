"""
Feature Selection and Importance Analysis Module for NIDS-ML

This module handles:
- Correlation analysis to identify redundant features
- Model-based feature importance (Random Forest)
- Recursive Feature Elimination (RFE)
- Variance threshold filtering
- Feature ranking and visualization
- Output selected features for downstream ML models

Author: Priyanshu Kumar
Date: November 11, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, VarianceThreshold
import logging
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore')

# Configure logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

results_dir = Path(__file__).parent.parent / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

plots_dir = results_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'feature_selection.log', encoding='utf-8'),
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


def load_data(path):
    """
    Load the preprocessed dataset and split into features and labels.
    
    Args:
        path (str): Path to the cleaned dataset CSV file
        
    Returns:
        tuple: (X, y, feature_names) where X is features DataFrame, y is labels Series
    """
    logger.info("="*60)
    logger.info("STEP 1: LOADING PREPROCESSED DATA")
    logger.info("="*60)
    
    data_path = Path(__file__).parent.parent / path
    logger.info(f"Loading data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"✓ Loaded dataset with shape: {df.shape}")
        logger.info(f"  Rows: {len(df):,}")
        logger.info(f"  Columns: {len(df.columns)}")
        
        # Identify label column
        label_cols = ['Label', 'label', 'Attack', 'attack', 'Class', 'class']
        label_col = None
        
        for col in label_cols:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            raise ValueError(f"No label column found. Expected one of: {label_cols}")
        
        logger.info(f"✓ Identified label column: '{label_col}'")
        
        # Split features and labels
        y = df[label_col]
        X = df.drop(columns=[label_col])
        
        feature_names = X.columns.tolist()
        
        logger.info(f"\n  Features (X): {X.shape}")
        logger.info(f"  Labels (y): {y.shape}")
        logger.info(f"  Total features: {len(feature_names)}")
        
        # Display label distribution
        logger.info(f"\nLabel Distribution:")
        label_counts = y.value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(y)) * 100
            label_name = "BENIGN" if label == 0 else "ATTACK"
            logger.info(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
        
        logger.info("-"*60 + "\n")
        
        return X, y, feature_names
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def correlation_analysis(X, threshold=0.9):
    """
    Analyze feature correlations and identify highly correlated pairs.
    
    Args:
        X (pd.DataFrame): Feature matrix
        threshold (float): Correlation threshold (default 0.9)
        
    Returns:
        list: Pairs of highly correlated features
    """
    logger.info("="*60)
    logger.info("STEP 2: CORRELATION ANALYSIS")
    logger.info("="*60)
    logger.info(f"Computing correlation matrix for {X.shape[1]} features...")
    
    # Compute correlation matrix
    corr_matrix = X.corr()
    
    logger.info(f"✓ Correlation matrix computed: {corr_matrix.shape}")
    
    # Find highly correlated pairs
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    logger.info(f"\nHighly Correlated Feature Pairs (|corr| > {threshold}):")
    logger.info(f"Found {len(high_corr_pairs)} pairs")
    
    if len(high_corr_pairs) > 0:
        logger.info("\nTop 10 highly correlated pairs:")
        for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
            logger.info(f"  {feat1} <-> {feat2}: {corr_val:.4f}")
    
    # Plot correlation heatmap (for top features to avoid overcrowding)
    logger.info("\nGenerating correlation heatmap...")
    
    # Select top 30 features by variance for visualization
    top_features = X.var().nlargest(30).index
    corr_subset = X[top_features].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_subset, 
                annot=False, 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap (Top 30 Features by Variance)', 
              fontsize=14, 
              fontweight='bold')
    plt.tight_layout()
    
    heatmap_path = plots_dir / 'correlation_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved correlation heatmap to: {heatmap_path}")
    plt.close()
    
    logger.info("-"*60 + "\n")
    
    return high_corr_pairs


def model_feature_importance(X, y, n_estimators=100, top_n=20):
    """
    Calculate feature importance using Random Forest.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Labels
        n_estimators (int): Number of trees in Random Forest
        top_n (int): Number of top features to display
        
    Returns:
        pd.DataFrame: Feature importance scores sorted by importance
    """
    logger.info("="*60)
    logger.info("STEP 3: MODEL-BASED FEATURE IMPORTANCE (Random Forest)")
    logger.info("="*60)
    
    logger.info(f"Training RandomForestClassifier with {n_estimators} estimators...")
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        max_depth=10,  # Limit depth to speed up training on large dataset
        verbose=0
    )
    
    # Sample data if too large (for faster training)
    if len(X) > 100000:
        logger.info(f"Dataset is large ({len(X):,} rows). Sampling 100,000 rows for faster training...")
        sample_indices = np.random.choice(len(X), 100000, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
    else:
        X_sample = X
        y_sample = y
    
    rf.fit(X_sample, y_sample)
    logger.info("✓ Random Forest training complete")
    
    # Extract feature importances
    importances = rf.feature_importances_
    feature_names = X.columns
    
    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    logger.info(f"\nTop {top_n} Most Important Features:")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']:.6f}")
    
    # Save feature importance to CSV
    importance_path = results_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"\n✓ Saved feature importance to: {importance_path}")
    
    # Plot feature importance
    logger.info("\nGenerating feature importance bar chart...")
    
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)
    
    bars = plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Features by Random Forest Importance', 
              fontsize=14, 
              fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    importance_plot_path = plots_dir / 'feature_importance_top20.png'
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved feature importance plot to: {importance_plot_path}")
    plt.close()
    
    logger.info("-"*60 + "\n")
    
    return importance_df


def recursive_feature_elimination(X, y, n_features=30):
    """
    Use RFE with Logistic Regression to select best features.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Labels
        n_features (int): Number of features to select
        
    Returns:
        list: Selected feature names
    """
    logger.info("="*60)
    logger.info("STEP 4: RECURSIVE FEATURE ELIMINATION (RFE)")
    logger.info("="*60)
    
    logger.info(f"Running RFE to select {n_features} best features...")
    logger.info("Using LogisticRegression as base estimator (max_iter=500)...")
    
    # Sample data if too large
    if len(X) > 100000:
        logger.info(f"Dataset is large ({len(X):,} rows). Sampling 100,000 rows for RFE...")
        sample_indices = np.random.choice(len(X), 100000, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
    else:
        X_sample = X
        y_sample = y
    
    # Create RFE selector
    estimator = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
    selector = RFE(estimator=estimator, n_features_to_select=n_features, step=1, verbose=0)
    
    # Fit RFE
    logger.info("Training RFE selector (this may take a few minutes)...")
    selector.fit(X_sample, y_sample)
    
    # Get selected features
    selected_mask = selector.support_
    selected_features = X.columns[selected_mask].tolist()
    
    logger.info(f"\n✓ RFE completed. Selected {len(selected_features)} features:")
    for i, feat in enumerate(selected_features, 1):
        logger.info(f"  {i}. {feat}")
    
    # Save selected features to file
    selected_features_path = results_dir / 'selected_features.txt'
    with open(selected_features_path, 'w', encoding='utf-8') as f:
        f.write("Selected Features (RFE)\n")
        f.write("="*60 + "\n")
        f.write(f"Total Selected: {len(selected_features)}\n")
        f.write(f"Selection Method: Recursive Feature Elimination (RFE)\n")
        f.write(f"Base Estimator: LogisticRegression(max_iter=500)\n")
        f.write("="*60 + "\n\n")
        for i, feat in enumerate(selected_features, 1):
            f.write(f"{i}. {feat}\n")
    
    logger.info(f"\n✓ Saved selected features to: {selected_features_path}")
    logger.info("-"*60 + "\n")
    
    return selected_features


def variance_threshold_filter(X, threshold=0.0):
    """
    Remove features with low variance (constant or near-constant).
    
    Args:
        X (pd.DataFrame): Feature matrix
        threshold (float): Variance threshold
        
    Returns:
        tuple: (X_filtered, removed_features)
    """
    logger.info("="*60)
    logger.info("STEP 5: VARIANCE THRESHOLD FILTERING")
    logger.info("="*60)
    
    logger.info(f"Applying VarianceThreshold (threshold={threshold})...")
    
    initial_features = X.shape[1]
    
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    removed_features = X.columns[~selected_mask].tolist()
    
    # Convert back to DataFrame
    X_filtered = pd.DataFrame(X_filtered, columns=selected_features, index=X.index)
    
    logger.info(f"✓ Variance filtering complete")
    logger.info(f"  Initial features: {initial_features}")
    logger.info(f"  Remaining features: {len(selected_features)}")
    logger.info(f"  Removed features: {len(removed_features)}")
    
    if len(removed_features) > 0:
        logger.info(f"\nRemoved features (low variance):")
        for feat in removed_features[:10]:  # Show first 10
            logger.info(f"  - {feat}")
        if len(removed_features) > 10:
            logger.info(f"  ... and {len(removed_features) - 10} more")
    
    logger.info("-"*60 + "\n")
    
    return X_filtered, removed_features


def save_selected_features(X_selected, y, output_path):
    """
    Save the dataset with only selected features.
    
    Args:
        X_selected (pd.DataFrame): Selected features
        y (pd.Series): Labels
        output_path (str): Path to save the CSV file
        
    Returns:
        str: Path where file was saved
    """
    logger.info("="*60)
    logger.info("STEP 6: SAVING SELECTED FEATURES DATASET")
    logger.info("="*60)
    
    # Combine features and labels
    df_selected = X_selected.copy()
    df_selected['Label'] = y.values
    
    # Save to CSV
    save_path = Path(__file__).parent.parent / output_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving selected features dataset...")
    logger.info(f"  Shape: {df_selected.shape}")
    logger.info(f"  Features: {X_selected.shape[1]}")
    logger.info(f"  Samples: {len(df_selected):,}")
    
    df_selected.to_csv(save_path, index=False)
    
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"\n✓ Saved selected features dataset to: {save_path}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info("-"*60 + "\n")
    
    return str(save_path)


def main():
    """
    Main execution function for feature selection pipeline.
    """
    logger.info("\n" + "="*60)
    logger.info("NIDS-ML FEATURE SELECTION AND IMPORTANCE ANALYSIS")
    logger.info("="*60 + "\n")
    
    try:
        # Step 1: Load data
        X, y, feature_names = load_data('data/processed/cleaned_data.csv')
        
        initial_feature_count = X.shape[1]
        
        # Step 2: Correlation analysis
        high_corr_pairs = correlation_analysis(X, threshold=0.9)
        
        # Step 3: Model-based feature importance
        importance_df = model_feature_importance(X, y, n_estimators=100, top_n=20)
        
        # Step 4: Recursive Feature Elimination
        selected_features = recursive_feature_elimination(X, y, n_features=30)
        
        # Step 5: Variance threshold filtering (optional)
        X_filtered, removed_features = variance_threshold_filter(X, threshold=0.0)
        
        # Get final selected features (intersection of RFE and variance filtering)
        final_selected = [f for f in selected_features if f in X_filtered.columns]
        X_final = X_filtered[final_selected]
        
        # Step 6: Save selected features
        save_selected_features(X_final, y, 'data/processed/selected_features.csv')
        
        # Final summary
        logger.info("="*60)
        logger.info("FEATURE SELECTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Initial feature count: {initial_feature_count}")
        logger.info(f"After variance filtering: {X_filtered.shape[1]}")
        logger.info(f"After RFE selection: {len(selected_features)}")
        logger.info(f"Final selected features: {len(final_selected)}")
        logger.info(f"Reduction: {initial_feature_count - len(final_selected)} features")
        logger.info(f"Percentage reduction: {((initial_feature_count - len(final_selected)) / initial_feature_count * 100):.2f}%")
        logger.info("="*60 + "\n")
        
        logger.info("✓ FEATURE SELECTION COMPLETED SUCCESSFULLY!")
        logger.info("\nOutput Files Generated:")
        logger.info(f"  1. results/feature_importance.csv")
        logger.info(f"  2. results/selected_features.txt")
        logger.info(f"  3. results/plots/correlation_heatmap.png")
        logger.info(f"  4. results/plots/feature_importance_top20.png")
        logger.info(f"  5. data/processed/selected_features.csv")
        logger.info("\nNext Step: Model Training and Evaluation (Part 4)")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
