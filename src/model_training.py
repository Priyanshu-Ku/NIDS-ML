"""
Model Training and Evaluation Module for NIDS-ML

This module handles:
- Loading selected features dataset
- Train/test splitting with stratification
- Training multiple ML models (LogisticRegression, RandomForest, SVM, XGBoost, DNN)
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrix and ROC curve visualizations
- Model comparison and selection
- Saving best model for deployment

Author: Priyanshu Kumar
Date: November 11, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import xgboost as xgb
import joblib
import logging
import warnings
from pathlib import Path
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Configure logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

results_dir = Path(__file__).parent.parent / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

plots_dir = results_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

models_dir = Path(__file__).parent.parent / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'model_training.log', encoding='utf-8'),
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
    Load the selected features dataset and split into features and labels.
    
    Args:
        path (str): Path to the dataset CSV file
        
    Returns:
        tuple: (X, y) where X is features DataFrame, y is labels Series
    """
    logger.info("="*60)
    logger.info("STEP 1: LOADING SELECTED FEATURES DATASET")
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
        
        logger.info(f"\n  Features (X): {X.shape}")
        logger.info(f"  Labels (y): {y.shape}")
        logger.info(f"  Feature count: {X.shape[1]}")
        
        # Display label distribution
        logger.info(f"\nLabel Distribution:")
        label_counts = y.value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(y)) * 100
            label_name = "BENIGN" if label == 0 else "ATTACK"
            logger.info(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
        
        logger.info("-"*60 + "\n")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple ML models and return trained models with metadata.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary of trained models with training times
    """
    logger.info("="*60)
    logger.info("STEP 3: TRAINING ML MODELS")
    logger.info("="*60)
    
    # Sample data if too large (for faster SVM training)
    if len(X_train) > 100000:
        logger.info(f"Dataset is large ({len(X_train):,} rows). Using stratified sample of 100K for SVM...")
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=100000, random_state=42)
        for sample_idx, _ in sss.split(X_train, y_train):
            X_train_svm = X_train.iloc[sample_idx] if hasattr(X_train, 'iloc') else X_train[sample_idx]
            y_train_svm = y_train.iloc[sample_idx] if hasattr(y_train, 'iloc') else y_train[sample_idx]
    else:
        X_train_svm = X_train
        y_train_svm = y_train
    
    models = {}
    
    # 1. Logistic Regression
    logger.info("\n1. Training Logistic Regression...")
    start_time = time.time()
    lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1, verbose=0)
    lr.fit(X_train, y_train)
    lr_time = time.time() - start_time
    models['Logistic Regression'] = {'model': lr, 'train_time': lr_time}
    logger.info(f"   ✓ Completed in {lr_time:.2f}s")
    
    # 2. Random Forest
    logger.info("\n2. Training Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_time
    models['Random Forest'] = {'model': rf, 'train_time': rf_time}
    logger.info(f"   ✓ Completed in {rf_time:.2f}s")
    
    # 3. SVM (on sampled data)
    logger.info("\n3. Training SVM (RBF kernel)...")
    start_time = time.time()
    svm = SVC(kernel='rbf', probability=True, random_state=42, verbose=False)
    svm.fit(X_train_svm, y_train_svm)
    svm_time = time.time() - start_time
    models['SVM'] = {'model': svm, 'train_time': svm_time}
    logger.info(f"   ✓ Completed in {svm_time:.2f}s")
    
    # 4. XGBoost
    logger.info("\n4. Training XGBoost...")
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(
        tree_method='hist',
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start_time
    models['XGBoost'] = {'model': xgb_model, 'train_time': xgb_time}
    logger.info(f"   ✓ Completed in {xgb_time:.2f}s")
    
    logger.info(f"\n✓ All models trained successfully!")
    logger.info("-"*60 + "\n")
    
    return models


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model with comprehensive metrics and visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    cm_path = plots_dir / f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved confusion matrix to: {cm_path.name}")
    plt.close()
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc_calc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_calc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = plots_dir / f'roc_curve_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved ROC curve to: {roc_path.name}")
    plt.close()
    
    # Classification report
    class_report = classification_report(y_test, y_pred, target_names=['Benign', 'Attack'])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': class_report
    }


def compare_models(results):
    """
    Compare all models and create a summary table.
    
    Args:
        results: Dictionary of model evaluation results
        
    Returns:
        pd.DataFrame: Comparison table
    """
    logger.info("="*60)
    logger.info("STEP 5: MODEL COMPARISON")
    logger.info("="*60)
    
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'Train Time (s)': metrics.get('train_time', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    logger.info("\nModel Performance Comparison:")
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_path = results_dir / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\n✓ Saved comparison table to: {comparison_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(comparison_df))
    width = 0.15
    
    for i, metric in enumerate(metrics_to_plot):
        offset = width * (i - 2)
        plt.bar(x + offset, comparison_df[metric], width, label=metric, alpha=0.8)
    
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, comparison_df['Model'], rotation=15, ha='right')
    plt.ylim([0, 1.05])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    comparison_plot_path = plots_dir / 'model_comparison.png'
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved comparison plot to: {comparison_plot_path.name}")
    plt.close()
    
    logger.info("-"*60 + "\n")
    
    return comparison_df


def save_best_model(models, results, comparison_df):
    """
    Select and save the best performing model based on F1-Score.
    
    Args:
        models: Dictionary of trained models
        results: Dictionary of evaluation results
        comparison_df: Comparison DataFrame
        
    Returns:
        tuple: (best_model_name, best_model)
    """
    logger.info("="*60)
    logger.info("STEP 6: SELECTING AND SAVING BEST MODEL")
    logger.info("="*60)
    
    # Get best model by F1-Score
    best_model_name = comparison_df.iloc[0]['Model']
    best_f1 = comparison_df.iloc[0]['F1-Score']
    best_auc = comparison_df.iloc[0]['ROC-AUC']
    
    logger.info(f"\n🏆 Best Model: {best_model_name}")
    logger.info(f"   F1-Score: {best_f1:.4f}")
    logger.info(f"   ROC-AUC:  {best_auc:.4f}")
    
    # Get the actual model object
    best_model = models[best_model_name]['model']
    
    # Save model
    model_path = models_dir / 'best_model.pkl'
    joblib.dump(best_model, model_path)
    logger.info(f"\n✓ Saved best model to: {model_path}")
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'f1_score': float(best_f1),
        'roc_auc': float(best_auc),
        'accuracy': float(comparison_df.iloc[0]['Accuracy']),
        'precision': float(comparison_df.iloc[0]['Precision']),
        'recall': float(comparison_df.iloc[0]['Recall']),
        'train_time': float(comparison_df.iloc[0]['Train Time (s)']),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = models_dir / 'best_model_metadata.txt'
    with open(metadata_path, 'w') as f:
        f.write("Best Model Metadata\n")
        f.write("="*60 + "\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"✓ Saved metadata to: {metadata_path}")
    
    # Save classification reports
    report_path = results_dir / 'model_metrics.txt'
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("NIDS-ML MODEL EVALUATION METRICS\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"MODEL: {model_name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write("-"*60 + "\n")
            f.write(metrics['classification_report'])
            f.write("\n")
    
    logger.info(f"✓ Saved all classification reports to: {report_path}")
    logger.info("-"*60 + "\n")
    
    return best_model_name, best_model


def main():
    """
    Main execution function for model training and evaluation pipeline.
    """
    logger.info("\n" + "="*60)
    logger.info("NIDS-ML MODEL TRAINING AND EVALUATION")
    logger.info("="*60 + "\n")
    
    overall_start_time = time.time()
    
    try:
        # Step 1: Load data
        X, y = load_data('data/processed/selected_features.csv')
        
        # Step 2: Train/Test Split
        logger.info("="*60)
        logger.info("STEP 2: TRAIN/TEST SPLIT")
        logger.info("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"Test set:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Check stratification
        logger.info("\nTrain set distribution:")
        train_counts = y_train.value_counts()
        for label, count in train_counts.items():
            percentage = (count / len(y_train)) * 100
            label_name = "BENIGN" if label == 0 else "ATTACK"
            logger.info(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
        
        logger.info("\nTest set distribution:")
        test_counts = y_test.value_counts()
        for label, count in test_counts.items():
            percentage = (count / len(y_test)) * 100
            label_name = "BENIGN" if label == 0 else "ATTACK"
            logger.info(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
        
        logger.info("-"*60 + "\n")
        
        # Step 3: Train models
        trained_models = train_models(X_train, y_train, X_test, y_test)
        
        # Step 4: Evaluate models
        logger.info("="*60)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("="*60 + "\n")
        
        results = {}
        for model_name, model_info in trained_models.items():
            model = model_info['model']
            train_time = model_info['train_time']
            
            metrics = evaluate_model(model, X_test, y_test, model_name)
            metrics['train_time'] = train_time
            results[model_name] = metrics
            logger.info("")
        
        logger.info("-"*60 + "\n")
        
        # Step 5: Compare models
        comparison_df = compare_models(results)
        
        # Step 6: Save best model
        best_model_name, best_model = save_best_model(trained_models, results, comparison_df)
        
        # Final Summary
        overall_time = time.time() - overall_start_time
        
        logger.info("="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total execution time: {overall_time:.2f}s ({overall_time/60:.2f} minutes)")
        logger.info(f"Models trained: {len(trained_models)}")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
        logger.info("="*60 + "\n")
        
        logger.info("✓ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("\nOutput Files Generated:")
        logger.info(f"  1. models/best_model.pkl (Saved {best_model_name})")
        logger.info(f"  2. models/best_model_metadata.txt")
        logger.info(f"  3. results/model_metrics.txt (All classification reports)")
        logger.info(f"  4. results/model_comparison.csv")
        logger.info(f"  5. results/plots/confusion_matrix_*.png (4 files)")
        logger.info(f"  6. results/plots/roc_curve_*.png (4 files)")
        logger.info(f"  7. results/plots/model_comparison.png")
        logger.info("\nNext Step: Explainable AI & Real-Time Detection (Part 5)")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
