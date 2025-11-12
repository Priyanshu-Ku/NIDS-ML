"""
NIDS-ML: Explainable AI (XAI) + Real-Time Detection Module
============================================================

Part 5: Model Interpretability and Live Traffic Classification

This module provides:
1. SHAP-based model explainability for Random Forest
2. Real-time packet capture and classification
3. Live traffic monitoring with predictions logging

Author: Priyanshu
Project: Network Intrusion Detection System using ML and DL
Version: 1.0
"""

import sys
import os
import logging
import warnings
from pathlib import Path
from datetime import datetime
import time
import traceback

# Data handling
import pandas as pd
import numpy as np
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP for explainability
import shap

# Scapy for packet capture (optional - graceful fallback)
try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("⚠️  Scapy not available. Real-time capture will use simulation mode.")

# ML libraries
from sklearn.preprocessing import StandardScaler

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
log_file = LOGS_DIR / 'explainability_realtime.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Real-time detection configuration
REALTIME_LOG = LOGS_DIR / 'realtime_detection.log'
REALTIME_CSV = RESULTS_DIR / 'realtime_predictions.csv'
CAPTURE_COUNT = 50  # Number of packets to capture in simulation mode
CONFIDENCE_THRESHOLD = 0.5


# ============================================================
# SECTION 1: EXPLAINABLE AI (XAI) WITH SHAP
# ============================================================

def load_model_and_data(model_path=None, data_path=None):
    """
    Load the trained model and processed dataset for SHAP analysis.
    
    Parameters:
    -----------
    model_path : str or Path, optional
        Path to the saved model file (default: models/best_model.pkl)
    data_path : str or Path, optional
        Path to the selected features dataset (default: data/processed/selected_features.csv)
    
    Returns:
    --------
    tuple
        (model, X_train_sample, y_train_sample, feature_names)
        - model: Trained ML model
        - X_train_sample: Sample of training data for SHAP (10K rows)
        - y_train_sample: Corresponding labels
        - feature_names: List of feature column names
    
    Raises:
    -------
    FileNotFoundError
        If model or data file doesn't exist
    """
    logger.info("=" * 60)
    logger.info("LOADING MODEL AND DATA FOR SHAP ANALYSIS")
    logger.info("=" * 60)
    
    # Set default paths
    if model_path is None:
        model_path = MODELS_DIR / 'best_model.pkl'
    if data_path is None:
        data_path = DATA_DIR / 'selected_features.csv'
    
    # Load trained model
    logger.info(f"Loading model from: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"✓ Loaded model: {type(model).__name__}")
    
    # Load dataset
    logger.info(f"Loading data from: {data_path}")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded dataset with shape: {df.shape}")
    
    # Identify label column
    label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    logger.info(f"✓ Identified label column: '{label_col}'")
    
    # Split features and labels
    X = df.drop(columns=[label_col])
    y = df[label_col]
    feature_names = X.columns.tolist()
    
    logger.info(f"  Features: {X.shape}")
    logger.info(f"  Labels: {y.shape}")
    logger.info(f"  Feature count: {len(feature_names)}")
    
    # Sample data for SHAP (use 10K samples to balance speed vs accuracy)
    sample_size = min(10000, len(X))
    logger.info(f"\nSampling {sample_size:,} rows for SHAP analysis...")
    
    # Stratified sampling to maintain class distribution
    benign_indices = y[y == 0].sample(n=sample_size // 2, random_state=42).index
    attack_indices = y[y == 1].sample(n=sample_size // 2, random_state=42).index
    sample_indices = benign_indices.union(attack_indices)
    
    X_sample = X.loc[sample_indices]
    y_sample = y.loc[sample_indices]
    
    logger.info(f"✓ Sample distribution:")
    logger.info(f"  BENIGN (0): {(y_sample == 0).sum():,} ({(y_sample == 0).sum() / len(y_sample) * 100:.2f}%)")
    logger.info(f"  ATTACK (1): {(y_sample == 1).sum():,} ({(y_sample == 1).sum() / len(y_sample) * 100:.2f}%)")
    logger.info("-" * 60 + "\n")
    
    return model, X_sample, y_sample, feature_names


def explain_model_with_shap(model, X_sample, feature_names, top_n=10):
    """
    Generate SHAP explanations for the trained model.
    
    This function computes SHAP values to interpret model predictions,
    visualizes global feature importance, and creates dependence plots
    for top contributing features.
    
    Parameters:
    -----------
    model : sklearn model
        Trained machine learning model
    X_sample : pd.DataFrame or np.ndarray
        Sample data for SHAP analysis (recommended: 5K-10K samples)
    feature_names : list
        List of feature column names
    top_n : int, default=10
        Number of top features to analyze in detail
    
    Returns:
    --------
    tuple
        (shap_values, explainer, top_features)
        - shap_values: SHAP values for all samples
        - explainer: SHAP TreeExplainer object
        - top_features: List of top N feature names by importance
    
    Saves:
    ------
    - results/plots/shap_summary.png: SHAP summary plot
    - results/plots/shap_bar.png: SHAP feature importance bar chart
    - results/plots/shap_dependence_<feature>.png: Dependence plots for top features
    - results/shap_feature_importance.csv: Feature importance values
    """
    logger.info("=" * 60)
    logger.info("SHAP EXPLAINABILITY ANALYSIS")
    logger.info("=" * 60)
    
    # Convert to numpy array if DataFrame
    if isinstance(X_sample, pd.DataFrame):
        X_array = X_sample.values
    else:
        X_array = X_sample
    
    logger.info(f"Computing SHAP values for {len(X_array):,} samples...")
    logger.info(f"Model type: {type(model).__name__}")
    
    # Create SHAP explainer (TreeExplainer for tree-based models)
    start_time = time.time()
    
    try:
        # Use TreeExplainer for Random Forest and tree-based models
        explainer = shap.TreeExplainer(model)
        logger.info("✓ Created SHAP TreeExplainer")
    except Exception as e:
        logger.warning(f"TreeExplainer failed, using KernelExplainer: {e}")
        # Fallback to KernelExplainer (slower but works for all models)
        background = shap.sample(X_array, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        logger.info("✓ Created SHAP KernelExplainer")
    
    # Calculate SHAP values
    logger.info("Calculating SHAP values (this may take a few minutes)...")
    shap_values = explainer.shap_values(X_array)
    
    elapsed = time.time() - start_time
    logger.info(f"✓ SHAP values computed in {elapsed:.2f}s")
    
    # For binary classification, use SHAP values for attack class (class 1)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_attack = shap_values[1]
        logger.info("✓ Using SHAP values for ATTACK class (class 1)")
    else:
        shap_values_attack = shap_values
    
    # ============================================================
    # 1. SHAP SUMMARY PLOT (Global Feature Importance)
    # ============================================================
    logger.info("\n1. Generating SHAP summary plot...")
    
    # Extract the correct SHAP values for plotting
    if len(shap_values_attack.shape) == 3:
        # For 3D array (n_samples, n_features, n_classes), use attack class (index 1)
        shap_plot_values = shap_values_attack[:, :, 1]
    else:
        shap_plot_values = shap_values_attack
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_plot_values, 
        X_array, 
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    plt.title('SHAP Summary Plot - Global Feature Importance (Attack Class)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    summary_path = PLOTS_DIR / 'shap_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"   ✓ Saved: {summary_path.name}")
    
    # ============================================================
    # 2. SHAP BAR PLOT (Mean Absolute SHAP Values)
    # ============================================================
    logger.info("2. Generating SHAP bar plot...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_plot_values,
        X_array,
        feature_names=feature_names,
        plot_type='bar',
        show=False,
        max_display=20
    )
    plt.title('SHAP Feature Importance (Mean |SHAP value|) - Attack Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    bar_path = PLOTS_DIR / 'shap_bar.png'
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"   ✓ Saved: {bar_path.name}")
    
    # ============================================================
    # 3. CALCULATE TOP FEATURES BY MEAN |SHAP|
    # ============================================================
    logger.info(f"\n3. Identifying top {top_n} features by SHAP importance...")
    
    # Calculate mean absolute SHAP values for each feature
    # SHAP values shape: (n_samples, n_features, n_classes) for binary classification
    # We want: mean over samples, then take attack class (class 1) values
    if len(shap_values_attack.shape) == 3:
        # Shape is (n_samples, n_features, n_classes)
        # Take attack class (index 1) and then mean over samples
        mean_shap_values = np.abs(shap_values_attack[:, :, 1]).mean(axis=0)
    else:
        # Shape is (n_samples, n_features)
        mean_shap_values = np.abs(shap_values_attack).mean(axis=0)
    
    logger.info(f"   Final mean SHAP shape: {mean_shap_values.shape}")
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Absolute_SHAP': mean_shap_values
    }).sort_values('Mean_Absolute_SHAP', ascending=False)
    
    top_features = feature_importance_df.head(top_n)['Feature'].tolist()
    
    logger.info(f"\n   Top {top_n} features:")
    for i, row in feature_importance_df.head(top_n).iterrows():
        logger.info(f"   {i+1:2d}. {row['Feature']:30s} | Mean |SHAP|: {row['Mean_Absolute_SHAP']:.6f}")
    
    # Save feature importance to CSV
    importance_csv = RESULTS_DIR / 'shap_feature_importance.csv'
    feature_importance_df.to_csv(importance_csv, index=False)
    logger.info(f"\n   ✓ Saved feature importance: {importance_csv.name}")
    
    # ============================================================
    # 4. SHAP DEPENDENCE PLOTS FOR TOP FEATURES
    # ============================================================
    # 4. SHAP DEPENDENCE PLOTS FOR TOP FEATURES
    # ============================================================
    logger.info(f"\n4. Generating SHAP dependence plots for top {min(5, top_n)} features...")
    
    for i, feature in enumerate(top_features[:5], 1):
        feature_idx = feature_names.index(feature)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_plot_values,  # Use the 2D array
            X_array,
            feature_names=feature_names,
            show=False
        )
        plt.title(f'SHAP Dependence Plot - {feature} (Attack Class)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Sanitize feature name for filename
        safe_feature_name = feature.replace(' ', '_').replace('/', '_').replace('\\', '_')
        dependence_path = PLOTS_DIR / f'shap_dependence_{safe_feature_name}.png'
        plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   {i}. Saved: {dependence_path.name}")
    
    # ============================================================
    # 5. SAMPLE-LEVEL EXPLANATIONS (FORCE PLOTS)
    # ============================================================
    logger.info("\n5. Generating sample-level explanations for attack predictions...")
    
    # Find indices of attack predictions
    if isinstance(X_sample, pd.DataFrame):
        predictions = model.predict(X_sample.values)
    else:
        predictions = model.predict(X_array)
    
    attack_indices = np.where(predictions == 1)[0]
    
    if len(attack_indices) > 0:
        # Select a few attack samples for explanation
        sample_count = min(3, len(attack_indices))
        selected_samples = attack_indices[:sample_count]
        
        logger.info(f"   Explaining {sample_count} attack predictions:")
        for idx in selected_samples:
            logger.info(f"\n   Sample {idx}:")
            logger.info(f"   Prediction: ATTACK (class 1)")
            
            # Get top 5 contributing features for this sample
            # Use the 2D SHAP values for attack class
            sample_shap = shap_plot_values[idx]
            top_contributors = np.argsort(np.abs(sample_shap))[-5:][::-1]
            
            logger.info("   Top 5 contributing features:")
            for rank, feat_idx in enumerate(top_contributors, 1):
                logger.info(f"      {rank}. {feature_names[feat_idx]:30s} | SHAP: {sample_shap[feat_idx]:+.6f}")
    else:
        logger.warning("   No attack predictions found in sample.")
    
    logger.info("\n" + "-" * 60 + "\n")
    
    return shap_plot_values, explainer, top_features


# ============================================================
# SECTION 2: REAL-TIME DETECTION
# ============================================================

def capture_live_packets(count=50, interface=None, timeout=30):
    """
    Capture live network packets using Scapy or simulate with random data.
    
    Parameters:
    -----------
    count : int, default=50
        Number of packets to capture
    interface : str, optional
        Network interface to capture on (e.g., 'eth0', 'Wi-Fi')
        If None, uses default interface
    timeout : int, default=30
        Maximum time to wait for packets (seconds)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with extracted packet features
    
    Notes:
    ------
    If Scapy is not available, generates simulated packet data for testing.
    """
    logger.info("=" * 60)
    logger.info("REAL-TIME PACKET CAPTURE")
    logger.info("=" * 60)
    
    if not SCAPY_AVAILABLE:
        logger.warning("⚠️  Scapy not available. Using SIMULATION MODE.")
        return simulate_live_packets(count)
    
    logger.info(f"Capturing {count} packets...")
    if interface:
        logger.info(f"Interface: {interface}")
    else:
        logger.info("Interface: Default")
    
    captured_packets = []
    
    def packet_callback(packet):
        """Process each captured packet."""
        if IP in packet:
            packet_info = {
                'timestamp': datetime.now(),
                'src_ip': packet[IP].src if IP in packet else None,
                'dst_ip': packet[IP].dst if IP in packet else None,
                'protocol': packet[IP].proto if IP in packet else None,
                'packet_length': len(packet),
                'ttl': packet[IP].ttl if IP in packet else None,
            }
            
            # TCP-specific features
            if TCP in packet:
                packet_info.update({
                    'src_port': packet[TCP].sport,
                    'dst_port': packet[TCP].dport,
                    'tcp_flags': packet[TCP].flags,
                    'tcp_window': packet[TCP].window,
                })
            elif UDP in packet:
                packet_info.update({
                    'src_port': packet[UDP].sport,
                    'dst_port': packet[UDP].dport,
                    'tcp_flags': 0,
                    'tcp_window': 0,
                })
            else:
                packet_info.update({
                    'src_port': 0,
                    'dst_port': 0,
                    'tcp_flags': 0,
                    'tcp_window': 0,
                })
            
            captured_packets.append(packet_info)
    
    try:
        # Capture packets
        sniff(
            iface=interface,
            prn=packet_callback,
            count=count,
            timeout=timeout,
            store=False
        )
        
        logger.info(f"✓ Captured {len(captured_packets)} packets")
        
        if len(captured_packets) == 0:
            logger.warning("No packets captured. Using simulation mode.")
            return simulate_live_packets(count)
        
        df = pd.DataFrame(captured_packets)
        logger.info(f"✓ Created DataFrame with shape: {df.shape}")
        logger.info("-" * 60 + "\n")
        
        return df
        
    except Exception as e:
        logger.error(f"Error capturing packets: {e}")
        logger.warning("Falling back to simulation mode.")
        return simulate_live_packets(count)


def simulate_live_packets(count=50):
    """
    Simulate live packet data for testing when Scapy is unavailable.
    
    Parameters:
    -----------
    count : int, default=50
        Number of simulated packets to generate
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with simulated packet features matching CICIDS2017 format
    """
    logger.info(f"Generating {count} simulated packets...")
    
    np.random.seed(int(time.time()) % 1000)
    
    # Generate realistic packet features based on CICIDS2017 dataset
    data = {
        'timestamp': [datetime.now() for _ in range(count)],
        'Flow_Duration': np.random.exponential(5000000, count),  # microseconds
        'Total_Fwd_Packets': np.random.poisson(10, count),
        'Total_Backward_Packets': np.random.poisson(8, count),
        'Total_Length_of_Fwd_Packets': np.random.exponential(1000, count),
        'Total_Length_of_Bwd_Packets': np.random.exponential(800, count),
        'Fwd_Packet_Length_Max': np.random.exponential(200, count),
        'Fwd_Packet_Length_Min': np.random.exponential(50, count),
        'Fwd_Packet_Length_Mean': np.random.exponential(100, count),
        'Fwd_Packet_Length_Std': np.random.exponential(50, count),
        'Bwd_Packet_Length_Max': np.random.exponential(200, count),
        'Bwd_Packet_Length_Min': np.random.exponential(50, count),
        'Bwd_Packet_Length_Mean': np.random.exponential(100, count),
        'Bwd_Packet_Length_Std': np.random.exponential(50, count),
        'Flow_Bytes/s': np.random.exponential(100000, count),
        'Flow_Packets/s': np.random.exponential(100, count),
        'Flow_IAT_Mean': np.random.exponential(10000, count),
        'Flow_IAT_Std': np.random.exponential(50000, count),
        'Flow_IAT_Max': np.random.exponential(100000, count),
        'Flow_IAT_Min': np.random.exponential(1000, count),
        'Fwd_IAT_Total': np.random.exponential(50000, count),
        'Fwd_IAT_Mean': np.random.exponential(10000, count),
        'Fwd_IAT_Std': np.random.exponential(20000, count),
        'Fwd_IAT_Max': np.random.exponential(50000, count),
        'Fwd_IAT_Min': np.random.exponential(1000, count),
        'Bwd_IAT_Total': np.random.exponential(50000, count),
        'Bwd_IAT_Mean': np.random.exponential(10000, count),
        'Bwd_IAT_Std': np.random.exponential(20000, count),
        'Bwd_IAT_Max': np.random.exponential(50000, count),
        'Bwd_IAT_Min': np.random.exponential(1000, count),
    }
    
    df = pd.DataFrame(data)
    
    # Add some attack-like patterns (30% probability)
    attack_mask = np.random.random(count) < 0.3
    df.loc[attack_mask, 'Flow_Packets/s'] *= 10  # High packet rate
    df.loc[attack_mask, 'Flow_IAT_Mean'] /= 10  # Low inter-arrival time
    
    logger.info(f"✓ Generated {count} simulated packets")
    logger.info(f"  ~{attack_mask.sum()} packets with attack-like patterns")
    logger.info(f"✓ DataFrame shape: {df.shape}")
    logger.info("-" * 60 + "\n")
    
    return df


def preprocess_live_data(df_live, selected_features):
    """
    Preprocess live packet data to match training data format.
    
    Parameters:
    -----------
    df_live : pd.DataFrame
        Raw live packet data
    selected_features : list
        List of feature names used during training
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed features ready for model prediction
    
    Notes:
    ------
    - Fills missing features with zeros
    - Ensures feature order matches training data
    - Handles missing values
    """
    logger.info("Preprocessing live packet data...")
    
    # Remove timestamp column if present
    if 'timestamp' in df_live.columns:
        timestamps = df_live['timestamp'].copy()
        df_live = df_live.drop(columns=['timestamp'])
    else:
        timestamps = None
    
    # Get available features
    available_features = set(df_live.columns)
    required_features = set(selected_features)
    
    # Add missing features with zeros
    missing_features = required_features - available_features
    if missing_features:
        logger.info(f"  Adding {len(missing_features)} missing features with zeros")
        for feat in missing_features:
            df_live[feat] = 0
    
    # Select only required features in correct order
    X_live = df_live[selected_features].copy()
    
    # Handle missing values
    if X_live.isnull().any().any():
        logger.warning("  Found missing values, filling with zeros")
        X_live = X_live.fillna(0)
    
    # Replace infinite values
    X_live = X_live.replace([np.inf, -np.inf], 0)
    
    logger.info(f"✓ Preprocessed data shape: {X_live.shape}")
    
    return X_live, timestamps


def classify_live_traffic(model, X_live, timestamps=None):
    """
    Classify live traffic using the trained model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained classification model
    X_live : pd.DataFrame or np.ndarray
        Preprocessed live traffic features
    timestamps : pd.Series, optional
        Timestamps for each packet
    
    Returns:
    --------
    pd.DataFrame
        Predictions with timestamps, labels, and confidence scores
    
    Saves:
    ------
    - Appends results to results/realtime_predictions.csv
    - Logs predictions to logs/realtime_detection.log
    """
    logger.info("=" * 60)
    logger.info("REAL-TIME TRAFFIC CLASSIFICATION")
    logger.info("=" * 60)
    
    # Make predictions
    logger.info(f"Classifying {len(X_live)} packets...")
    predictions = model.predict(X_live)
    probabilities = model.predict_proba(X_live)
    
    # Get confidence scores (probability of predicted class)
    confidence_scores = np.max(probabilities, axis=1)
    
    # Create results DataFrame
    if timestamps is not None:
        results_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Prediction': predictions,
            'Confidence': confidence_scores,
            'Attack_Probability': probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        })
    else:
        results_df = pd.DataFrame({
            'Timestamp': [datetime.now()] * len(predictions),
            'Prediction': predictions,
            'Confidence': confidence_scores,
            'Attack_Probability': probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        })
    
    # Add human-readable labels
    results_df['Label'] = results_df['Prediction'].map({0: 'BENIGN', 1: 'ATTACK'})
    
    # Calculate statistics
    benign_count = (predictions == 0).sum()
    attack_count = (predictions == 1).sum()
    
    logger.info(f"\n✓ Classification complete:")
    logger.info(f"  BENIGN: {benign_count:,} ({benign_count/len(predictions)*100:.2f}%)")
    logger.info(f"  ATTACK: {attack_count:,} ({attack_count/len(predictions)*100:.2f}%)")
    logger.info(f"  Average confidence: {confidence_scores.mean():.4f}")
    
    # Display real-time predictions in console
    logger.info("\n" + "=" * 60)
    logger.info("REAL-TIME PREDICTIONS (Sample)")
    logger.info("=" * 60)
    
    for i in range(min(10, len(results_df))):
        row = results_df.iloc[i]
        timestamp_str = row['Timestamp'].strftime('%H:%M:%S')
        label = row['Label']
        confidence = row['Confidence']
        
        # Color coding for terminal (if supported)
        if label == 'ATTACK':
            logger.info(f"[{timestamp_str}] Incoming packet → Predicted: {label} ({confidence:.2f} confidence)")
        else:
            logger.info(f"[{timestamp_str}] Incoming packet → Predicted: {label} ({confidence:.2f} confidence)")
    
    if len(results_df) > 10:
        logger.info(f"... ({len(results_df) - 10} more predictions)")
    
    logger.info("-" * 60)
    
    # Save to CSV (append mode)
    logger.info(f"\nSaving predictions to: {REALTIME_CSV}")
    
    if REALTIME_CSV.exists():
        # Append to existing file
        results_df.to_csv(REALTIME_CSV, mode='a', header=False, index=False)
        logger.info("✓ Appended predictions to existing CSV")
    else:
        # Create new file
        results_df.to_csv(REALTIME_CSV, index=False)
        logger.info("✓ Created new predictions CSV")
    
    # Log to real-time detection log
    with open(REALTIME_LOG, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Real-Time Detection Session: {datetime.now()}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total packets: {len(results_df)}\n")
        f.write(f"BENIGN: {benign_count} ({benign_count/len(predictions)*100:.2f}%)\n")
        f.write(f"ATTACK: {attack_count} ({attack_count/len(predictions)*100:.2f}%)\n")
        f.write(f"Average confidence: {confidence_scores.mean():.4f}\n\n")
        f.write("Sample predictions:\n")
        f.write(results_df.head(20).to_string())
        f.write("\n" + "="*60 + "\n")
    
    logger.info(f"✓ Logged session to: {REALTIME_LOG}")
    logger.info("-" * 60 + "\n")
    
    return results_df


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main execution pipeline for Explainability and Real-Time Detection.
    
    Workflow:
    ---------
    1. Load trained model and data
    2. Perform SHAP explainability analysis
    3. Capture/simulate live network traffic
    4. Classify live traffic in real-time
    5. Generate comprehensive reports and visualizations
    """
    start_time = time.time()
    
    try:
        logger.info("\n" + "=" * 60)
        logger.info("NIDS-ML: EXPLAINABILITY + REAL-TIME DETECTION")
        logger.info("=" * 60)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60 + "\n")
        
        # ============================================================
        # PART 1: EXPLAINABLE AI (SHAP)
        # ============================================================
        
        logger.info("PART 1: EXPLAINABLE AI WITH SHAP\n")
        
        # Load model and data
        model, X_sample, y_sample, feature_names = load_model_and_data()
        
        # Generate SHAP explanations
        shap_values, explainer, top_features = explain_model_with_shap(
            model, 
            X_sample, 
            feature_names,
            top_n=10
        )
        
        # ============================================================
        # PART 2: REAL-TIME DETECTION
        # ============================================================
        
        logger.info("PART 2: REAL-TIME TRAFFIC DETECTION\n")
        
        # Capture or simulate live packets
        df_live = capture_live_packets(count=CAPTURE_COUNT)
        
        # Preprocess live data
        X_live, timestamps = preprocess_live_data(df_live, feature_names)
        
        # Classify live traffic
        results_df = classify_live_traffic(model, X_live, timestamps)
        
        # ============================================================
        # SUMMARY
        # ============================================================
        
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
        logger.info(f"SHAP analysis: ✓ Complete")
        logger.info(f"Real-time detection: ✓ Complete")
        logger.info(f"Total packets classified: {len(results_df):,}")
        logger.info("=" * 60)
        
        logger.info("\n✓ EXPLAINABILITY + REAL-TIME DETECTION COMPLETED SUCCESSFULLY!")
        logger.info("\nOutput Files Generated:")
        logger.info("  1. results/plots/shap_summary.png")
        logger.info("  2. results/plots/shap_bar.png")
        logger.info("  3. results/plots/shap_dependence_*.png (top 5 features)")
        logger.info("  4. results/shap_feature_importance.csv")
        logger.info("  5. results/realtime_predictions.csv")
        logger.info("  6. logs/realtime_detection.log")
        logger.info("  7. logs/explainability_realtime.log")
        
        logger.info("\nNext Steps:")
        logger.info("  → Create Streamlit dashboard (dashboard.py)")
        logger.info("  → Finalize project documentation")
        logger.info("  → Generate architecture diagrams")
        logger.info("  → Prepare deployment guide")
        logger.info("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("ERROR OCCURRED DURING EXECUTION")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
