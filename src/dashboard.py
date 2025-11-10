"""
Interactive Dashboard Module for NIDS-ML

This module handles:
- Streamlit-based interactive web dashboard
- Real-time detection monitoring and visualization
- Model performance metrics display
- SHAP explainability visualization
- Live traffic statistics and alerts
- Historical intrusion analysis
- Model comparison interface

Author: [Your Name]
Date: November 10, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime, timedelta
import time
import logging

# Import custom modules
# from realtime_detection import RealtimeDetector
# from shap_explainability import SHAPExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NIDS-ML Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-danger {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-success {
        background-color: #00cc00;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class NIDSDashboard:
    """
    Main dashboard class for NIDS visualization and monitoring.
    """
    
    def __init__(self):
        """Initialize dashboard."""
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'detector' not in st.session_state:
            st.session_state.detector = None
        if 'detection_data' not in st.session_state:
            st.session_state.detection_data = []
        if 'is_monitoring' not in st.session_state:
            st.session_state.is_monitoring = False
    
    def render_sidebar(self):
        """Render sidebar with navigation and controls."""
        st.sidebar.title("🛡️ NIDS-ML Control Panel")
        
        # Navigation
        page = st.sidebar.radio(
            "Navigation",
            ["🏠 Home", "📊 Model Performance", "🔴 Live Detection", 
             "🧠 Explainability", "📈 Analytics", "⚙️ Settings"]
        )
        
        st.sidebar.markdown("---")
        
        # Model selection
        st.sidebar.subheader("Model Selection")
        model_options = ["Random Forest", "XGBoost", "SVM", "Deep Learning"]
        selected_model = st.sidebar.selectbox("Choose Model", model_options)
        
        st.sidebar.markdown("---")
        
        # System stats
        st.sidebar.subheader("System Status")
        st.sidebar.metric("Status", "🟢 Active" if st.session_state.is_monitoring else "🔴 Inactive")
        
        return page, selected_model
    
    def render_home(self):
        """Render home page with overview."""
        st.markdown('<h1 class="main-header">🛡️ Network Intrusion Detection System</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        ### Welcome to NIDS-ML Dashboard
        
        This intelligent system uses **Machine Learning** and **Deep Learning** to detect network intrusions in real-time.
        
        **Key Features:**
        - 🎯 Multi-model ensemble detection
        - 🔍 Real-time packet analysis
        - 🧠 Explainable AI with SHAP
        - 📊 Comprehensive analytics
        - ⚡ Live monitoring dashboard
        """)
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Packets", "12,456", "+234")
        with col2:
            st.metric("Intrusions Detected", "23", "+3")
        with col3:
            st.metric("Model Accuracy", "98.5%", "+0.2%")
        with col4:
            st.metric("Detection Rate", "99.1%", "+0.5%")
        
        # TODO: Add recent activity chart
        st.subheader("Recent Activity")
        # self.plot_recent_activity()
    
    def render_model_performance(self):
        """Render model performance comparison page."""
        st.header("📊 Model Performance Analysis")
        
        # TODO: Load actual model results
        st.subheader("Model Comparison")
        
        # Sample data - replace with actual results
        model_data = {
            'Model': ['Random Forest', 'XGBoost', 'SVM', 'KNN', 'Logistic Regression'],
            'Accuracy': [98.5, 98.2, 96.8, 95.3, 94.1],
            'Precision': [98.3, 98.0, 96.5, 95.0, 93.8],
            'Recall': [98.7, 98.4, 97.0, 95.5, 94.3],
            'F1-Score': [98.5, 98.2, 96.7, 95.2, 94.0]
        }
        
        df = pd.DataFrame(model_data)
        
        # Display table
        st.dataframe(df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']))
        
        # Visualization
        fig = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for metric in metrics:
            fig.add_trace(go.Bar(name=metric, x=df['Model'], y=df[metric]))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score (%)",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Select a model to view its confusion matrix")
            # TODO: Implement confusion matrix visualization
    
    def render_live_detection(self):
        """Render live detection monitoring page."""
        st.header("🔴 Live Network Monitoring")
        
        # Control buttons
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button("▶️ Start Monitoring", disabled=st.session_state.is_monitoring):
                st.session_state.is_monitoring = True
                st.success("Monitoring started!")
        
        with col2:
            if st.button("⏹️ Stop Monitoring", disabled=not st.session_state.is_monitoring):
                st.session_state.is_monitoring = False
                st.warning("Monitoring stopped!")
        
        # Live statistics
        st.subheader("Live Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Packets/sec", "245")
        with col2:
            st.metric("Intrusions/min", "0.3")
        with col3:
            st.metric("Avg Latency", "2.5ms")
        with col4:
            st.metric("CPU Usage", "34%")
        
        # Recent detections table
        st.subheader("Recent Detections")
        
        # TODO: Display real detection data
        sample_detections = pd.DataFrame({
            'Timestamp': [datetime.now() - timedelta(minutes=i) for i in range(10)],
            'Source IP': [f'192.168.1.{i}' for i in range(10)],
            'Dest IP': [f'10.0.0.{i}' for i in range(10)],
            'Type': ['BENIGN'] * 8 + ['DDoS', 'PortScan'],
            'Confidence': np.random.uniform(0.85, 0.99, 10)
        })
        
        st.dataframe(sample_detections, use_container_width=True)
        
        # Live chart placeholder
        st.subheader("Traffic Over Time")
        chart_placeholder = st.empty()
        
        # TODO: Implement real-time chart updates
    
    def render_explainability(self):
        """Render SHAP explainability page."""
        st.header("🧠 Model Explainability (SHAP)")
        
        st.markdown("""
        Understanding **why** the model makes certain predictions is crucial for trust and debugging.
        We use **SHAP (SHapley Additive exPlanations)** to provide interpretable insights.
        """)
        
        # Feature importance
        st.subheader("Global Feature Importance")
        st.info("Upload a SHAP summary plot or generate one from trained models")
        
        # TODO: Display actual SHAP plots
        # st.image('../logs/shap_summary.png')
        
        # Individual prediction explanation
        st.subheader("Explain Individual Prediction")
        
        prediction_id = st.number_input("Enter Prediction ID", min_value=0, max_value=1000, value=0)
        
        if st.button("Generate Explanation"):
            st.info(f"Generating SHAP explanation for prediction #{prediction_id}...")
            # TODO: Generate and display waterfall/force plot
    
    def render_analytics(self):
        """Render analytics and historical data page."""
        st.header("📈 Historical Analytics")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Attack type distribution
        st.subheader("Attack Type Distribution")
        
        # Sample data
        attack_types = pd.DataFrame({
            'Attack Type': ['DDoS', 'PortScan', 'Brute Force', 'Web Attack', 'Infiltration'],
            'Count': [450, 320, 180, 95, 25]
        })
        
        fig = px.pie(attack_types, values='Count', names='Attack Type', 
                    title='Attack Distribution (Last 7 Days)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series
        st.subheader("Intrusion Trends")
        # TODO: Add time series visualization
    
    def render_settings(self):
        """Render settings page."""
        st.header("⚙️ Settings")
        
        st.subheader("Model Configuration")
        
        threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.85, 0.05)
        st.write(f"Predictions with confidence < {threshold} will be flagged for review")
        
        st.subheader("Network Interface")
        interface = st.text_input("Network Interface", "eth0")
        
        st.subheader("Alert Settings")
        email_alerts = st.checkbox("Enable Email Alerts")
        if email_alerts:
            email = st.text_input("Alert Email")
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")
    
    def run(self):
        """Main dashboard run method."""
        page, selected_model = self.render_sidebar()
        
        # Route to appropriate page
        if page == "🏠 Home":
            self.render_home()
        elif page == "📊 Model Performance":
            self.render_model_performance()
        elif page == "🔴 Live Detection":
            self.render_live_detection()
        elif page == "🧠 Explainability":
            self.render_explainability()
        elif page == "📈 Analytics":
            self.render_analytics()
        elif page == "⚙️ Settings":
            self.render_settings()


def main():
    """Main entry point for dashboard."""
    dashboard = NIDSDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
