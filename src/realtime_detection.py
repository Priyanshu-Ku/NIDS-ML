"""
Real-Time Detection Module for NIDS-ML

This module handles:
- Live packet capture using Scapy
- Feature extraction from network packets in real-time
- Real-time classification using trained ML models
- Logging detected intrusions with timestamps
- Alert generation for suspicious traffic
- Integration with dashboard for live monitoring

Author: [Your Name]
Date: November 10, 2025
"""

import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
import joblib
import logging
from datetime import datetime
from collections import deque
import threading
import time
from typing import Callable, Optional, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/realtime_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PacketFeatureExtractor:
    """
    Extracts relevant features from network packets for ML classification.
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.packet_history = deque(maxlen=1000)  # Store recent packets for flow-based features
        
    def extract_features(self, packet) -> Dict:
        """
        Extract features from a single packet.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            dict: Extracted features
        """
        features = {}
        
        try:
            # Basic packet info
            features['timestamp'] = datetime.now()
            features['packet_size'] = len(packet)
            
            # IP layer features
            if IP in packet:
                features['src_ip'] = packet[IP].src
                features['dst_ip'] = packet[IP].dst
                features['protocol'] = packet[IP].proto
                features['ttl'] = packet[IP].ttl
                features['ip_len'] = packet[IP].len
                features['ip_flags'] = packet[IP].flags
            
            # TCP layer features
            if TCP in packet:
                features['src_port'] = packet[TCP].sport
                features['dst_port'] = packet[TCP].dport
                features['tcp_flags'] = packet[TCP].flags
                features['tcp_window'] = packet[TCP].window
                features['tcp_dataofs'] = packet[TCP].dataofs
            
            # UDP layer features
            elif UDP in packet:
                features['src_port'] = packet[UDP].sport
                features['dst_port'] = packet[UDP].dport
                features['udp_len'] = packet[UDP].len
            
            # ICMP features
            if ICMP in packet:
                features['icmp_type'] = packet[ICMP].type
                features['icmp_code'] = packet[ICMP].code
            
            # Payload features
            if Raw in packet:
                features['payload_size'] = len(packet[Raw].load)
            else:
                features['payload_size'] = 0
            
            # TODO: Add flow-based features (packets per second, bytes per second, etc.)
            # TODO: Add statistical features computed from packet history
            
            logger.debug(f"Extracted {len(features)} features from packet")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    def create_feature_vector(self, packet_features: Dict) -> np.ndarray:
        """
        Convert packet features to ML-ready feature vector.
        
        Args:
            packet_features (dict): Raw packet features
            
        Returns:
            np.ndarray: Feature vector matching training data format
        """
        # TODO: Implement feature vector creation matching training format
        pass


class RealtimeDetector:
    """
    Performs real-time network intrusion detection on live traffic.
    """
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Initialize real-time detector.
        
        Args:
            model_path (str): Path to trained ML model
            scaler_path (str): Path to fitted scaler (optional)
        """
        self.model = None
        self.scaler = None
        self.feature_extractor = PacketFeatureExtractor()
        self.detection_queue = deque(maxlen=100)  # Store recent detections
        self.is_running = False
        self.packet_count = 0
        self.intrusion_count = 0
        
        self.load_model(model_path)
        if scaler_path:
            self.load_scaler(scaler_path)
        
    def load_model(self, model_path: str):
        """
        Load trained ML model.
        
        Args:
            model_path (str): Path to model file
        """
        logger.info(f"Loading model from {model_path}")
        try:
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_scaler(self, scaler_path: str):
        """
        Load fitted scaler.
        
        Args:
            scaler_path (str): Path to scaler file
        """
        logger.info(f"Loading scaler from {scaler_path}")
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
    
    def process_packet(self, packet):
        """
        Process a single packet and detect intrusion.
        
        Args:
            packet: Scapy packet object
        """
        try:
            self.packet_count += 1
            
            # Extract features
            features = self.feature_extractor.extract_features(packet)
            
            if not features:
                return
            
            # Convert to feature vector
            feature_vector = self.feature_extractor.create_feature_vector(features)
            
            # Scale features if scaler is available
            if self.scaler:
                feature_vector = self.scaler.transform([feature_vector])
            
            # Predict
            prediction = self.model.predict(feature_vector)[0]
            prediction_proba = self.model.predict_proba(feature_vector)[0] if hasattr(self.model, 'predict_proba') else None
            
            # Create detection record
            detection = {
                'timestamp': features.get('timestamp', datetime.now()),
                'src_ip': features.get('src_ip', 'N/A'),
                'dst_ip': features.get('dst_ip', 'N/A'),
                'src_port': features.get('src_port', 'N/A'),
                'dst_port': features.get('dst_port', 'N/A'),
                'protocol': features.get('protocol', 'N/A'),
                'prediction': prediction,
                'confidence': max(prediction_proba) if prediction_proba is not None else None,
                'packet_size': features.get('packet_size', 0)
            }
            
            # Log intrusion if detected
            if prediction != 'BENIGN' and prediction != 0:  # Adjust based on your label encoding
                self.intrusion_count += 1
                self.log_intrusion(detection)
                logger.warning(f"INTRUSION DETECTED: {detection}")
            
            # Add to detection queue
            self.detection_queue.append(detection)
            
            # Log every 100 packets
            if self.packet_count % 100 == 0:
                logger.info(f"Processed {self.packet_count} packets, {self.intrusion_count} intrusions detected")
                
        except Exception as e:
            logger.error(f"Error processing packet: {str(e)}")
    
    def log_intrusion(self, detection: Dict):
        """
        Log detected intrusion to file.
        
        Args:
            detection (dict): Detection information
        """
        # TODO: Implement intrusion logging to file
        pass
    
    def start_capture(self, interface: Optional[str] = None, filter_rule: Optional[str] = None):
        """
        Start capturing and analyzing live packets.
        
        Args:
            interface (str): Network interface to capture from (None = all)
            filter_rule (str): BPF filter rule (e.g., 'tcp port 80')
        """
        logger.info(f"Starting packet capture on interface: {interface or 'all'}")
        logger.info(f"Filter: {filter_rule or 'none'}")
        
        self.is_running = True
        
        try:
            # Start packet sniffing
            sniff(
                iface=interface,
                filter=filter_rule,
                prn=self.process_packet,
                store=False,
                stop_filter=lambda x: not self.is_running
            )
        except Exception as e:
            logger.error(f"Error during packet capture: {str(e)}")
            self.is_running = False
    
    def stop_capture(self):
        """Stop packet capture."""
        logger.info("Stopping packet capture")
        self.is_running = False
    
    def get_statistics(self) -> Dict:
        """
        Get detection statistics.
        
        Returns:
            dict: Statistics about detection performance
        """
        return {
            'total_packets': self.packet_count,
            'total_intrusions': self.intrusion_count,
            'intrusion_rate': self.intrusion_count / self.packet_count if self.packet_count > 0 else 0,
            'recent_detections': list(self.detection_queue)[-10:]  # Last 10 detections
        }
    
    def get_recent_detections(self, n: int = 10) -> List[Dict]:
        """
        Get N most recent detections.
        
        Args:
            n (int): Number of recent detections to return
            
        Returns:
            list: Recent detections
        """
        return list(self.detection_queue)[-n:]


def run_realtime_detection(model_path: str, interface: Optional[str] = None, 
                          filter_rule: Optional[str] = None):
    """
    Main function to run real-time detection.
    
    Args:
        model_path (str): Path to trained model
        interface (str): Network interface
        filter_rule (str): Packet filter
    """
    detector = RealtimeDetector(model_path=model_path)
    
    try:
        detector.start_capture(interface=interface, filter_rule=filter_rule)
    except KeyboardInterrupt:
        logger.info("Detection stopped by user")
        detector.stop_capture()
        stats = detector.get_statistics()
        logger.info(f"Final Statistics: {stats}")


if __name__ == "__main__":
    # Example usage
    # Run with: python realtime_detection.py
    # Requires administrator/root privileges for packet capture
    
    model_path = '../models/random_forest_model.pkl'
    # run_realtime_detection(model_path, interface='eth0', filter_rule='tcp')
    pass
