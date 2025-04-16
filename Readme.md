# Kubernetes Crash Prediction and Remediation System

An AI/ML-powered system for predicting and automatically remediating issues in Kubernetes clusters.

## Overview

This project uses machine learning to predict potential issues in Kubernetes clusters before they cause service disruptions. It monitors cluster metrics, predicts problems like pod crashes, resource exhaustion, and node failures, then automatically applies remediation actions to prevent or mitigate these issues.

Key features:

- **Proactive Monitoring**: Continuously collects metrics from Kubernetes clusters
- **ML-Based Prediction**: Uses multiple models to predict different types of issues
- **Automated Remediation**: Applies fixes like pod restarts, resource limit adjustments, and node cordoning
- **Configurable**: Flexible configuration for different environments and requirements
- **Visualization**: Visualizes metrics and predictions for better understanding

## Architecture

The system consists of several components:

1. **Data Collection**: Gathers metrics from Kubernetes API and Prometheus
2. **Feature Engineering**: Processes raw metrics for machine learning
3. **Prediction Models**:
   - Resource usage prediction (CPU, memory, disk, network)
   - Pod crash prediction
   - Node failure prediction
   - Network issue prediction
4. **Remediation Engine**:
   - Decision engine to select appropriate actions
   - Action library with Kubernetes API integration
5. **Driver Program**: Orchestrates the entire process

## Installation

### Prerequisites

- Python 3.9+
- Kubernetes cluster with API access
- Prometheus (optional but recommended)

### Install using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/k8s-predictor.git
cd k8s-predictor

# Install dependencies
pip install -r requirements.txt