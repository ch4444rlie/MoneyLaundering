# Anti-Money Laundering (AML) Detection with Graph Neural Networks

## Project Overview
This project aims to develop an unsupervised anomaly detection system for identifying potential money laundering patterns in financial transaction networks, inspired by initiatives like Project Aurora. It leverages synthetic data to simulate realistic AML typologies (e.g., smurfing, money mules, complex layering) and employs a Graph Convolutional Network (GCN) to detect suspicious entities. The project is designed to avoid data leakage, ensuring features like `flagged_tx_count` are excluded, and focuses on graph-based features such as degree, transaction velocity, and round-tripping.

**Note**: This project is actively in development, with ongoing improvements to model performance, feature engineering, and anomaly score distribution.

## Project Goals
- **Unsupervised Detection**: Identify suspicious entities using a GCN trained on reconstruction errors, without relying on labeled data for training.
- **Realistic Synthetic Data**: Generate transaction networks with AML typologies (smurfing, money mules, complex layering) to mimic real-world financial systems.
- **Graph-Based Features**: Utilize network properties (e.g., degree, clustering coefficient, round-trip count) to capture complex patterns.
- **Federated Learning (FL) Simulation**: Incorporate institution-based local models to simulate collaborative learning across regions (US, EU, ASIA, HIGH_RISK).
- **Performance Metrics**: Optimize for precision, recall, F1, AUPRC, and AUC, targeting F1 > 0.3 and AUPRC > 0.5.

## Current Status
- **Data Generation**: Synthetic data is generated with 5000 entities and ~25000 transactions, including 1% suspicious entities exhibiting AML typologies (Cell 2). The Barabási-Albert graph is used, but exploration of Watts-Strogatz graphs is in progress for better cycle generation.
- **Feature Extraction**: Graph-based features (e.g., degree, in-degree, transaction velocity, high-value ratio, round-trip count) are computed using NetworkX (Cell 3), replacing Kuzu queries for efficiency.
- **Model**: A SimpleGCN with federated learning simulation is implemented (Cell 4). Current metrics show improvement (Average Precision: 0.0326, Recall: 0.3000, F1: 0.0575, AUPRC: 0.2445, AUC: 0.9765), but issues with anomaly score distribution (many true anomalies at zero, some at 2000) and ROC curve spiking indicate ongoing challenges.
- **Challenges**:
  - Sparse anomalies (1% suspicious entities) lead to unstable folds in cross-validation.
  - Anomaly score distribution shows poor separation, with many scores between 0–800.
  - AUPRC inconsistencies (e.g., 1.0000 in Fold 5) suggest numerical issues or insufficient positive samples.
- **Next Steps**:
  - Increase suspicious entity proportion to 2% for better training stability.
  - Enhance graph structure with Watts-Strogatz model for realistic cycles.
  - Add features like `cycle_score` to capture weighted transaction loops.
  - Tune GCN architecture (e.g., increase hidden layers to 256→64→16) and epochs to 500.
  - Normalize anomaly scores to avoid extreme values (0, 2000).

## Repository Structure
- **Cell 1**: Imports required libraries (pandas, NumPy, NetworkX, PyTorch, scikit-learn).
- **Cell 2**: Generates synthetic transaction data with AML typologies (smurfing, money mules, complex layering).
- **Cell 3**: Builds a directed graph and extracts features (degree, clustering coefficient, transaction velocity, etc.).
- **Cell 4**: Implements a SimpleGCN with federated learning simulation, trains the model, and evaluates performance using 5-fold cross-validation.

## Requirements
- Python 3.8+
- Libraries: `pandas`, `numpy`, `networkx`, `torch`, `scikit-learn`, `matplotlib`, `scipy`
- Install via: `pip install pandas numpy networkx torch scikit-learn matplotlib scipy`


## Current Metrics (In Progress)
- **Cross-Validation Results** (5 folds):
  - Average Precision: 0.0326 (± std TBD)
  - Average Recall: 0.3000 (± std TBD)
  - Average F1: 0.0575 (± std TBD)
  - Average AUPRC: 0.2445 (± std TBD)
  - Average AUC: 0.9765 (± std TBD)
- **Issues**: Poor anomaly score separation, unstable AUPRC, and early ROC curve spiking.
- **Goal**: Achieve F1 > 0.3 and AUPRC > 0.5 with improved score distribution.

