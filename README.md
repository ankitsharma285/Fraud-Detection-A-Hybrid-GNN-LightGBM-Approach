# Fraud-Detection-A-Hybrid-GNN-LightGBM-Approach
This repository contains a high-performance fraud detection pipeline applied to the IEEE-CIS Fraud Detection dataset. The project explores the synergy between relational data (graphs) and tabular features to identify sophisticated fraudulent patterns.

# Project Overview

Fraud detection is often a battle between identifying individual suspicious transactions and uncovering organized fraud rings. This project addresses both by implementing a multi-model ensemble:

Graph Neural Networks (PyTorch Geometric): Captures the "relational" signal by modeling transactions, credit cards, and devices as nodes in a heterogeneous graph.

LightGBM: Captures the "statistical" signal using advanced feature engineering, group-based aggregations, and non-linear decision boundaries.

# Performance & Insights

LightGBM Baseline: Achieved a PR-AUC of 0.7606, proving the strength of tabular aggregates and tree-based splits.

GNN Integration: Provided a complementary signal (PR-AUC 0.56) that focuses on the connectivity and shared history of fraudulent entities.

The Hybrid Advantage: Combining these models allows for the detection of both "Attribute Fraud" (unusual amounts) and "Relational Fraud" (linked accounts). [Currently working on this feature]

# Repository Structure

main_gnn.py: The core GNN training and graph construction script.
lightGBM.py: LightGBM training with group-aggregation logic.
engine.py: Training and evaluation loops for the PyTorch models.
helper.py: Memory-efficient mapping and utility functions.
