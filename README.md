# MoleculeGNN 🧬

Predicting molecular properties using Graph Neural Networks (GNNs). This project applies deep learning to chemistry by representing molecules as graphs and training neural networks to predict toxicity, solubility, and other properties.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

**Goal**: Build a Graph Neural Network that can predict molecular properties from chemical structure.

**Why GNNs?**: Traditional machine learning treats molecules as fixed-length feature vectors (fingerprints), losing structural information. GNNs learn directly from the molecular graph structure, capturing complex patterns like functional groups and ring systems.

<!-- **Dataset**: Tox21 - 8,000 molecules labeled for 12 different toxicity endpoints -->

## Architecture

- **Input**: Molecular structure (SMILES string)
- **Representation**: Graph (atoms = nodes, bonds = edges)
- **Model**: Graph Convolutional Network (GCN) 
- **Output**: Physical Property Prediction and Toxicity Prediction for biomolecules


## License
This project is licensed under the MIT License - see the LICENSE file for details.