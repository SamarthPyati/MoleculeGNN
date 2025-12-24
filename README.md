# MoleculeGNN 🧬

Predicting molecular properties using Graph Neural Networks (GNNs). This project applies deep learning to chemistry by representing molecules as graphs and training neural networks to predict toxicity, solubility, and other properties.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

**Goal**: Build a Graph Neural Network that can predict molecular properties from chemical structure.

**Why GNNs?**: Traditional machine learning treats molecules as fixed-length feature vectors (fingerprints), losing structural information. GNNs learn directly from the molecular graph structure, capturing complex patterns like functional groups and ring systems.

## Architecture

- **Input**: Molecular structure (SMILES string)
- **Representation**: Graph (atoms = nodes, bonds = edges)
- **Model**: Graph Convolutional Network (GCN) 
- **Output**: Physical Property Prediction and Toxicity Prediction for biomolecules

## Installation

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Put raw CSV datasets in `data/raw/` (example: `data/raw/ESOL.csv`).

3. Run training:
   ```bash
   python main.py --dataset data/raw/ESOL.csv --model gcn --task regression
   ```

## Run the app

- Streamlit UI (recommended):
  From the project root run:
  ```bash
  streamlit run app.py --server.port 8501
  ```
  Then open http://localhost:8501 in your browser.

Default model path used by the UI: `molecule_gnn_final.pt`

## Notes
- Modify command-line args in `main.py` for batch size, epochs, splits, etc.
- Trained model saved to the path given by `--save-path`.

## License
This project is licensed under the MIT License - see the LICENSE file for details.