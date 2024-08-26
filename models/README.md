# Models

## Overview

The `models` module implements all of our forecasting models and their auxiliary functionalities -- e.g., adjacency matrices for GNNs, learnable feature extraction, etc. -- as well as all training algorithms. The **input** to this module will be either PyTorch `torch.data.Data` (or `torch_geometric.data.Data`) or `torch.Tensor` data, and the **output** will be a `torch.Tensor`. Some guidelines for model implementations:

- Make sure DL/ML models are implemented in **PyTorch**.
- Make sure the model input and output are standardized in accordance with the desired training algorithm. If not, either modify the model or implement a new training algorithm/modify an existing one.
- Cite any papers you have drawn from/replicated to implement the model.
- All analogous and equivalent rules for implementing training algorithms.

## Implementation Details

- `./adjacency_constructions` implements all adjacency matrices to be used in GNNs.
- `./stgnn` implements our STGNN (spatio-temporal GNN). Consult commments/docstrings for further details.
- `./trainers` implements all of our training algorithms.
  - `./trainers/classic_trainer.py` vanilla training algorithm.
  - `./trainers/curriculum_trainer.py` prediction-length curriculum trainer.

## TODO

1.  Bring code up-to-date with our Jupyter notebooks; in particular, bring over CATS code in a separate `./feature_selection` sub-module.
