# Data Processing

## Overview

The `data_processing` module implements all of the functionality to process raw input data to be used for model training and testing. At the highest level, the input given to the `data_processing` module is raw data in the form of a `.csv`, `.h5`, ..., and the output should be a `torch.data.data.DataLoader`/`torch_geometric.data.DataLoader` with batched train/test data. The rough sequence of steps taken by raw input data is as follows:

1.  **Pandas** is used to convert the raw data into a `DataFrame` to allow for all of the following steps.
2.  All desired pre-processing and transformations are applied to the data.
3.  Any needed pre-engineered feature extraction is performed.
4.  The data is sliced into **training/testing** pairs.
5.  The data is converted to a **PyTorch** `Dataset`.
6.  `torch_geometric.data.DataLoader` is used to batch the data.

## Implementation Details

In all of these files, please check the comments/docstrings for further details.

- `data_processing/transformation_funcs.py` implements all pre-processing/transformation functions to be applied to the data. As of now, these functions take in a `pandas.DataFrame` and modify its rows/columns as a side effect.

- `data_processing/processing_classes.py`, so far, implements `Operation`, `DataTransformation`, `ColumnCombination`, `OperationHistory`, and `OperationSequence`.

  - `Operation` is a class that wraps a transformation function and adds utilities to maintain an internal history as a list of separate `pandas.DataFrame`s.

  - `DataTransformation` is a particular `Operation` that performs in-place transformations across any number of columns.

  - `ColumnCombination` is an `Operation` that performs any computations that require two or more columns to be combined (added, multiplied, etc...).

  - `OperationHistory` acts as a dictionary to access the data at any state in a large sequence of `Operations`.

  - `OperationSequence` wraps a large sequence of operations and records them in an `OperationHistory`. This class is unfinished.

- `data_processing/processor.py` implements a class `PreprocessData` that completes some final steps in our preprocessing, specifically for our real-time market data, including dummy variable computation, possible quantile truncation, and feature extraction. This should probably be renamed to `RealTimeMarketPreprocess` or something and should be incorporated into the `./processing_classes.py` file.

- `data_processing/construct_pairs.py` implements a function `form_training_pairs()` that slices the data into training/testing pairs.

- `data_processing/dataset_constructors.py` implements utilities to output batched `DataLoader` objects in various forms. As of now, it does so in accordance with our _curriculum learning_ approach.

## TODO

1.  Bring code up-to-date with our Jupyter notebooks.
2.  Separate adjacency matrix construction functionality from `form_training_pairs()`.
3.  (Possibly) separate curriculum functionality from `DatasetConstructor`. Although, it may be better to just keep this how it is.
