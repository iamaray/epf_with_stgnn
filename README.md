# epf_with_stgnn

Code implementing electricity price forecasting with a spatial-temporal graph neural network.

# Model Architecture


# Roadmap
1. Implement simplified STGNN model on the data from three locations in the ERCOT market.
    - Graph Convolution module (GC)
    - Temporal Convolution module (TC)
    - Output module
    - Trainer class with `train()` function
3. Traint/test on the data.
4. Present simplified model/performance to Moyi.
5. Iterate model to level of complexity as presented in the MTGNN paper.
6. Train/test this on data and compare to simpler model.
7. Present these results to Moyi.
8. Implement the R-vine copula method of constructing adjacency matrix.

