# epf_with_stgnn

A spatio-temporal graph neural network (STGNN) for multivariate time series forecasting (MTF) applied specifically to the task of real-time market electricity price forecasting (RTM-EPF).

# Model Architecture

The basis of our model heavily follows the work of Wu et al. in _Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks_ and of Y. Yang et al. in _Forecasting day-ahead electricity prices with spatial dependence_. However, we have heavily modified the model under our own implemenation along with multiple additions, including two different learned adjacency matrix constructions.

# Roadmap

1. Implement further dynamic adjacency matrix constructions.
2. Streamline data processing code to more easily design experiments.
3. Implement a learnable auxiliary feature module.
