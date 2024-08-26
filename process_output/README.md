# Output Processing

## Overview

The `process_output` implements all functionality to process, analyze, and visualize model output. The **inputs** to this module are forecasting model outputs along with actuals. The **outputs** are performance metrics and visualizations such as network diagrams and plots. In general, the following steps are taken:

1.  Model outputs are taken in along iwth the original unnormalized data.
2.  If the model was trained on normalized data, the necessary reverse normalizations are applied to the model outputs.
3.  Unnormalized performance metrics are computed between the forecasts and actuals.
4.  All plots/visualizations are created on the unnormalized data.

## Implementation Details

- `reverse_norm.py` takes normalized forecasts and passes them through inverse normalizations in the proper order.
- `./plotting.py` produces all plots and visualizations.

## TODO

1. Bring code up-to-date with our Jupyter notebooks.
2. Implement unnormalization pipeline to trivialize this procedure.
