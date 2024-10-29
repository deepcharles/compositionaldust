# An Efficient Algorithm For Exact Segmentation of Large Compositional and Categorical Time Series

Python implementation of the algorithm described in **(Truong and Runge, 2024)**

- Truong, C. and Runge, V. (2024), An Efficient Algorithm for Exact Segmentation of Large Compositional and Categorical Time Series. Stat, 13: e70012. https://doi.org/10.1002/sta4.70012



## Python

To install the Python package, run in a terminal

```bash
python -m pip install git+https://github.com/deepcharles/compositionaldust.git
```

## Usage

Create a 3D signal with Dirichlet distributed components. 
The parameter of the Dirichlet distribution is piecewise constant.

```python
from compositionaldust import generate_signal

signal, bkps_true = generate_signal(n_samples=1_000, n_dims=3)

print(f"The true change-points are {bkps_true}.")
```

We can now use our method to estimate the true change-point positions.
The value of penalty controls the number of change-points that will be estimated: large values will detect few changes, and vice-versa.

```python
from compositionaldust import get_bkps

bkps_pred = get_bkps(signal=signal, penalty=1)

print(f"The predicted change-points are {bkps_pred}.")
```
