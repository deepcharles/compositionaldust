from itertools import cycle
from typing import List, Tuple

import numpy as np


def generate_signal(
    n_samples: int = 1_000, n_dims: int = 5, random_state=None
) -> Tuple[np.ndarray, List]:
    """
    Generate a signal array with Dirichlet distributed components.

    This function generates a signal array with Dirichlet distributed components. The
    number of samples in the output signal is specified by the `n_samples` parameter,
    and the number of dimensions in the output signal is specified by the `n_dims`
    parameter.

    Args:
        n_samples (int): Number of samples in the output signal. Default is 1000.
        n_dims (int): Number of dimensions in the output signal. Default is 5.
        random_state (Optional[np.random.RandomState]): Random state to be used for
            generating the numbers. If none is provided, a default random state will be
            used.

    Returns:
        Tuple[np.ndarray, List]: A tuple containing the generated signal array (a 2D
            numpy array with shape `(n_samples, n_dims)` containing Dirichlet
            distributed signals) and a list of breakpoints (`bkps`).
    """

    rng = np.random.default_rng(random_state)

    # Generate the number of breaks and the parameters for the Dirichlet distribution
    n_bkps = rng.integers(low=1, high=6, endpoint=True)
    alpha_even = 100 * rng.dirichlet(alpha=[1, 1])
    alpha_odd = 100 * rng.dirichlet(alpha=[1, 1])

    # Generate the list of segment lengths and the signal
    list_of_segment_lengths = [n_samples // (n_bkps + 1)] * (n_bkps + 1)
    list_of_segment_lengths[-1] += n_samples % (n_bkps + 1)
    signal = list()

    for segment_length, alpha in zip(
        list_of_segment_lengths, cycle([alpha_even, alpha_odd])
    ):
        sub_signal = rng.dirichlet(alpha=alpha, size=segment_length)
        signal.append(sub_signal)

    # Concatenate the segments and return the signal array
    signal = np.concatenate(signal)
    bkps = np.cumsum(list_of_segment_lengths).tolist()
    return signal, bkps
