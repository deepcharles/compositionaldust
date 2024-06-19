from typing import List

import numpy as np

from compositionaldust.core.dust import dust
from compositionaldust.utils import (from_path_matrix_to_bkps,
                                     one_hot_encode_signal)


def get_bkps(signal: np.ndarray, penalty: float) -> List[int]:
    if (signal.ndim == 1) or (signal.shape[1] == 1):
        signal = one_hot_encode_signal(signal)

    signal = np.require(
        signal,
        dtype=np.float32,
        requirements=["C_CONTIGUOUS", "WRITEABLE", "OWNDATA"],
    )
    path_vec = dust(signal=signal, penalty=penalty)
    return from_path_matrix_to_bkps(path_vec=path_vec)
