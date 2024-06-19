cimport cython
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

from compositionaldust.core.random cimport random_value, choice_i
from compositionaldust.core.cost_symbolic cimport CostSymbolic

np.import_array()

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef Py_ssize_t[::1] dust(float[:, ::1] signal, float penalty):
    
    cdef:
        CostSymbolic cost = CostSymbolic(signal)
        Py_ssize_t n_samples = cost.n_samples
        Py_ssize_t start, end, k_last_bkp, n_admissible_bkps, best_bkp, last_bkp, n_admissible_bkps_new, aux_bkp
        float[::1] soc_vec = np.zeros((n_samples+1,), dtype=np.float32)
        Py_ssize_t[::1] path_vec = np.empty((n_samples+1,), dtype=int)
        Py_ssize_t[::1] admissible_bkps = np.empty((n_samples+1,), dtype=np.intp)
        float current_cost, current_soc, min_soc, entropy, entropy_threshold
        bint to_prune
    
    path_vec[0] = 0
    soc_vec[0] = 0.0
    admissible_bkps[0] = 0
    n_admissible_bkps = 1
    best_bkp = -1


    for end from 1 <= end < n_samples + 1:
        min_soc = INFINITY
        for k_last_bkp from 0 <= k_last_bkp < n_admissible_bkps:
            last_bkp = admissible_bkps[k_last_bkp]
            current_cost = cost.evaluate(last_bkp, end)
            current_soc = soc_vec[last_bkp] + current_cost + penalty

            if current_soc < min_soc:
                min_soc = current_soc
                best_bkp = last_bkp
            
        soc_vec[end] = min_soc
        path_vec[end] = best_bkp
        
        # pruning
        n_admissible_bkps_new = 0
        for k_last_bkp from 0 <= k_last_bkp < n_admissible_bkps:
            to_prune = False
            last_bkp = admissible_bkps[k_last_bkp]
            current_cost = cost.evaluate(last_bkp, end)
            
            # Pelt pruning
            if soc_vec[last_bkp] + current_cost > min_soc:
                to_prune = True
            
            # dust pruning
            if (not to_prune) and (n_admissible_bkps_new > 1):
                # draw random pruning index
                aux_bkp = choice_i(admissible_bkps[:n_admissible_bkps_new])
                # compute mu_max
                mu_max = cost.compute_mu_max(start=aux_bkp, mid=last_bkp, end=end)
                if mu_max > 0:
                    mu = random_value() * mu_max
                    entropy = cost.compute_entropy(start=aux_bkp, mid=last_bkp, end=end, mu=mu)
                    entropy_threshold = soc_vec[end] - soc_vec[last_bkp] - mu * (soc_vec[last_bkp] - soc_vec[aux_bkp])
                    entropy_threshold /= <float>(end - last_bkp) - mu * <float>(last_bkp - aux_bkp)
                    if entropy > entropy_threshold:
                        to_prune = True
            
            if not to_prune:
                admissible_bkps[n_admissible_bkps_new] = last_bkp
                n_admissible_bkps_new += 1
        
        # add "end" to the set of admissible change-points
        admissible_bkps[n_admissible_bkps_new] = end
        n_admissible_bkps_new += 1
        n_admissible_bkps = n_admissible_bkps_new
    
    return path_vec
