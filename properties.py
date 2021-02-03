





import numpy as np
import scipy
from . import forces, kernels






def pressure(state, particle_i, h):
    '''
    '''

    rho = 0

    for i in range(state.shape[0]):
        if i == particle_i: continue
        rho = rho + state[:,3] * kernels.quadratic_kernel(
                                    np.linalg.norm(
                                        state[i, 1:3],
                                        state[particle_i, 1:3]
                                    ),
                                    h
                                )
    
    return rho


def density_all(state, h):
    '''
    Calculates the density at each particle given world state.
    '''

    # isolate particle position data
    coords = state[:,1:3]
    p_dists = scipy.spatial.distance.pdist(coords)
    return np.multiply(
        np.sum(
            scipy.spatial.distance.squareform(
                [kernels.cubic_spline(d, h=h) for d in p_dists]
            ),
            axis=1
        ),
        state[:,3]
    )

