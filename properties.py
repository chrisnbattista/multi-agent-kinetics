





import numpy as np
import scipy
from . import forces, kernels



def density_all(state, h):
    '''
    Calculates the density at each particle given world state.
    '''

    # isolate particle position data
    coords = state[:,3:5]
    p_dists = scipy.spatial.distance.pdist(coords)
    return np.multiply(
        np.sum(
            scipy.spatial.distance.squareform(
                [kernels.cubic_spline(d, h=h) for d in p_dists]
            ),
            axis=1
        ),
        state[:,2]
    )

