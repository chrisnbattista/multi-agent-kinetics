





import numpy as np
import scipy
from . import forces, kernels



def density_all(state, h):
    '''
    Calculates the density at each particle given world state.
    '''

    # isolate particle position data
    coords = state[:,3:5]
    k_vals = scipy.spatial.distance.squareform([kernels.cubic_spline(d, h=h) for d in scipy.spatial.distance.pdist(coords)])
    summed = np.sum(k_vals, axis=1)
    return np.multiply(summed, state[:,2])

