import numpy as np
import scipy

from hts.multi_agent_kinetics import kernels

def density_all(state):
    '''
    Calculates the density for a particle given world state.
    '''

    # isolate particle position data
    coords = world[:,1:3]
    p_dists = scipy.spatial.distance.pdist(coords)
    kernel = scipy.spatial.distance.squareform(
        kernels.b_spline(q=p_dists)
    )
    kernel *= 10 # HARD CODED MASS

    return np.sum(kernel, axis=1)

