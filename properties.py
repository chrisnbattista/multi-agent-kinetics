





import numpy as np
from sfc.multi_agent_kinetics import forces, kernels






def pressure(state, particle_i, h):
    '''
    '''

    rho = 0

    for i in range(state.shape[0]):
        if i == particle_i: continue
        rho = rho + \
            state[i, 3] * kernels.quadratic_kernel(
                            np.linalg.norm(
                                state[i, 1:3],
                                state[particle_i, 1:3]
                            ),
                            h)
    
    return rho
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

