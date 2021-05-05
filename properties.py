





import numpy as np
import scipy
from . import forces, kernels, worlds



def density_all(world, h):
    '''
    Calculates the density at each particle given world state.
    '''
    state = world.get_state()
    pos = worlds.pos[world.spatial_dims]

    # isolate particle position data
    coords = state[:,pos]
    k_vals = scipy.spatial.distance.squareform(
        [kernels.quadratic(d, h=h) for d in scipy.spatial.distance.pdist(coords)]
    )
    summed = np.sum(k_vals, axis=1)
    return np.multiply(summed, state[:,2])

