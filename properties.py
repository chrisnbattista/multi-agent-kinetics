





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