import numpy as np
import torch
import random, math
from . import worlds

def initialize_random_sphere(n_particles,
                                radius,
                                center=None,
                                min_dist=4,
                                random_speed=0,
                                spatial_dims=3,
                                mass=1):
    '''
        n_particles:    initial number of particles []
        radius:         initial radius of particle distribution [m]
    '''

    schema = worlds.schemas[str(spatial_dims) + 'd']

    if center == None:
        center = np.zeros((spatial_dims,))

    ## Set up initial conditions (ICs)
    world_state = torch.empty ( (n_particles, len(schema)) )

    # create a random distribution of particles
    for i in range(n_particles):

        try:
            iter(mass)
            m_i = mass[i] * random.random()
        except TypeError: m_i = mass * random.random()

        smallest_interparticle_distance = 0

        vs = []
        for j in range(spatial_dims):
            vs.append(2*(random.random()*0.5) * random_speed)
        
        pos = worlds.pos[spatial_dims]
        
        if spatial_dims == 1:
            while smallest_interparticle_distance < min_dist:
                r = 2*(random.random()-0.5) * radius
                candidate_b_1 = center[0] + r
                distances = np.abs(world_state[:i, 3] - candidate_b_1)
                if i > 0:
                    smallest_interparticle_distance = distances.min()
                else:
                    break
            
            world_state[i, :] = torch.tensor((0, i, m_i, candidate_b_1, vs[0]))

        elif spatial_dims > 1:
            while smallest_interparticle_distance < min_dist:
                test_pos = torch.zeros(spatial_dims)
                for j in range(spatial_dims):
                    test_pos[j] = center[j] + 2*(random.random()-0.5) * radius
                offset = world_state[:i, pos] - test_pos
                norms = np.linalg.norm(offset, axis=1)
                if i > 0:
                    smallest_interparticle_distance = norms.min()
                else:
                    break
            
            world_state[i, :] = torch.tensor((0, i, m_i, *test_pos, *vs))

    return world_state