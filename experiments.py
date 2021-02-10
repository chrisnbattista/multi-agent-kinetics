





import numpy as np
import random, math
from . import worlds





def initialize_random_circle(n_particles, radius, center=(0,0), min_dist=4, random_speed=0, spatial_dims=2):
    '''
        n_particles:    initial number of particles []
        radius:         initial radius of particle distribution [m]
    '''

    schema = worlds.schemas[str(spatial_dims) + 'd']

    ## Set up initial conditions (ICs)
    world_state = np.empty ( (n_particles, len(schema)) )

    # create a random distribution of particles
    for i in range(n_particles):

        smallest_interparticle_distance = 0

        vs = []
        for j in range(spatial_dims):
            vs.append(random.random() * random_speed)
        
        if spatial_dims == 1:
            while smallest_interparticle_distance < min_dist:
                r = random.random() * radius
                candidate_b_1 = center[0] + r
                distances = np.abs(world_state[:i, 3] - candidate_b_1)
                if i > 0:
                    smallest_interparticle_distance = distances.min()
                else:
                    break
            
            world_state[i, :] = (0, i, 10, candidate_b_1, vs[0])

        elif spatial_dims == 2:
            while smallest_interparticle_distance < min_dist:
                theta = random.random() * 2 * math.pi
                r = random.random() * radius
                candidate_b_1 = center[0] + r * math.cos(theta)
                candidate_b_2 = center[1] + r * math.sin(theta)
                test_df = world_state[:i, 3:5] - [candidate_b_1, candidate_b_2]
                norms = np.linalg.norm( test_df, axis=1 )
                if i > 0:
                    smallest_interparticle_distance = norms.min()
                else:
                    break

            world_state[i, :] = (0, i, 10, candidate_b_1, candidate_b_2, vs[0], vs[1])

    return world_state