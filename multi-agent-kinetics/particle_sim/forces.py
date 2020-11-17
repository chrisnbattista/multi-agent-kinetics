import numpy as np
import scipy
from sklearn.preprocessing import normalize

def lennard_jones_potential(epsilon, omega, r):
    return (4 * epsilon * omega**12)/r**12 - (4 * epsilon * omega**6)/r**6

def pairwise_world_lennard_jones_potential(world, epsilon, omega):
    '''
    Not timestep dependent as it is not time dependent as it is a potential field.
    '''

    # isolate particle position data
    coords = world[:,1:3]

    # calculate pairwise potentials
    potentials = np.nan_to_num(
        scipy.spatial.distance.squareform(
            lennard_jones_potential(
                epsilon,
                omega,
                scipy.spatial.distance.pdist(coords)
            )
        )
    )

    # get absolute pairwise displacements
    size = coords.shape[0]
    differences = (coords[:, np.newaxis] - coords)
    differences = normalize(differences.reshape(-1, 2)).reshape(size, size, 2)

    # slow draft code
    # # normalize displacements using L2-norm
    # differences = np.nan_to_num(
    #     differences / \
    #         np.sqrt(
    #             np.sum(
    #                 np.power(
    #                     differences,
    #                     2
    #                 ),
    #                 axis=2
    #             )
    #         )[:, :, np.newaxis]
    # )

    # scale displacements by potentials
    answer = np.sum(differences * potentials[:, :, np.newaxis], axis=1)
    ##print(answer)
    return answer

# def slow_pairwise_world_lennard_jones_potential(world, epsilon, omega):
#     '''
#     '''
#
#     potential_matrix = np.zeros( (world.shape[0], 2) )
#     for i in range(world.shape[0]):
#         for j in range(world.shape[0]):
#             if i == j: continue
#             norm = np.linalg.norm(world.iloc[j][['b_1', 'b_2']] - world.iloc[i][['b_1', 'b_2']])
#             magnitude = lennard_jones_potential(epsilon, omega, norm)
#             direction = ((world.iloc[j] - world.iloc[i])/norm)[['b_1', 'b_2']]
#             potential_matrix[j] += direction * magnitude
#
#     return potential_matrix

def viscous_damping_force(world, c):
    '''
    F_damping = -cv
    '''

    return -c * world[:, 4:6]
