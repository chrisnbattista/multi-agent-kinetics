





import numpy as np
import scipy
import numexpr as ne
from sklearn.preprocessing import normalize
import itertools
from . import properties, kernels

def lennard_jones_potential(epsilon, sigma, r):
    return ne.evaluate('( (4 * epsilon * sigma**12)/r**12 - (4 * epsilon * sigma**6)/r**6 )')


def sum_world_lennard_jones_potential(world, epsilon, sigma):
    '''
    '''

    state = world.get_state()

    # isolate particle position data
    coords = state[:,1:3]

    # calculate pairwise potentials and distances
    p_dists = scipy.spatial.distance.pdist(coords)

    return np.sum(lennard_jones_potential(epsilon, sigma, p_dists))


def lennard_jones_force(epsilon, sigma, r, context=None):
    return ne.evaluate('24 * epsilon / r * ( (2)*(sigma/r)**12 - (sigma/r)**6 )')


def pairwise_world_lennard_jones_force(world, epsilon, sigma, **kwargs):
    '''
    '''

    state = world.get_state()

    # isolate particle position data
    coords = state[:,3:5]

    # calculate pairwise forces and distances
    p_dists = scipy.spatial.distance.pdist(coords)
    forces = scipy.spatial.distance.squareform(
            lennard_jones_force(epsilon, sigma, p_dists) / p_dists
    )

    # get absolute pairwise displacements
    ##size = coords.shape[0]
    differences = (coords[:, np.newaxis] - coords)
    ##differences = normalize(differences.reshape(-1, 2)).reshape(size, size, 2)

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
    expanded_forces = forces[:, :, np.newaxis]
    return ne.evaluate('sum(differences * expanded_forces, axis=1)')

# def slow_pairwise_world_lennard_jones_potential(world, epsilon, sigma):
#     '''
#     '''
#
#     potential_matrix = np.zeros( (world.shape[0], 2) )
#     for i in range(world.shape[0]):
#         for j in range(world.shape[0]):
#             if i == j: continue
#             norm = np.linalg.norm(world.iloc[j][['b_1', 'b_2']] - world.iloc[i][['b_1', 'b_2']])
#             magnitude = lennard_jones_potential(epsilon, sigma, norm)
#             direction = ((world.iloc[j] - world.iloc[i])/norm)[['b_1', 'b_2']]
#             potential_matrix[j] += direction * magnitude
#
#     return potential_matrix


def viscous_damping_force(world, c, **kwargs):
    '''
    F_damping = -cv
    '''

    state = world.get_state()

    if state.shape[1] == 7:
        null_forces = np.zeros((world.n_agents,2))
        return np.where(
            np.c_[world.context['damping_active']],
            -c * state[:, 5:7],
            null_forces
        )
        ##return -c * state[:, 5:7]
    else:
        return -c * state[:, 4][:, np.newaxis]


def linear_attractor(world, lamb, target=None, **kwargs):
    '''
    Exerts a constant gravitational force from the origin - assuming
    the ground level simplified form of constant acceleration gravity.
    '''

    state = world.get_state()

    if target != None:
        null_forces = np.zeros((world.n_agents,2))
        return np.where(
            np.c_[world.context['following_active']],
            (-lamb * (state[:, 3:5] - state[target, 3:5])),
            null_forces
        )
    else:
        if state.shape[1] == 7:
            return -lamb / state[:, 2, None] * state[:, 3:5]
        elif state.shape[1] == 5:
            return (-lamb / state[:, 2] * state[:, 3])[:, np.newaxis]


def sum_world_gravity_potential(world, lamb, **kwargs):
    '''
    '''

    state = world.get_state()

    G = np.sum( np.abs( 0.5 * lamb * np.linalg.norm(state[:, 1:3], axis=1)**2 / state[:, 3] ) )
    ##print(G)
    return G


def pressure_force(i, state, pressure, h=1, context=None):
    '''
    Computes the pressure force for one particle from the world state
    i = particle index
    state = world state matrix
    returns: [force_x, force_y]
    '''

    

    

    # isolate particle position data
    if state.shape[1] == 7:
        coords = state[:,3:5]
        spatial_dims = 2
        p_dists = scipy.spatial.distance.pdist(coords)
    else:
        coords = state[:,3]
        spatial_dims = 1
        ##print(list(itertools.combinations(coords, 2)))
        p_dists = [c[0] - c[1] for c in itertools.combinations(coords, 2)]
        ##print(p_dists)
    

    dists_i = scipy.spatial.distance.squareform(
        p_dists
    )[:,i]
    k_vals = [kernels.cubic_spline_grad(d, h=h) for d in dists_i]
    

    if not context['sph_active'][i]:
        ##print(f"skipping agent {i}")
        return np.zeros((spatial_dims,))
    if all(not k for k in k_vals): return np.zeros((spatial_dims,))

    
    pairwise_force_mags = np.zeros(state.shape[0])


    densities = properties.density_all(state, h=h)
    own_density = densities[i]
    densities[i] = 0
    densities = np.ma.masked_where(densities<0.0001, densities)

    ##print("DENS")
    ##print(densities)

    # https://lucasschuermann.com/writing/implementing-sph-in-2d
    pressures = np.full(state.shape[0], pressure)

    ##print(pressures)

    # m * p / rho matrix
    pairwise_force_mags = \
        np.multiply((-1), state[:,2]**2) * k_vals
    #print(pairwise_force_mags)
    pairwise_force_mags = \
        np.multiply(
            (pressures[i] / own_density**2) + (pressures / densities**2),
            pairwise_force_mags
        )
    pairwise_force_mags = np.ma.filled(pairwise_force_mags, 0)
    
    # pairwise_force_mags = \
    #     np.ma.filled(
    #         np.multiply(
    #             pairwise_force_mags,
    #             pressures[i] / np.ma.masked_equal(densities**2, 0)
    #         ),
    #         0
    #     )
    #print(pairwise_force_mags)
        
        
        # * (pressures / densities**2) + (pressures[i] / densities[i]**2),
       #     k_vals
      #  )

    if spatial_dims == 2:
        directions = normalize((coords[:, np.newaxis] - coords)[i], axis=1)
        ##print(pairwise_force_mags)
        forces = np.sum(directions * pairwise_force_mags[:, None], axis=0)
        ##print(forces)
    else:
        directions = 1 * np.sign((coords[:, np.newaxis] - coords)[i])
        forces = np.sum(
            np.multiply(
                pairwise_force_mags,
                directions
            )
        )
    return np.clip(forces, -1000, 1000)

def world_pressure_force(world, pressure, h=1, context=None):
    '''
    Apply the pressure force to all particles.
    '''

    state = world.get_state()

    if not 'total_sph_delta_v' in world.scratch_material:
        world.scratch_material['total_sph_delta_v'] = 0

    if state.shape[1] == 7: spatial_dims = 2
    else: spatial_dims = 1
    force_accumulator = np.zeros((state.shape[0], spatial_dims))
    for i in range(state.shape[0]):
        force_accumulator[i,:] += pressure_force(i, state, pressure, h=h, context=context)
    
    world.scratch_material['total_sph_delta_v'] += np.linalg.norm(force_accumulator/10)

    return force_accumulator

def viscosity_force(i, state, nu=0.0001, h=1, context=None):
    '''
    Computes the viscosity force for one particle from the world state
    i = particle index
    state = world state matrix
    '''

    densities = properties.density_all(state, h=h)

    # isolate particle position data
    coords = state[:,1:3]
    p_dists = scipy.spatial.distance.pdist(coords)
    dists_i = scipy.spatial.distance.squareform(
        p_dists
    )[:,i]

    # https://lucasschuermann.com/writing/implementing-sph-in-2d

    velocities = np.linalg.norm(state[:, 3:5], axis=1)

    pairwise_force_mags = np.nan_to_num(
        np.multiply(
            nu * \
            state[:,3] * \
            np.abs(velocities - velocities[i]) / densities,
            [kernels.cubic_spline_grad_double(d, h=h) for d in dists_i]
        )
    )

    differences = (coords[:, np.newaxis] - coords)[i]
    forces = differences * pairwise_force_mags[:, None]
    
    return np.sum(
        np.delete(
            forces,
            i
        )
    )

def world_viscosity_force(world, h=1, context=None):
    '''
    Apply the viscosity force to all particles.
    '''

    state = world.get_state()

    if state.shape[1] == 7: spatial_dims = 2
    else: spatial_dims = 1
    force_accumulator = np.zeros((state.shape[0], spatial_dims))
    for i in range(state.shape[0]):
        force_accumulator[i,:] += viscosity_force(i, state, h=h)
    
    return force_accumulator

def swarm_leader_force(world, leader_force=np.zeros((2,)), context=None):
    '''
    for sph leader
    '''

    state = world.get_state()

    force_accumulator = np.zeros((state.shape[0], context['spatial_dims']))
    force_accumulator[context['swarm_leader'],:] += leader_force
    return force_accumulator