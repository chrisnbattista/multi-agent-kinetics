import numpy as np
import scipy
import numexpr as ne
from sklearn.preprocessing import normalize
import itertools
from . import properties, kernels, worlds

def get_kernel_values(pairwise_differences, kernel_func, **kwargs):
    distances = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(
                pairwise_differences
            )
    )[:,i]
    return np.vectorize(lambda ri, **kwargs: kernel_func(ri,**kwargs))(distances)

def apply_interaction_kernel(world, kernel_func, agent_indexes=None, context=None, **kwargs):
    '''Applies the specified kernel pairwise between all agents, or between certain ones if agent_indexes is specified.
    
    Ref Lu et al 2018.'''

    state = world.get_state()
    pos = worlds.pos[world.spatial_dims]
    if agent_indexes == None:
        agent_indexes = slice(state.shape[0])

    pairwise_differences = state[agent_indexes, pos] - state[agent_indexes, pos, None]
    print(pairwise_differences)

    kernel_values = get_kernel_values(pairwise_differences, kernel_func)
    print(kernel_values)

    print((kernel_values * pairwise_differences).shape)

    return kernel_values * pairwise_differences

def gravity(world, context, attractor=None):
    '''Applies Newton's Law of gravity towards the point attractor.'''
    state = world.get_state()
    pos = state[:,worlds.pos[world.spatial_dims]]
    if attractor == None: attractor = np.zeros((world.spatial_dims,))
    diff = np.subtract(
            attractor,
            pos
        )
    # simple gravity:
    altitude = np.linalg.norm(diff, axis=1) - 6371000
    diff_normalized = normalize(diff)
    return diff_normalized \
            * properties.gravitational_constant(altitude=altitude)[:,None] \
            * state[:,worlds.mass[3],None]
    
    # too smart for own good gravity:
    # return (np.divide(
    #         1,
    #         (diff * np.linalg.norm(diff))
    #     )* earth_mass * 9.81 * state[:,worlds.mass[3], None])

def newtons_law_of_gravitation(world, G, context):
    '''Applies classical gravity between all particles in a world.'''
    ##G = (6.674*(10**(-11))) # true value
    state = world.get_state()
    mass = state[:,worlds.mass[world.spatial_dims]]
    position = state[:,worlds.pos[world.spatial_dims]]
    pairwise_distances = scipy.spatial.distance.pdist(position)
    squared_pairwise_distances = np.square(pairwise_distances)
    r_squared_mat = scipy.spatial.distance.squareform(squared_pairwise_distances)
    one_over_r_squared_mat = np.reciprocal(r_squared_mat)
    one_over_r_squared_mat = np.nan_to_num(one_over_r_squared_mat, posinf=0, neginf=0)
    m1_m2_over_r_squared_mat = one_over_r_squared_mat * mass * mass[...,None]
    F_G_mags = G * m1_m2_over_r_squared_mat
    forces = np.zeros((state.shape[0], world.spatial_dims))
    for i in range(state.shape[0]):
        for j in range(state.shape[0]):
            if i == j: continue
            forces[i,:] = forces[i,:] + normalize((position[j,:] - position[i,:]).reshape(1,-1)) * F_G_mags[i,j]
    return forces

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


def viscous_damping_force(world, c, context=None, **kwargs):
    '''
    F_damping = -cv
    '''

    state = world.get_state()
    vel = worlds.vel[world.spatial_dims]

    null_forces = np.zeros((world.n_agents,world.spatial_dims))
    return np.where(
        np.c_[world.context['damping_active']],
        -c * state[:, vel],
        null_forces
    )


def spline_attractor(world, lamb, target=None, h=1, context=None, **kwargs):

    state = world.get_state()
    pos = worlds.pos[kwargs['context']['spatial_dims']]

    null_forces = np.zeros((world.n_agents, kwargs['context']['spatial_dims']))

    r = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(state[:,pos])
        )[:,target]

    if target != None:
        F = np.where(
            np.c_[world.context['following_active']],
            lamb * np.vectorize(lambda r: kernels.cubic_spline_grad(r=r, h=h))(r)[:, None] * normalize((state[:, pos] - state[target, pos])),
            null_forces
        )
    else:
        return null_forces

    ##print(F)
    return F
 

def sum_world_gravity_potential(world, lamb, context=None, **kwargs):
    '''
    '''

    state = world.get_state()

    G = np.sum( np.abs( 0.5 * lamb * np.linalg.norm(state[:, 1:3], axis=1)**2 / state[:, 3] ) )
    ##print(G)
    return G


def world_pressure_force(world, reference_density, h=1, gamma=1, context=None):

    n_sph = sum(context['sph_active'])
    state = world.get_state()
    pos = worlds.pos[world.spatial_dims]
    mass = worlds.mass[world.spatial_dims]

    c = 1481 # m / s
    B = c**2 * reference_density
    particle_densities = properties.density_all(world, h)
    particle_pressures = B * ( (particle_densities / reference_density)**gamma - 1)
    particle_pressures = np.nan_to_num(
                                        particle_pressures,
                                        posinf=0,
                                        neginf=0
                                        )

    P_over_rho_sq = np.divide(particle_pressures, particle_densities**2)
    P_over_rho_sq = np.nan_to_num(
                                        P_over_rho_sq,
                                        posinf=0,
                                        neginf=0
                                        )
    
    F = np.zeros((world.n_agents, world.spatial_dims))
    for i in range(n_sph):
        term_1 = P_over_rho_sq[i]
        term_2 = P_over_rho_sq
        offsets = state[:, pos] - (state[i, pos])
        r = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(offsets)
        )[:,i]
        magnitudes = (-1) \
                    * (term_1 + term_2) \
                    * np.vectorize(lambda ri: kernels.quadratic_grad(ri,h))(r)
                    ##* state[:n_sph,mass] \ # not using this as we want F not dv_dt
        directions = normalize(offsets)
        F[i] = np.sum(magnitudes[:,None] * directions)

    return F

"""def pressure_force(i, world, pressure, h=1, context=None):
    '''
    Computes the pressure force for one particle from the world state
    i = particle index
    state = world state matrix
    returns: [force_x, force_y]
    '''

    state = world.get_state()

    pos = worlds.pos[context['spatial_dims']]
    coords = state[:,pos]
    if context['spatial_dims'] == 1:
        p_dists = [c[0] - c[1] for c in itertools.combinations(coords, 2)]
    else:
        p_dists = scipy.spatial.distance.pdist(coords)    

    dists_i = scipy.spatial.distance.squareform(
        p_dists
    )[:,i]
    k_vals = [kernels.quadratic_grad(d, h=h) for d in dists_i]

    if (context is not None) and ('sph_active' in context) and (not context['sph_active'][i]):
        return np.zeros((context['spatial_dims'],))
    if all(not k for k in k_vals): return np.zeros((context['spatial_dims'],))

    pairwise_force_mags = np.zeros(state.shape[0])

    densities = properties.density_all(world, h=h)
    own_density = densities[i]
    densities[i] = 0

    pressures = np.full(state.shape[0], pressure)

    # m * p / rho matrix
    pairwise_force_mags = \
        np.multiply((-1), state[:,2]**2) * k_vals
    pairwise_force_mags = \
        np.multiply(
            (pressures[i] / own_density**2) + (pressures / densities**2),
            pairwise_force_mags
        )
    pairwise_force_mags = np.ma.filled(pairwise_force_mags, 0)

    if context['spatial_dims'] > 1:
        directions = normalize((coords[:, np.newaxis] - coords)[i], axis=1)
        forces = np.sum(directions * pairwise_force_mags[:, None], axis=0)
    else:
        directions = 1 * np.sign((coords[:, np.newaxis] - coords)[i])
        forces = np.sum(
            np.multiply(
                pairwise_force_mags,
                directions
            )
        )
    return np.clip(forces, -10000, 10000)

def world_pressure_force(world, pressure, h=1, context=None):
    '''
    Apply the pressure force to all particles, with associated bookkeeping.

    Leverages get_agents_sph function to collect distributed 
    '''

    state = world.get_state()

    if not 'total_sph_delta_v' in world.scratch_material:
        world.scratch_material['total_sph_delta_v'] = 0

    force_accumulator = np.zeros((state.shape[0], world.spatial_dims))
    for i in range(len(world.control_agents)):
        ##ego_state = state.copy()
        ##print(world.control_agents[i].X_update)
        ##np.put(ego_state, [[i, i], [3,4]], world.control_agents[i].X_update[:-1])
        force_accumulator[i,:] += pressure_force(
                                                    i,
                                                    world,
                                                    pressure,
                                                    h=h,
                                                    context=context
                                                )
    
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
    
    return force_accumulator """

def swarm_leader_force(world, leader_force=np.zeros((2,)), context=None):
    '''
    for sph leader
    '''

    state = world.get_state()

    force_accumulator = np.zeros((state.shape[0], context['spatial_dims']))
    force_accumulator[context['swarm_leader'],:] += leader_force
    return force_accumulator