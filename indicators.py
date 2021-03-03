import numpy as np
import numexpr as ne

def potential_energy(world, potentials):
    '''
    Calculates potential energy of a whole particle system.
    '''

    # Requires a list of all potentials, which are to return scalar
    # global values
    V = 0

    for potential in potentials:
        v = potential(world.get_state())
        V += v

    return V

def kinetic_energy(world):
    '''
    Calculates kinetic energy of a whole particle system.
    '''

    vel_mags = np.linalg.norm(world.get_state()[:,4:6], axis=1)
    T = np.sum( ((vel_mags**2) * world.get_state()[:,3] / 2),
                axis=0)

    return T

def hamiltonian(world, potentials=[]):
    '''
    Calculate Hamiltonian (kinetic and potential energy) of entire particle system.
    '''

    return \
        kinetic_energy(world.get_state()) + \
        potential_energy(world.get_state(), potentials)

def mse_trajectories(reference_trajectories, test_trajectories, n_particles):
    '''
    Calculates the aggregate mean squared error between two sets of particle
    trajectories, with the sets containing an arbitrary but corresponding
    number of particles.
    Input:
    n x m matrix, where n is number of entries, and m is number of spatial dims
    Output:
    scalar MSE based on row-wise norms
    '''

    differences = test_trajectories - reference_trajectories
    norms = np.linalg.norm(differences, axis=1)
    return ne.evaluate('sum(norms**2)') / n_particles

def total_sph_delta_v(world):
    '''
    Returns the total SPH-related delta_v stored in the world's accumulator and clears it.
    '''
    if not 'total_sph_delta_v' in world.scratch_material: return 0
    t = world.scratch_material['total_sph_delta_v']
    world.scratch_material['total_sph_delta_v'] = 0
    return t