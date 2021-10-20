import numpy as np
import numexpr as ne
from . import potentials

def hamiltonian(world, global_potentials=[]):
    '''
    Calculate Hamiltonian (kinetic and potential energy) of entire particle system.
    '''

    return \
        potentials.kinetic_energy(world) + \
        potentials.potential_energy(world, global_potentials)

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

def mse_hamiltonians(world_1, world_2):
    '''Computes the mean squared loss over the Hamiltonian of two evolving particle systems.'''
    hamiltonian_idx = world_1.indicator_schema.index("Hamiltonian")
    hamiltonian_1 = world_1.get_indicator_history()[:,hamiltonian_idx]
    hamiltonian_2 = world_2.get_indicator_history()[:,hamiltonian_idx]
    differences = hamiltonian_1 - hamiltonian_2
    print(differences)
    return ne.evaluate('sum(differences**2)') / world_1.n_agents

def total_sph_delta_v(world):
    '''
    Returns the total SPH-related delta_v stored in the world's accumulator and clears it.
    '''
    if not 'total_sph_delta_v' in world.scratch_material: return 0
    t = world.scratch_material['total_sph_delta_v']
    world.scratch_material['total_sph_delta_v'] = 0
    return t
