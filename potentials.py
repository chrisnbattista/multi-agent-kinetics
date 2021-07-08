import numpy as np
import scipy
import torch
from . import worlds

def potential_energy(world, potentials):
    '''
    Calculates potential energy of a whole particle system.
    '''

    # Requires a list of all potentials, which are to return scalar
    # global values
    V = 0

    for potential in potentials:
        v = potential(world)
        V += v

    return V

def kinetic_energy(world):
    '''
    Calculates kinetic energy of a whole particle system.
    '''

    vel_mags = torch.linalg.norm(world.get_state()[:,worlds.vel[world.spatial_dims]], axis=1)
    T = torch.sum( ((vel_mags**2) * world.get_state()[:,worlds.mass[world.spatial_dims]] / 2),
                axis=0)

    return T

def gravitational_potential_energy(world, G):
    '''Returns the total GPE of a world system based on Newton's law of gravitation and a given G.'''
    state = world.get_state()
    position = state[:, worlds.pos[world.spatial_dims]]
    mass = state[:, worlds.mass[world.spatial_dims]]
    one_over_r = torch.reciprocal(torch.tensor(scipy.spatial.distance.pdist(position.detach())))
    one_over_r_mat = torch.tensor(scipy.spatial.distance.squareform(one_over_r))
    one_over_r_mat = torch.nan_to_num(one_over_r_mat, posinf=0, neginf=0)
    m_M_over_r_mat = one_over_r_mat * mass * mass[...,None]
    pairwise_U = torch.sum(torch.triu(m_M_over_r_mat))
    U_mat = (-1) * G * pairwise_U
    return torch.sum(U_mat)