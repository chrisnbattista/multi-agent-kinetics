import numpy as np
import scipy
import scipy.spatial
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

    return kinetic_energy_from_state(world.get_state(), world.spatial_dims)


def kinetic_energy_from_state(state, spatial_dims):
    vel_mags = torch.linalg.norm(state[:, worlds.vel[spatial_dims]], axis=1)
    T = torch.sum(((vel_mags**2) * state[:, worlds.mass[spatial_dims]] / 2),
                  axis=0)

    return T


def gravitational_potential_energy(world, G):
    '''Returns the total GPE of a world system based on Newton's law of gravitation and a given G.'''
    state = world.get_state()
    return gravitational_potential_energy_from_state(state, world.spatial_dims,
                                                     G)


def gravitational_potential_energy_from_state(state, spatial_dims, G):
    position = state[:, worlds.pos[spatial_dims]]
    mass = state[:, worlds.mass[spatial_dims]]
    one_over_r = torch.reciprocal(
        torch.tensor(scipy.spatial.distance.pdist(position.detach())))
    one_over_r_mat = torch.tensor(
        scipy.spatial.distance.squareform(one_over_r))
    one_over_r_mat = torch.nan_to_num(one_over_r_mat, posinf=0, neginf=0)
    m_M_over_r_mat = one_over_r_mat * mass * mass[..., None]
    pairwise_U = torch.sum(torch.triu(m_M_over_r_mat))
    U_mat = (-1) * G * pairwise_U
    return torch.sum(U_mat)