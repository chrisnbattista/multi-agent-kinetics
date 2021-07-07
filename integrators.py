import numpy as np
import pandas as pd
from . import worlds

## Simple scalar integration functions

def integrate_rect(delta_t, initial_value):
    return initial_value * delta_t

def double_integrate_rect(delta_t, initial_value):
    return initial_value * delta_t**2


## World level integration functions

def integrate_rect_world(world, state, force_matrix, timestep):
    '''
    '''

    pos = worlds.pos[world.spatial_dims]
    vel = worlds.vel[world.spatial_dims]

    # Velocity from force
    # F = ma = m * derivative(v)
    # v = integral(F) / m
    velocity_matrix = np.divide(
        integrate_rect(timestep, force_matrix),
        state[:,worlds.mass[world.spatial_dims],None]
    )
    state[:, vel] = state[:, vel] + velocity_matrix

    # Displacement from velocity
    state[:, pos] = state[:, pos] + integrate_rect(timestep, state[:, vel])

    return state
