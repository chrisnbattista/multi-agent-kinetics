import numpy as np
import pandas as pd

## Simple scalar integration functions

def integrate_rect(delta_t, initial_value):
    return initial_value * delta_t

def double_integrate_rect(delta_t, initial_value):
    return initial_value * delta_t**2


## World level integration functions

def integrate_rect_world(world, force_matrix, timestep):
    '''
    '''
    if world.shape[1] == 7:
        vel_idx = slice(5,7)
        pos_idx = slice(3,5)
    else:
        vel_idx = slice(4,4)
        pos_idx = slice(3,3)

    # Velocity from force
    # F = ma = m * derivative(v)
    # v = integral(F) / m
    velocity_matrix = np.divide(
        integrate_rect(timestep, force_matrix),
        world[:,2,None]
    )
    world[:, vel_idx] += velocity_matrix

    # Displacement from velocity
    world[:, pos_idx] += integrate_rect(timestep, world[:, vel_idx])

    return world
