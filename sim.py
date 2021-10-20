from . import indicators, experiments, forces, integrators, worlds
import numpy as np
import torch
import time, random

def generate_generic_ic(agent_list):
    '''Generate a worlds.World compatible initial_state based on a list of agent properties lists: mass, position, velocity. only compatible with 3D.'''
    initial_state = torch.zeros((len(agent_list), len(worlds.schemas['3d'])))
    for i in range(len(agent_list)):
        initial_state[i, :] = (0, i, *agent_list[i])
    return initial_state

def run_random_circle_sim(params, seed, forces, indicators=[], indicator_schema=[], noise=None, spatial_dims=3):
    '''
    Sets up a random experiment using the random seed and params and runs it to completion.
    '''

    random.seed(seed)

    ## ICs
    if 'mass' in params: mass = params['mass']
    else: mass = 1
    initial_state = experiments.initialize_random_sphere(
        radius=params['size']/2,
        n_particles=params['n_agents'],
        min_dist=params['min_dist'],
        random_speed=params['init_speed'],
        mass=mass,
        spatial_dims=spatial_dims
        )

    ## Create World object
    world = worlds.World(
        initial_state=initial_state,
        n_agents=params['n_agents'],
        n_timesteps=params['n_timesteps'],
        timestep=params['timestep'],
        forces=forces,
        indicators=indicators,
        indicator_schema=indicator_schema,
        noise=noise,
        spatial_dims=spatial_dims
    )
    
    ## Forward run
    world.advance_state(params['n_timesteps']-1)
    
    return world

def run_two_body_sim(params, seed, forces, indicators=[], indicator_schema=[], noise=lambda x: x):
    '''Initializes two masses and runs a simulation of their behavior.'''

    random.seed(seed)

    noise = random.random()

    ## ICs=
    #schema '2d': ('t', 'id', 'm', 'b_1', 'b_2', 'v_1', 'v_2'),
    initial_state = torch.tensor([
        [0, 0, params['small_m'], params['separation'], 0, 0, params['tangent_speed'] * noise],
        [0, 1, params['large_m'], 0, 0, 0, 0]
    ])

    ## Create World object
    world = worlds.World(
        initial_state=initial_state,
        n_agents=2,
        n_timesteps=params['n_timesteps'],
        timestep=params['timestep'],
        forces=forces,
        indicators=indicators,
        indicator_schema=indicator_schema
    )
    
    ## Forward run
    world.advance_state(params['n_timesteps']-1)
    
    return world