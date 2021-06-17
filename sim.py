from . import indicators, experiments, forces, integrators, worlds
import numpy as np
import time, random

def generate_generic_ic(agent_list):
    '''Generate a worlds.World compatible initial_state based on a list of agent properties lists: mass, position, velocity. only compatible with 3D.'''
    initial_state = np.zeros((len(agent_list), len(worlds.schemas['3d'])))
    for i in range(len(agent_list)):
        initial_state[i, :] = (0, i, *agent_list[i])
    return initial_state

def run_random_circle_sim(params, seed, forces):
    '''
    Sets up a random experiment using the random seed and params and runs it to completion.
    '''

    random.seed(seed)

    ## ICs
    if 'mass' in params: mass = params['mass']
    else: mass = 1
    initial_state = experiments.initialize_random_circle(
        radius=params['size']/2,
        center=(params['size']/2, params['size']/2),
        n_particles=params['n_agents'],
        min_dist=params['min_dist'],
        random_speed=params['init_speed'],
        mass=mass)

    ## Create World object
    world = worlds.World(
        initial_state=initial_state,
        n_agents=params['n_agents'],
        n_timesteps=params['n_timesteps'],
        timestep=params['timestep'],
        forces=forces,
        indicators=[
            lambda world: indicators.kinetic_energy(world)
        ]
    )
    
    ## Forward run
    world.advance_state(params['n_timesteps']-1)
    
    return world