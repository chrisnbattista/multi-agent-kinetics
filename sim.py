





from . import indicators, experiments, forces, integrators, worlds
import numpy as np
import time, random




def run_random_circle_lj_sim(params, seed):
    '''
    Sets up a random experiment using the random seed and params and runs it to completion.
    '''

    random.seed(seed)

    ## ICs
    initial_state = experiments.initialize_random_circle(
        radius=params['size']/2,
        center=(params['size']/2, params['size']/2),
        n_particles=params['n_agents'],
        min_dist=params['min_dist'],
        random_speed=params['init_speed'])

    ## Create World object
    world = worlds.World(
        initial_state=initial_state,
        n_agents=params['n_agents'],
        n_timesteps=params['n_timesteps'],
        timestep=params['timestep'],
        forces=[
            lambda world, context: forces.pairwise_world_lennard_jones_force(world, epsilon=params['epsilon'], sigma=params['sigma'])
        ],
        indicators=[
            lambda world: indicators.kinetic_energy(world)
        ]
    )
    
    ## Forward run
    world.advance_state(params['n_timesteps']-1)
    
    return world