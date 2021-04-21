import numpy as np
from . import worlds




def recenter_on_agent(world, state, agent_i, context=None):
    '''
    Moves all other agents to keep the plane centered on the agent with index agent_i.
    '''

    pos = worlds.pos[context['spatial_dims']]

    if not 'cumulative_recentering' in world.context:
        world.context['cumulative_recentering'] = np.zeros((context['spatial_dims'],))
    
    offset = state[agent_i,pos]
    
    world.context['cumulative_recentering'] += offset
    state[:, pos] = state[:, pos] - offset

def linear_motion(world, state, velocity, context=None):
    '''
    Moves all agents a linear offset (velocity) each unit time.
    '''

    pos = worlds.pos[context['spatial_dims']]

    state[:,pos] += velocity * world.timestep_length

def constrain_to_orbit(world, state):
    '''
    '''
    state[:,4] = np.where(
        np.array(world.context['following_active']),
        0,
        state[:,4]
    )
        