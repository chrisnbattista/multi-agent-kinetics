import numpy as np





def recenter_on_agent(world, state, agent_i):
    '''
    Moves all other agents to keep the plane centered on the agent with index agent_i.
    '''

    if not 'cumulative_recentering' in world.context:
        world.context['cumulative_recentering'] = np.zeros((world.spatial_dims,))
    
    offset = state[agent_i,3:5]
    
    world.context['cumulative_recentering'] += offset
    state[:, 3:5] = state[:, 3:5] - offset

def linear_motion(world, state, velocity):
    '''
    Moves all agents a linear offset (velocity) each unit time.
    '''

    state[:,3:5] += velocity * world.timestep_length

def constrain_to_orbit(world, state):
    '''
    '''
    state[:,4] = np.where(
        np.array(world.context['following_active']),
        0,
        state[:,4]
    )
        