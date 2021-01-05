





import numpy as np
from sfc.multi_agent_kinetics import experiments, integrators





schemas = {
    '2d': ('id', 'b_1', 'b_2', 'm', 'v_1', 'v_2', 't'),
    '3d': ('id', 'm', 'b_1', 'b_2', 'b_3', 'v_1', 'v_2', 'v_3', 't')
}





class World:

    def __init__(self,
                spatial_dims=2,
                n_agents=0,
                n_timesteps=None,
                timestep_length=0.01,
                forces=[],
                indicators=[],
                integrator=integrators.integrate_rect_world):
        '''
        '''

        self.spatial_dims = spatial_dims
        self.schema = schemas[str(spatial_dims)+'d']

        self.n_timesteps = n_timesteps
        self.current_timestep = 0
        if n_timesteps:
            self.fixed_length = True
            self.history = np.empty( (n_timesteps * n_agents, len(self.schema)))
        else:
            self.fixed_length = False
            self.history = np.empty( (0, len(self.schema)))

        self.n_agents = n_agents
        # control actions is basically forces to be applied, originating from the agent
        self.control_actions =  np.zeros( ( self.n_agents, self.spatial_dims ) )

        self.forces = forces
        self.indicators = indicators
        self.integrator = integrator

        self.timestep_length = timestep_length
    
    def set_state(self, world_state):
        '''
        '''

        if self.fixed_length:
            self.history[self.current_timestep:self.current_timestep+self.n_agents, :] = world_state
        else:
            self.history = np.concatenate((self.history, world_state))
    
    def get_state(self):
        '''
        '''

        return self.history[self.current_timestep - self.n_agents:, :]
    
    def _set_indicators(self, indicators):
        '''
        Not implemented yet.
        '''
        pass

    def add_force(self, force_function):
        '''
        '''

        self.forces.append(force_function)
    
    def add_indicator(self, indicator_function):
        '''
        '''

        self.indicators.append(indicator_function)
    
    def add_control_actions(self):
        '''
        '''

        self.forces.append(self.apply_control_actions)
    
    def apply_control_actions(self, world):
        '''
        Function matching the specifications for multi-agent-kinetics.experiments to use as a force.
        '''

        return self.control_actions
    
    def time_step(self):
        '''
        Advances the underlying world state by one time step.
        '''

        new_state, new_indicators = experiments.advance_timestep(
            world=self.get_state(),
            timestep=self.timestep_length,
            integrator=self.integrator,
            forces=self.forces,
            indicators=self.indicators
        )

        self.set_state(new_state)
        self._set_indicators(new_indicators)