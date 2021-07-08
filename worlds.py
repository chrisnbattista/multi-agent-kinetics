## TO DO: Remove explicit indexes, change to lookups from object schema
import torch
import numpy as np
from . import experiments, integrators, decorators

def torch_tile(tensor, dim, n):
    """Tile n times along the dim axis"""
    if dim == 0:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,n,1).view(-1,tensor.shape[1])
    else:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,1,n).view(tensor.shape[0], -1)

schemas = {
    '1d': ('t', 'id', 'm', 'b_1', 'v_1'),
    '2d': ('t', 'id', 'm', 'b_1', 'b_2', 'v_1', 'v_2'),
    '3d': ('t', 'id', 'm', 'b_1', 'b_2', 'b_3', 'v_1', 'v_2', 'v_3')
}
time = [
    None,
    0,
    0,
    0
]
ID = [
    None,
    1,
    1,
    1
]
pos = [
    None,
    slice(3,4),
    slice(3,5),
    slice(3,6)
]
vel = [
    None,
    slice(4,5),
    slice(5,7),
    slice(6,9)
]
mass = [
    None,
    2,
    2,
    2
]

class World:

    def __init__(self,
                initial_state=None,
                spatial_dims=2,
                n_agents=0,
                n_timesteps=None,
                timestep=0.0001,
                forces=[],
                constraints=[],
                controllers=[],
                indicators=[],
                indicator_schema=[],
                integrator=integrators.integrate_rect_world,
                noise=None,
                context=None,
                **kwargs):
        '''
        Create a new World object with fixed parameters: forces, indicators, fixed timestep, etc.
        If n_timesteps is left as None, the World can run indefinitely (very inefficient).
        '''

        self.spatial_dims = spatial_dims
        self.schema = schemas[str(spatial_dims)+'d']
        self.indicator_schema = indicator_schema

        if (initial_state is not None):
            self.n_agents = initial_state.shape[0]
        else:
            self.n_agents = n_agents

        self.n_timesteps = n_timesteps
        if n_timesteps:
            self.fixed_length = True
            self.history = torch.empty( (n_timesteps * self.n_agents, len(self.schema)))
            self.indicator_history = torch.empty( (n_timesteps, len(indicators)) )
        else:
            self.fixed_length = False
            self.history = torch.empty( (0, len(self.schema)))
            self.indicator_history = torch.empty( (0, len(indicators)) )

        self.forces = forces
        self.constraints = constraints
        self.controllers = controllers

        self.integrator = integrator
        self.indicators = indicators
        self.noise = noise

        self.timestep_length = timestep

        self.context = context

        # saved material to help with indicator computation
        self.scratch_material = {}

        self.current_timestep = None
        if initial_state is not None:
            self._add_state_to_history(initial_state)
            ## Compute indicators
            indicator_results = torch.empty( (1, len(self.indicators)) )
            for j in range(len(self.indicators)):
                indicator_results[0, j] = self.indicators[j](self)
            self._add_state_to_indicators(indicator_results)
    
    def _add_state_to_history(self, new_state):
        '''
        Adds the new_state to the state history of the world and advances the timestep by one.
        '''

        if self.current_timestep is None:
            self.current_timestep = 0
        else:
            self.current_timestep += 1

        if self.fixed_length:
            self.history[
                (self.current_timestep*self.n_agents) : (self.current_timestep*self.n_agents + self.n_agents), :] = new_state
        else:
            self.history = torch.cat((self.history, new_state), axis=0)
        
        return self
    
    def _add_state_to_indicators(self, new_indicators):
        '''
        cats new_indicators to the indicators timeseries at index self.current_timestep.
        '''

        if self.fixed_length:
            self.indicator_history[self.current_timestep, :] = new_indicators
        else:
            self.indicator_history = torch.cat((self.indicator_history, new_indicators))
        
        return self
    
    def get_state(self):
        '''
        Returns a view of the latest entry in the world state history.
        '''

        if self.current_timestep is None:
            return None

        return self.history[(self.current_timestep*self.n_agents) : ((self.current_timestep+1)*self.n_agents), :]
    
    def get_state_with_indicators(self):
        '''
        Returns a view of the latest entry in the world state history, along with the latest indicators.
        '''
        return torch.cat(
            (
                self.get_state(),
                torch_tile(self.indicator_history[(self.current_timestep), :].reshape(1,-1), 0, self.n_agents)
            ),
            axis=1
        )
    
    def get_history(self):
        '''
        Returns full state evolution time series.
        '''
        return self.history
    
    def get_indicator_history(self):
        '''
        Returns full indicator time series.
        '''
        return self.indicator_history
    
    def get_full_history_with_indicators(self):
        '''
        Returns an array representing both state and indicators across entire time span.
        '''
        return \
            torch.cat(
                (
                    self.history,
                    torch_tile(self.indicator_history, 0, self.n_agents)
                ),
                axis=1
            )
    
    def advance_state(self, steps=1):
        '''
        Advances the underlying world state by the specified number of time steps and records the state trajectory.
        '''

        for i in range(steps):

            state = self.get_state().clone().detach() # state is initially old state

            ## Calculate forces
            # Initialize matrix to hold forces keyed to id
            force_matrix = torch.zeros( (state.shape[0], self.spatial_dims) )
            # Apply natural forces
            for force in self.forces:
                force_matrix = force_matrix + force(self, self.context)
            # Apply control forces
            for controller in self.controllers:
                force_matrix = force_matrix + controller.control(self, self.context)

            ## Advance the timestep itself
            state[:,0] += self.timestep_length

            ## Integrate forces over timestep
            state = self.integrator(self, state, force_matrix, self.timestep_length)

            ## Add noise
            if self.noise != None:
                state[pos[self.spatial_dims]] = self.noise(state[pos[self.spatial_dims]])

            ## Apply any constraints
            if self.constraints:
                for constraint in self.constraints:
                    constraint(self, state, self.context)

            # state is now new state, so append it to the history and advance the internal
            # timestep counter
            self._add_state_to_history(state)

            ## Compute indicators
            indicator_results = torch.empty( (1, len(self.indicators)) )
            for j in range(len(self.indicators)):
                indicator_results[0, j] = self.indicators[j](self)
            self._add_state_to_indicators(indicator_results)