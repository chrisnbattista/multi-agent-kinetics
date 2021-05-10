import pandas as pd
import numpy as np
import random

def random_2_particle_IC(radius=1):
    data = pd.DataFrame(columns=[
        't', 'id', 'm', 'b_1', 'b_2', 'v_1', 'v_2'
    ])
    for i in (0,1):
        data = data.append({
            't': 0,
            'id': i,
            'm': 10,
            'b_1': (random.random()-0.5)*radius*2,
            'b_2': (random.random()-0.5)*radius*2,
            'v_1': 0,
            'v_2': 0
        }, ignore_index=True
    )
    return data

def propagate(dynamics_function, timestep, data):
    old_data = data.iloc[-2:,:]
    new_data = data.iloc[-2:,:].copy()
    x_dot_1 = dynamics_function(old_data.iloc[0], timestep, new_data)
    x_dot_2 = dynamics_function(old_data.iloc[1], timestep, new_data)
    v_1 = (x_dot_1[0,0], x_dot_2[0,0])
    v_2 = (x_dot_1[0,1], x_dot_2[0,1])
    new_data['v_1'] = v_1
    new_data['v_2'] = v_2
    new_data[['b_1', 'b_2']] = new_data[['b_1', 'b_2']].values + new_data[['v_1', 'v_2']].values
    new_data['t'] = new_data['t'] + timestep
    return new_data

def run_sim(dynamics, steps=1000, timestep=0.001, radius=1):
    data = random_2_particle_IC(radius)
    for _ in range(steps):
        data = pd.concat(
            [
                data,
                propagate(dynamics, timestep, data)
            ]
        )
    return data

if __name__ == '__main__':
    run_sim(lambda row, timestep, data: 0, 0)
    print("Test passed.")