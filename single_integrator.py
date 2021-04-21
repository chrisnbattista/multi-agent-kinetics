import pandas as pd
import random

def random_2_particle_IC(radius=1):
    data = pd.DataFrame(columns=[
        't', 'id', 'm', 'b_1', 'b_2' 'v_1', 'v_2'
    ])
    for i in (0,1):
        data.append([
            0,
            i,
            10,
            (random.random()-0.5)*radius*2,
            (random.random()-0.5)*radius*2,
            0,
            0
        ])
    return data

def propagate(dynamics_function, data):
    old_data = data.iloc[-2:,:]
    new_data = data.iloc[-2:,:].copy()
    new_data[['v_1', 'v_2']] = old_data.apply(dynamics_function, axis=1)
    new_data[['b_1', 'b_2']] = new_data[['b_1', 'b_2']] + new_data[['v_1', 'v_2']]
    return new_data

def run_sim(dynamics, steps=1000, radius=1):
    data = random_2_particle_IC(radius)
    for _ in range(steps):
        data = pd.concat(
            [
                data,
                propagate(dynamics, data)
            ]
        )
    return data

if __name__ == '__main__':
    run_sim(lambda x: 0, 0)
    print("Test passed.")