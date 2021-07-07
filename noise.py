import numpy as np

def gaussian_noise(state, mean, std):
    noise = np.random.normal(mean, std, state.shape)
    return state + noise