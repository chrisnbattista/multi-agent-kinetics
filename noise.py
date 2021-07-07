import numpy as np

def gaussian_noise(state, mean, std):
    try:
        noise = np.random.normal(mean, std, state.shape)
    except AttributeError:
        noise = float(np.random.normal(mean, std, (1,)))
    return state + noise