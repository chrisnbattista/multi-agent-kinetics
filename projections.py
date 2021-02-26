import math
import numpy as np


def mercator_projection(p, r):
    '''
    projects from a position p on a 2d surface onto a sphere of radius r
    '''

    longitude = p[0] / r
    ai = np.exp(p[1]/r)
    latitude = 2 * np.arctan(ai) - math.pi/2

    return np.array([
        r * np.cos(latitude) * np.cos(longitude),
        r * np.cos(latitude) * np.sin(longitude),
        r * np.sin(latitude)
    ])

def stereographic_projection(p, r):
    '''
    '''

    return r * \
        np.array([
            2*p[0] / (1 + p[0]**2 + p[1]**2),
            2*p[1] / (1 + p[0]**2 + p[1]**2),
            (-1 + p[0]**2 + p[1]**2) / (1 + p[0]**2 + p[1]**2)
        ])