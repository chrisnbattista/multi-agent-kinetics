import math
import numpy as np


def mercator_projection(p, r):
    '''
    projects from a position p on a 2d surface onto a sphere of radius r
    '''

    return np.array([
        r * np.cos(p[1]) * np.cos(p[0]),
        r * np.cos(p[1]) * np.sin(p[0]),
        r * np.sin(p[1])
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