# code continuous Gaussian kernel function
# goal: effectively make interactions local, weighting by distance essentially

import math

def quadratic(r, h):
    """Implements a quadratic kernel function with support radius of 2h."""
    if r/h < 2:
        return 15 / (16*math.pi * h**4) * (r/h/2 - 1)
    else:
        return 0

def cubic_spline(r, sigma=None, h=1):
    '''
    Computes the cubic spline function
    sigma = tuning constant
    h = 
    r = r_1 - r_2 (scalar distance)
    '''

    if sigma == None:
        sigma = 10/(7*math.pi*h**2)

    if r > 2*h:
        return 0
    elif (r <= 2*h) and (r >= h):
        return sigma * 0.25 * (2 - r/h)**3
    else:
        return sigma * ( 1 - 1.5 * (r/h)**2 * (1 - r/h/2) ) 

def cubic_spline_grad(r, sigma=None, h=1):
    '''
    Computes the B-spline function
    sigma = tuning constant
    h = 
    r = r_1 - r_2 (scalar distance)
    '''

    if sigma == None:
        sigma = 10/(7*math.pi*h**2)

    if r > 2*h:
        return 0
    elif (r >= h) and (r <= 2*h):
        return sigma * -3*(r/h - 2)**2
    else:
        return sigma * 0.75 * r/h * (3*r/h - 4)

def cubic_spline_grad_double(r, sigma=None, h=1):
    '''
    Computes the B-spline function
    sigma = tuning constant
    h = 
    r = r_1 - r_2 (scalar distance)
    '''

    if sigma == None:
        sigma = 10/(7*math.pi*h**2)

    if r > 2*h:
        return 0
    elif (r >= h) and (r <= 2*h):
        return sigma * (12 - 6*r/h)
    else:
        return sigma * 0.5 * (9*r/h - 6)