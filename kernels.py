# code continuous Gaussian kernel function
# goal: effectively make interactions local, weighting by distance essentially

import math

three_d_sigma = 10/(7*math.PI)

def cubic_spline(q, sigma=three_d_sigma, h=1):
    '''
    Computes the B-spline function
    sigma = tuning constant
    h = 
    q = r_1 - r_2 (scalar distance)
    '''

    if q > 2:
        return 0
    elif (q >= 1) and (q <= 2):
        return sigma**3 * (2 - q)**3
    else:
        return sigma**3 * ( (2-q)**3 - 4*(1-q)**3 ) 

def cubic_spline_grad(q, sigma=three_d_sigma, h=1):
    '''
    Computes the B-spline function
    sigma = tuning constant
    h = 
    q = r_1 - r_2 (scalar distance)
    '''

    if q > 2:
        return 0
    elif (q >= 1) and (q <= 2):
        return sigma**3 * (-3) * (2 - q)**2
    else:
        return sigma**3 * ( (-3) * (2-q)**2 - (-3) * 4*(1-q)**2 ) 

def cubic_spline_grad_double(q, sigma=three_d_sigma, h=1):
    '''
    Computes the B-spline function
    sigma = tuning constant
    h = 
    q = r_1 - r_2 (scalar distance)
    '''

    if q > 2:
        return 0
    elif (q >= 1) and (q <= 2):
        return sigma**3 * (6) * (2 - q)**2
    else:
        return sigma**3 * ( (6) * (2-q) - (6) * 4*(1-q) )