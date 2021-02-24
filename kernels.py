# code continuous Gaussian kernel function
# goal: effectively make interactions local, weighting by distance essentially

import math
import numpy as np
import matplotlib.pyplot as plt

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
    elif (r >= h) and (r <= 2*h):
        return sigma**3 * 0.25 * (2 - r/h)**3
    else:
        return sigma**3 * ( 1 - 1.5 * (r/h)**2 * (1 - r/h/2) ) 

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
        return sigma**3 * -3*(r/h - 2)**2
    else:
        return sigma**3 * 0.75 * r/h * (3*r/h - 4)

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
        return sigma**3 * (12 - 6*r/h)
    else:
        return sigma**3 * 0.5 * (9*r/h - 6)

def gaussian_function(r, h = 1, a = 1, b = 0, c = 1):
    y = r/h
    value = a*np.exp((-((y-b)**2)/2*(c)**2))
    if r > h:
        return 0
    elif r <= h:
        print('{:.4f}'.format(value))
        return value
    
    mean = b
    standard_deviation = c

    x_values = np.arange(-10, 10, 0.1)
    y_values = scipy.stats.norm(mean, standard_deviation)

    plt.plot(x_values, y_values.pdf(x_values))

# test gaussian_function(r = 5, h = 10, a = 10, b = 0, c = 1)
