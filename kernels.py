# code continuous Gaussian kernel function
# goal: effectively make interactions local, weighting by distance essentially

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from . import viz, projections

def quadratic(r, h):
    """Implements a quadratic kernel function with support radius of 2h."""
    if r/h < 2:
        return 15 / (16 * math.pi * h**3) * ((r/h)**2 / 4 - r/h + 1)
    else:
        return 0

def quadratic_grad(r, h):
    """Implements the gradient of a quadratic kernel function with support radius of 2h."""
    if r/h < 2:
        return 15 / (16 * math.pi * h**4) * (r/h/2 - 1)
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
        sigma = 1/(math.pi*h**3)

    if r > 2*h:
        return 0
    elif (r >= h) and (r <= 2*h):
        return sigma / 4 * -3*(r/h - 2)**2
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
        return sigma**3 * 0.5 * (9*r/h - 6)

def gaussian_function(r, h = 1, a = 1, b = 0, c = 1, 
                    show_indicators=False,
                    indicators=None,
                    indicator_labels=None,
                    fig_title=None,
                    agent_colors=None,
                    agent_sizes=None):
    y = r/h
    value = a*np.exp((-((y-b)**2)/2*(c)**2))
    if r > h:
        print("0")
        return 0
    elif r <= h:
        print('{:.4f}'.format(value))
        print("Plotting Gaussian bell curve")
        fig, ax = viz.set_up_figure(title="Gaussian Curve")
        ax[0].clear()
        x_axis = np.arange(-10, 10, 0.001)
        plt.plot(x_axis, norm.pdf(x_axis,b,c))
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(10)
        return value
    
    

    
    

<<<<<<< HEAD
        return sigma * 0.5 * (9*r/h - 6)
=======
>>>>>>> d565bdd27d4220c9296c977a535262a8bea33570
