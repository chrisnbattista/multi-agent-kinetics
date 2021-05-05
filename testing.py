import matplotlib
import matplotlib.pyplot as plt
from multi_agent_kinetics import projections, kernels, viz
from inspect import getmembers, isfunction
from kernels import quadratic, cubic_spline, cubic_spline_grad, cubic_spline_grad_double, gaussian_function

plots = []
plot = plt.figure()
plots.append(plot)


