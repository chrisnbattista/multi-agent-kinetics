from multi_agent_kinetics import projections, kernels, viz

print("Plotting Gaussian bell curve")
fig, ax = viz.set_up_figure(title="Gaussian curve")

ax[0].clear()

kernels.gaussian_function(0.5)