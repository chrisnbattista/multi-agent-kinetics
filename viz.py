
two_d_video_images = []



import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.patches as patches

import itertools, random

from . import projections


floating_plots = {}

def get_floating_plot(name, **params):
    '''
    '''
    if not name in floating_plots:
        floating_plots[name] = plt.figure(**{**params, **{'figsize':(6,3)}})
        floating_plots[name].canvas.set_window_title(name)
    return floating_plots[name]

def set_up_figure(title='Plot', plot_type='2d+ind'):
    plt.ion()
    plt.show()
    sns.set_theme()
    sns.color_palette("dark")
    # if plot_type == '2d+ind':
    #     fig, ax = plt.subplots(2,1,
    #                         gridspec_kw={'height_ratios': [4, 1]},
    #                         figsize=(6, 7.5)
    #     )
    # elif plot_type == '2d_proj_orbit':
    fig, ax = plt.subplots(1,1,
                        ##gridspec_kw={'height_ratios': [4, 0]},
                        figsize=(6, 6)
    )
    fig.canvas.set_window_title(title)
    return fig, ax

def trace_trajectories(world, fig, ax, fig_title=''):
    '''
    Performs colored line plots of all particle trajectories in system.
    '''

    ax.clear()

    for i in range(world.n_agents):
        sns.scatterplot(
            x=world.history[i::world.n_agents,3],
            y=world.history[i::world.n_agents,4],
            ax=ax,
            ci=None,
            s=5,
            palette="dark"
        )
        '''sns.scatterplot(
            x=(world.history[i,1],),
            y=(world.history[i,2],),
            ax=ax,

        )'''

    try:
        sns.lineplot(
                    x=world.history[::100, 0],
                    y=world.indicator_history[::100, 0],
                    ax=ax[1],
                    legend=False
        )
    except:
        pass
    
    fig.canvas.set_window_title(fig_title)
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)

def trace_predicted_vs_real_trajectories(y, y_pred, title, fig, ax):
    '''
    '''

    n_agents = len(np.unique(y[:,1]))

    sns.scatterplot(
        x=y[:,0],
        y=y[:,1],
        ax=ax,
        ci=None,
        s=6,
        c=["b"]*y.shape[0],
        palette="dark",
        marker="x"
    )
    sns.scatterplot(
        x=y_pred[:,0],
        y=y_pred[:,1],
        ax=ax,
        ci=None,
        s=4,
        c=['r']*y_pred.shape[0],
        palette="dark",
        marker='o'
    )
    fig.canvas.set_window_title(title)
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)

def render_2d_orbit_state(world,
                    fig,
                    ax,
                    show_indicators=False,
                    indicator_labels=None,
                    fig_title=None,
                    agent_colors=None,
                    agent_sizes=None,
                    agent_markers=None,
                    h=None,
                    t=0):
    '''
    Display the particles described in the world array onto the figure and axis provided.
    Activates the plt event loop so that the figure is displayed.
    Also, display the indicator timeseries below if present.

    Expects:
        world object
        plt figure
        axes object
    '''

    state = world.get_state()

    ax.clear()
    ax.plot(
        (-25, 25),
        (0, 0),
        c='g'
    )

    n_sph = sum(world.context['sph_active'])

    p = sns.scatterplot(
            x=state[:n_sph+1,3],
            y=state[:n_sph+1,4],
            c=agent_colors[:n_sph+1],
            marker='o',
            ##s=agent_sizes,
            ax=ax
    )
    p = sns.scatterplot(
            x=state[n_sph+1:,3],
            y=state[n_sph+1:,4],
            c=agent_colors[n_sph+1:],
            marker='^',
            ##s=agent_sizes,
            ax=ax
    )

    #sum(world.context['sph_active'])

    if h:
        kd_tree = scipy.spatial.cKDTree(state[:,3:5])
        h_pairs = kd_tree.query_pairs(r=h)
        if h_pairs:
            lines = []
            for pair in h_pairs:
                if pair[0] < n_sph or pair[1] < n_sph:
                    lines.append((state[pair[0],3:5], state[pair[1],3:5]))
            lc = matplotlib.collections.LineCollection(lines, colors='k')
            ax.add_collection(lc)

    if show_indicators:
        indicators = world.get_indicator_history()
        n_indicators = len(world.indicators)
        for ind in range(n_indicators):
            i_fig = get_floating_plot(indicator_labels[ind][0])
            i_ax = i_fig.gca()
            i_ax.clear()
            sns.lineplot(
                x=np.linspace(0, world.current_timestep*world.timestep_length, world.current_timestep),
                y=indicators[:world.current_timestep, ind],
                ax=i_ax,
                legend=False
            )

            i_ax.set_xlabel(indicator_labels[ind][1])
            i_ax.set_ylabel(indicator_labels[ind][2])
            i_fig.tight_layout()
            if indicator_labels:
                i_ax.legend(loc='lower right', labels=[indicator_labels[ind][0]])
    
    if fig_title != None:
        fig.canvas.set_window_title(fig_title)
    
    note = 'Sim time: {:.2f} ksec | h: {:.2f}'.format(world.current_timestep*world.timestep_length, h)
    if not len(fig.texts):
        fig.text(0.01, 0.01, note)
    else:
        fig.texts[0].set_text(note)

    ax.set(xlim=(-25, 25), ylim=(-25, 25))

    ax.set_xlabel('orbital plane basis 1 (km)')
    ax.set_ylabel('orbital plane basis 2 (km)')

    ##two_d_video_images.append(ax.get_images()[0])

    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)

def render_projected_2d_orbit_state(
                    world,
                    fig,
                    ax,
                    show_indicators=False,
                    indicators=None,
                    indicator_labels=None,
                    fig_title=None,
                    agent_colors=None,
                    agent_sizes=None,
                    orbit_radius=1,
                    scaling_factor=1,
                    t=0):
    '''
    '''

    state = world.get_state()

    transformed_world = np.apply_along_axis(lambda p: projections.mercator_projection(p=p[3:5]+world.context['cumulative_recentering'], r=orbit_radius), 1, state/scaling_factor)

    for i in (0,1):
        ax.set_yticklabels([])
        ax.set_xticklabels([])  
        ax.set_yticks([])
        ax.set_xticks([])
    ax = fig.gca(projection='3d')
    ax.clear()
    ax.set_xlim(-orbit_radius, orbit_radius)
    ax.set_ylim(-orbit_radius, orbit_radius)
    ax.set_zlim(-orbit_radius, orbit_radius)

    ax.set_xlabel('ECI position, basis 1 (km)')
    ax.set_ylabel('ECI position, basis 2 (km)')
    ax.set_zlabel('ECI position, basis 3 (km)')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = orbit_radius * np.outer(np.cos(u), np.sin(v))
    y = orbit_radius * np.outer(np.sin(u), np.sin(v))
    z = orbit_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.1)
    
    ax.scatter(
        transformed_world[:,0],
        transformed_world[:,1],
        transformed_world[:,2],
        c=agent_colors,
        ##s=agent_sizes
    )

    theta = np.linspace(0, 2 * np.pi, 201)
    x = orbit_radius*np.cos(theta)
    y = orbit_radius*np.sin(theta)
    ax.plot(x,
                y,
                c='g')
    
    if fig_title != None:
        fig.canvas.set_window_title(fig_title)

    note = 'Sim time: {:.2f} ksec'.format(world.current_timestep*world.timestep_length)
    if not len(fig.texts):
        fig.text(0.01, 0.01, note)
    else:
        fig.texts[0].set_text(note)

    ##ax.set(xlim=(-10, 50), ylim=(-30, 30))

    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)

def render_1d_orbit_state(world,
                    fig,
                    ax,
                    h,
                    show_indicators=False,
                    indicators=None,
                    indicator_labels=None,
                    fig_title=None,
                    agent_colors=None,
                    agent_sizes=None):

    # print(sample_data)
    # print(sample_data.shape)
    # print(sample_data[:,2])

    ax.clear()

    # maximum is the position of the lead agent
    maximum = max(world[:,3])
    minimum = min(world[:,3])

    sns.lineplot(
        x = [minimum, maximum], 
        y = [0, 0], 
        color = 'g', 
        ax = ax
    )

    color = []
    for c in range (0, 5):
        if world[c, 3] == maximum:
            color.append('b')
        else:
            color.append('k')
    
    p = sns.scatterplot(
            x=world[:,3],
            y=[0,0,0,0,0],
            c= color,
            #c=agent_colors, ## these are the offending lines.
            #s=agent_sizes,  ## maybe check to see if the arguments are None higher up in this function, and initialize them if so?
            ax=ax         ## or, just remove if not needed, or, use if/else to only pass them if not None
    )
    
    # control x and y limits
    p.set_xlim([-30, 30])
    p.set_ylim([-30, 30])
    p.set_aspect('equal')
    
<<<<<<< HEAD

    r = h
    for c in range (0, 5):
        #fig, ax = plt.subplots()
        circle1 = plt.Circle((world[c,3], 0), r, color="orange", fill=False)
        p.add_artist(circle1)
        if world[c, 3] == maximum:
            circle2 = plt.Circle((world[c,3], 0), r, color="green", fill=False)
            p.add_artist(circle2)
=======
    # sns.scatterplot(data[:, 0], data[:, 1], c = color, ax = ax)
>>>>>>> 5c41f3ce478293a5d3b429326fdf0ec2bd1f340e

    if show_indicators and indicators:
        n_indicators = indicators.shape[1]
        ax[1].clear()
        for ind in range(1, n_indicators+1):
            sns.lineplot(
                x=indicators[:i, 0],
                y=indicators[:i, ind],
                ax=ax[1],
                legend=False
            )
        
    if indicator_labels:
        plt.legend(loc='lower right', labels=indicator_labels)
    
    if fig_title != None:
        fig.canvas.set_window_title(fig_title)
    # if not len(fig.texts):
    #     fig.text(0.01, 0.01, note)
    # else:
    #     fig.texts[0].set_text(note)

    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)

def generate_cost_plot(model,
                        data,
                        criterion,
                        param_ranges,
                        index_range):
    '''

    '''

    ordered_keys = list(param_ranges.keys())
    ordered_ranges = [param_ranges[k] for k in ordered_keys]
    combinations = list(itertools.product(*ordered_ranges))

    cost_data = np.zeros( (len(combinations), len(ordered_keys)+1) )

    for k in tqdm(index_range):

        x, y = data[k]

        for j in range(len(combinations)):

            cost_data[j, 1:3] = combinations[j]

            for i in range(len(ordered_keys)):
                model.__getattr__(ordered_keys[i]).data = torch.Tensor([combinations[j][i]])
            
            y_pred = model(*x)
            y_pred = y_pred.detach()

            if y.ndim > 1 and int(random.randint(0, int(len(combinations)/2))) == int(j/2):
                trace_predicted_vs_real_trajectories(y, y_pred, str(combinations[j]), *set_up_figure())
            
            cost_data[j,0] = np.add(
                cost_data[j,0],
                np.sum(
                    (y - y_pred).numpy()**2
                )
            )

    cost_data[:,0] = np.abs(cost_data[:,0])
    cost_data[:,0] = np.log10(
        cost_data[:,0] / len(index_range),
        out=np.zeros_like(cost_data[:,0]),
        where=(cost_data[:,0]!=0)
    )
    print(cost_data)

    if (len(ordered_keys) == 2): # can be plotted for mortals' understanding, no hypercubes...
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ##surf = ax.plot_trisurf(cost_data[:,1], cost_data[:,2], cost_data[:,0])
        ax.scatter(cost_data[:,1], cost_data[:,2], cost_data[:,0])
        ax.set_xlabel(ordered_keys[0])
        ax.set_ylabel(ordered_keys[1])
        ax.set_zlabel("cost")
        print(f'''Min value: {cost_data[:,0].min()}
        Position: {combinations[np.argmin(cost_data[:,0])]}
        ''')
        while True:
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.01)