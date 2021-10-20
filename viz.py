two_d_video_images = []
from multi_agent_kinetics import indicators, worlds
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.patches as patches
import itertools, random
from . import projections, worlds

def set_up_3d_plot():
    '''Prepares a figure and axis for 3D plots.'''
    orbit_fig = plt.figure()
    orbit_ax = orbit_fig.add_subplot(111, projection='3d')
    orbit_ax.view_init(60, -140)
    orbit_ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    orbit_ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    orbit_ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    return orbit_fig, orbit_ax

def set_up_plot():
    '''Sets up a generic 2D plot figure and axis.'''
    plot_fig = plt.figure()
    plot_ax = plot_fig.add_subplot(111)
    return plot_fig, plot_ax

def plot_earth(orbit_ax, earth_rad=6371000):
    '''Plots a wireframe of the Earth.'''
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v) * earth_rad
    y = np.sin(u)*np.sin(v) * earth_rad
    z = np.cos(v) * earth_rad
    orbit_ax.plot_wireframe(x, y, z, colors="gray", linewidths=0.5, alpha=0.4)

floating_plots = {}

def get_floating_plot(name, **params):
    '''
    '''
    if not name in floating_plots:
        floating_plots[name] = plt.figure(**{**params, **{'figsize':(6,3)}})
        floating_plots[name].canvas.set_window_title(name)
    return floating_plots[name]

set_up_figure_3d = set_up_3d_plot

def set_up_figure(title='Plot', plot_type='2d+ind'):
    plt.ion()
    plt.show()
    sns.set_theme()
    sns.color_palette("dark")
    if plot_type == '2d+ind':
        fig, ax = plt.subplots(2,1,
                            gridspec_kw={'height_ratios': [4, 1]},
                            figsize=(6, 7.5)
        )
    elif plot_type == '3d_plot':
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111,
                            projection='3d'
        )
    else:
        fig, ax = plt.subplots(1,1,
                            ##gridspec_kw={'height_ratios': [4, 0]},
                            figsize=(6, 6)
        )
    fig.canvas.set_window_title(title)
    return fig, ax

def trace_trajectories(world, fig, ax, fig_title='', indicator_legend=[], trajectory_legend=[]):
    '''
    Performs colored line plots of all particle trajectories in system.
    '''

    full_history = world.get_full_history_with_indicators()

    if world.spatial_dims == 2:
        for i in range(world.n_agents):
            sns.scatterplot(
                x=full_history[i::world.n_agents,3],
                y=full_history[i::world.n_agents,4],
                ax=ax[0],
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
            for j in range(world.get_indicator_history().shape[1]):
                sns.lineplot(
                    x=full_history[world.n_agents::100, 0],
                    y=full_history[world.n_agents::100, 7+j],
                    ax=ax[1],
                    legend=False
                )
        except Exception as e:
            print(e)
    
    elif world.spatial_dims == 3:
        print("ayy")
        for i in range(world.n_agents):
            traj = full_history[i::world.n_agents, worlds.pos[world.spatial_dims]]
            if traj.size()[0] > 1000: step = 100
            else: step = 1
            ax.scatter(
                xs=traj[::step,0],
                ys=traj[::step,1],
                zs=traj[::step,2],
                s=1
            )
    
    fig.canvas.set_window_title(fig_title)
    if trajectory_legend:
        ax[0].legend(loc='lower left', labels=trajectory_legend)
    if indicator_legend:
        ax[1].legend(loc='lower left', labels=indicator_legend)

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
    n_sph = sum(world.context['sph_active'])

    ax.clear()
    """ ax.plot(
        (-25, 25),
        (0, 0),
        c='g'
    ) """

    if h:
        pos = worlds.pos[world.spatial_dims]
        kd_tree = scipy.spatial.cKDTree(state[:,pos])
        h_pairs = kd_tree.query_pairs(r=h*2)
        if h_pairs:
            lines = []
            for pair in h_pairs:
                if pair[0] < n_sph or pair[1] < n_sph:
                    lines.append((state[pair[0],3:5], state[pair[1],3:5]))
            lc = matplotlib.collections.LineCollection(lines, colors='lightgrey')
            ax.add_collection(lc)

    p = sns.scatterplot(
            x=state[:n_sph,3],
            y=state[:n_sph,4],
            c=agent_colors[:n_sph],
            marker='o',
            ##s=agent_sizes,
            ax=ax
    )
    p = sns.scatterplot(
            x=state[n_sph:,3],
            y=state[n_sph:,4],
            c=agent_colors[n_sph:],
            marker='^',
            ##s=agent_sizes,
            ax=ax
    )

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

    ax.set(xlim=(-5, 5), ylim=(-5, 5))

    ax.set_xlabel('orbital plane basis 1 (km)')
    ax.set_ylabel('orbital plane basis 2 (km)')

    ##two_d_video_images.append(ax.get_images()[0])

    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)

def render_3d_orbit_state(world,
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

    n_sph = sum(world.context['sph_active'])
    ax.scatter(
        state[:n_sph,3],
        state[:n_sph,4],
        state[:n_sph,5],
        c=agent_colors[:n_sph]
    )
    ax.scatter(
        state[n_sph:,3],
        state[n_sph:,4],
        state[n_sph:,5],
        c=agent_colors[n_sph:],
        marker='^'
    )
    # p = sns.scatterplot(
    #         x=state[:n_sph,3],
    #         y=state[:n_sph,4],
    #         c=agent_colors[:n_sph],
    #         marker='o',
    #         ##s=agent_sizes,
    #         ax=ax
    # )
    # p = sns.scatterplot(
    #         x=state[n_sph:,3],
    #         y=state[n_sph:,4],
    #         c=agent_colors[n_sph:],
    #         marker='^',
    #         ##s=agent_sizes,
    #         ax=ax
    # )

    if False:#h:
        kd_tree = scipy.spatial.cKDTree(state[:,3:5])
        h_pairs = kd_tree.query_pairs(r=h*2)
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

    ax.set(xlim=(2888-25, 2888+25), ylim=(5890-25, 5890+25), zlim=(1530-25,1530+25))

    ax.set_xlabel('orbital plane basis 1 (km)')
    ax.set_ylabel('orbital plane basis 2 (km)')
    ax.set_zlabel('orbital plane basis 3 (km)')

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

    pos = worlds.pos[world.spatial_dims]

    transformed_world = np.apply_along_axis(
        lambda p: projections.mercator_projection(
            p=p[pos]+world.context['cumulative_recentering'],
            r=orbit_radius),
            1,
            state/scaling_factor)

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
    
    r = h
    for c in range (0, 5):
        #fig, ax = plt.subplots()
        circle1 = plt.Circle((world[c,3], 0), r, color="orange", fill=False)
        p.add_artist(circle1)
        if world[c, 3] == maximum:
            circle2 = plt.Circle((world[c,3], 0), r, color="green", fill=False)
            p.add_artist(circle2)

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