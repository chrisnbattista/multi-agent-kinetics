





import matplotlib.pyplot as plt
import seaborn as sns





def set_up_figure():
    plt.ion()
    plt.show()
    sns.set_theme()
    return plt.subplots(2,1,
                        gridspec_kw={'height_ratios': [4, 1]},
                        figsize=(7.5, 9)
    )

def render_state(world, fig, ax, show_indicators=False, indicators=None, indicator_labels=None, fig_title=''):
    '''
    Display the particles described in the world array onto the figure and axis provided.
    Activates the plt event loop so that the figure is displayed.
    Also, display the indicator timeseries below if show_indicators=True.

    Expects:
        world object with only one timestep of data
        plt figure
        axes object with one axis if show_indicators=False, two axes otherwise
    '''

    ax[0].clear()

    p = sns.scatterplot(
            x=world[:,1],
            y=world[:,2],
            color='k',
            ax=ax[0]
    )

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
    
    fig.canvas.set_window_title(fig_title)
    # if not len(fig.texts):
    #     fig.text(0.01, 0.01, note)
    # else:
    #     fig.texts[0].set_text(note)

    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)