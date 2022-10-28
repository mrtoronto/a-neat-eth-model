from statistics import median
import copy
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# import graphviz
import matplotlib.pyplot as plt
import numpy as np

HOURS_IN_WEEK = 7 * 24

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    # plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    t_values = [t for t, I, v, u in spikes]
    v_values = [v for t, I, v, u in spikes]
    u_values = [u for t, I, v, u in spikes]
    I_values = [I for t, I, v, u in spikes]

    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(3, 1, 2)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(3, 1, 3)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


# def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
#              node_colors=None, fmt='svg'):
#     """ Receives a genome and draws a neural network with arbitrary topology. """
#     # Attributes for network nodes.
#     if graphviz is None:
#         warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
#         return

#     # If requested, use a copy of the genome which omits all components that won't affect the output.
#     if prune_unused:
#         if show_disabled:
#             warnings.warn("show_disabled has no effect when prune_unused is True")

#         genome = genome.get_pruned_copy(config.genome_config)

#     if node_names is None:
#         node_names = {}

#     assert type(node_names) is dict

#     if node_colors is None:
#         node_colors = {}

#     assert type(node_colors) is dict

#     node_attrs = {
#         'shape': 'circle',
#         'fontsize': '9',
#         'height': '0.2',
#         'width': '0.2'}

#     dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

#     inputs = set()
#     for k in config.genome_config.input_keys:
#         inputs.add(k)
#         name = node_names.get(k, str(k))
#         input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
#         dot.node(name, _attributes=input_attrs)

#     outputs = set()
#     for k in config.genome_config.output_keys:
#         outputs.add(k)
#         name = node_names.get(k, str(k))
#         node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

#         dot.node(name, _attributes=node_attrs)

#     for n in genome.nodes.keys():
#         if n in inputs or n in outputs:
#             continue

#         attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
#         dot.node(str(n), _attributes=attrs)

#     for cg in genome.connections.values():
#         if cg.enabled or show_disabled:
#             input, output = cg.key
#             a = node_names.get(input, str(input))
#             b = node_names.get(output, str(output))
#             style = 'solid' if cg.enabled else 'dotted'
#             color = 'green' if cg.weight > 0 else 'red'
#             width = str(0.1 + abs(cg.weight / 5.0))
#             dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

#     dot.render(filename, view=view)
#     return dot


def _get_action_colors(daily_balances):
    return [
        '#000' if i['action'] == 8 else \
        '#808080' if (i['action'] % 2 == 0 and i['action'] < 8) else \
        '#808080' if (i['action'] % 2 == 1 and i['action'] > 8) else \
        '#DAA520' for i in daily_balances
    ]

def plot_agent(daily_balances, rois, filename, config):
    
    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(20,18), gridspec_kw={'height_ratios': [1, 1, 1, 1, 2]})
    fig.set_facecolor('white')

    dates = mdates.date2num([i['datetime'] for i in daily_balances])
    eth_assets_in_usd = [(i['eth'] * i['ETH_price'] / config.capital) for i in daily_balances]
    eth_btc_assets_in_usd = [(i['eth'] * i['ETH_price'] / config.capital) + (i['btc'] * i['BTC_price'] / config.capital) for i in daily_balances]
    all_in_usd = [(i['eth'] * i['ETH_price'] / config.capital) + (i['btc'] * i['BTC_price'] / config.capital) + (i['usd'] / config.capital) for i in daily_balances]

    axs[0].plot(dates, [i['profit'] / config.capital for i in daily_balances], label='profit', color='green')
    axs[1].plot(dates, [(i['usd'] + (i['eth'] * i['ETH_price']) + (i['btc'] * i['BTC_price'])) / config.capital for i in daily_balances], label='Agent Net Worth', color='red')
    axs[2].fill_between(dates, 0, eth_assets_in_usd, label='ETH', color='grey')
    axs[2].fill_between(dates, eth_assets_in_usd, eth_btc_assets_in_usd, label='BTC', color='goldenrod')
    axs[2].fill_between(dates, eth_btc_assets_in_usd, all_in_usd, label='USD', color='green')
    axs[3].plot(dates, [i['ETH_price'] for i in daily_balances], label='ETH price', color='grey', linewidth="2")
    axs_3_twinx = axs[3].twinx()
    axs_3_twinx.plot(dates, [i['BTC_price'] for i in daily_balances], label='BTC price', color='goldenrod', linewidth="2")
    axs[4].scatter(
        dates, 
        [i['action'] for i in daily_balances], 
        label='action', 
        marker='o',
        c=_get_action_colors(daily_balances)
    )

    axs[0].set_title(f'Profit Taken')
    axs[1].set_title('Agent Net worth')
    axs[2].set_title('Assets')
    start_eth_price = daily_balances[0]['ETH_price']
    end_eth_price = daily_balances[-1]['ETH_price']
    eth_price_change = 100 * ((end_eth_price / start_eth_price) - 1)
    start_btc_price = daily_balances[0]['BTC_price']
    end_btc_price = daily_balances[-1]['BTC_price']
    btc_price_change = 100 * ((end_btc_price / start_btc_price) - 1)
    axs[3].set_title(
        f'ETH Price - (${start_eth_price} - ${end_eth_price} | {eth_price_change:.3f}%) --- '
        f'BTC Price - (${start_btc_price} - ${end_btc_price} | {btc_price_change:.3f}%)'
    )
    axs[4].set_title('Agent Action')

    for ax in [0, 1, 2, 3, 4]:
        axs[ax].grid()
        for i in range(((config.data.shape[0] // config.data_per_interval) + 1)):
            if i * config.data_per_interval == len(dates):
                axs[ax].axvline(x=dates[-1])
            else:
                axs[ax].axvline(x=dates[i * config.data_per_interval])

    profit = daily_balances[-1]['profit']
    roi = profit / config.capital
    if config.data_interval == '15m':
        scaler = 4
    else:
        scaler = 1

    if config.interval == 'daily':
        periods_in_data = int(len(dates) // (24 * scaler))
        last_title_line = f"After {periods_in_data} days"
    else:
        periods_in_data = int(len(dates) // (24 * 7 * scaler))
        last_title_line = f"After {periods_in_data} weeks"


    last_title_line += f" - Given ${config.capital:,} principle"
    
    fig.suptitle(f"{filename} - ROI (profit): {roi * 100:.3f}% (${profit:,.3f})\n" + \
    f"Average {config.interval} ROI: {(np.mean(rois) - 1) * 100:.3f}% ||| Median {config.interval} ROI: {(np.median(rois) - 1) * 100:.3f}%\n" + \
    f"Positive periods: {len([r for r in rois if r > 1])} ||| Negative periods: {len([r for r in rois if r < 1])}\n" +
    last_title_line, y=0.98)
    # f"{end_usd:,.3f} ||| Eth: \${end_eth_usd:,.3f} ({end_eth:,.3f}eth) \n"
    # f"Net-worth: {net_worth:,.3f}", y=0.98)
    fig.legend()

    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])
    axs[2].set_xticklabels([])
    axs[3].set_xticklabels([])

    axs[4].set_yticks(range(0,17))
    axs[4].set_ylim((17, -1))
    axs[4].set_yticklabels([
        'Buy ETH - 20% of USD', 
        'Buy BTC - 20% of USD', 
        'Buy ETH - 10% of USD', 
        'Buy BTC - 10% of USD', 
        'Buy ETH - 5% of USD', 
        'Buy BTC - 5% of USD', 
        'Buy ETH - 2.5% of USD', 
        'Buy BTC - 2.5% of USD', 
        'No Action', 
        'Sell ETH - 2.5% of USD', 
        'Sell BTC - 2.5% of USD', 
        'Sell ETH - 5% of USD', 
        'Sell BTC - 5% of USD', 
        'Sell ETH - 10% of USD', 
        'Sell BTC - 10% of USD', 
        'Sell ETH - 20% of USD', 
        'Sell BTC - 20% of USD', 
    ])

    axs[4].xaxis_date()
    fig.autofmt_xdate()
    plt.show()