import pandas as pd
import random
from statistics import median
import numpy as np
import neat
from scripts.process_action import _process_action_eth_btc
from scripts.visualize import plot_agent

HOURS_IN_WEEK = 7 * 24

def _calc_nw(balances, eth_price, btc_price):
    return (balances['usd'] + (balances['eth'] * eth_price) + (balances['btc'] * btc_price))

def _reset_balances(balances, CAPITAL):
    balances['eth'] = 0
    balances['btc'] = 0
    balances['usd'] = CAPITAL
    return balances

def _check_reset_balances(config, idx, skip=False):
    if not skip:
        if (idx % config.data_per_interval == 0):
            return True
        else:
            return False
    else:
        return False

def _take_profits(capital, balances, eth_price, btc_price):
    """
    Convert excess networth to USDT, assume 0.01% fee from binance
    """
    nw = _calc_nw(balances, eth_price, btc_price)
    ### If bots made money, this is positive and is the amount we can take before account is reset to `capital``
    ### If bots lost money, this is negative and removes the amount we need from previous profits to reup to `capital`
    profits_or_losses = (nw - capital)
    if profits_or_losses > 0:
        # If wins, remove fees necessary to convert to USDT
        profits_or_losses = profits_or_losses * 0.999
    else:
        # If loses, remove fees necessary to rebalance wallet
        profits_or_losses = profits_or_losses - (abs(profits_or_losses) * 0.001)
    balances['profit'] += profits_or_losses

    balances = _reset_balances(balances, capital)
    return balances

def _calc_fitness(config, rois, daily_balances):
    if config.fitness_metric == 'avg_roi':
        ### Outliers make lucky agents seem more fit than they should be
        return max(0, (sum(rois) / len(rois)))
    elif config.fitness_metric == 'median_roi':
        ### Fitness function doesn't reward luck as much as it should
        return max(0, median(rois))
    elif config.fitness_metric == 'median_avg_roi':
        ### Reward luck and consistency
        ### Rewarding luck will teach agents to exploit lucky discoveries
        ### Consistency is important in the real world
        median_roi = median(rois)
        avg_roi = (sum(rois) / len(rois))
        if median_roi < 0 and avg_roi < 0:
            return -1 * median_roi * avg_roi
        else:
            return median_roi * avg_roi
    elif config.fitness_metric == 'profit':
        ### Profit taken
        ### Rewards lucky intervals too much
        return daily_balances[-1]['profit'] / config.capital
    else:
        print('UNKNOWN FITNESS METRIC')
        return None

def _make_inputs(config, balances, feats):
    
    if config.eth and config.btc:
        inputs = feats + [balances['usd'], balances['eth'], balances['btc']]
    elif config.eth:
        inputs = feats + [balances['usd'], balances['eth']]
    elif config.btc:
        inputs = feats + [balances['usd'], balances['btc']]
    else:
        print('THERES A PROBLEM')
        return None
    return inputs


def _run_sim_loop(config, net, rois, daily_balances=[]):
    balances = {'usd': 0, 'eth': 0, 'btc': 0, 'fitness': 0, 'profit': 0}

    for idx, row in config.data.iterrows():
        eth_price = config.prices[row['datetime']]['eth']
        btc_price = config.prices[row['datetime']]['btc']
        if _check_reset_balances(config, idx):
            balances = _reset_balances(balances, config.capital)
            yester_nw = _calc_nw(balances, eth_price, btc_price)


        inputs = _make_inputs(config, balances, row.iloc[1:].tolist())
        action = net.activate(inputs)
        balances = _process_action_eth_btc(action, balances, eth_price, btc_price)

        daily_balances.append({
            'action': np.argmax(action),
            'eth': balances['eth'],
            'btc': balances['btc'],
            'usd': balances['usd'],
            'ETH_price': eth_price,
            'BTC_price': btc_price,
            'datetime': row['datetime'],
            'fitness': balances['fitness'],
            'profit': balances['profit']
        })

        # Last hour of the week
        if (idx % config.data_per_interval == config.data_per_interval - 1):
            nw = _calc_nw(balances, eth_price, btc_price)
            rois.append((nw / yester_nw))
            yester_nw = nw
            balances = _take_profits(config.capital, balances, eth_price, btc_price)
            if config.cv:
                return rois, daily_balances

    if len(rois) == 0:
        nw = _calc_nw(balances, eth_price, btc_price)
        rois.append((nw / yester_nw))
        yester_nw = nw
        balances = _take_profits(config.capital, balances, eth_price, btc_price)

    return rois, daily_balances

def _run_fitness_sim(genome, config, plot, filename, debug, eval_run):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    rois = []

    if config.cv:
        daily_balances = []

        ### With a shifted time, each data point will be new bc balances will be different than training
        # config.data = config.training_features
        for n in range(config.n_cv):
            rand_int = random.randint(
                int((n/config.n_cv) * config.data_per_interval), 
                int(((n + 1)/config.n_cv) * config.data_per_interval)
            )

            config.data = config.training_features.iloc[rand_int:,:].copy()
            config.data = config.data.reset_index(drop=True)
            rois, daily_balances = _run_sim_loop(config, net, rois, daily_balances)
            config.data = None
    else:
        rois, daily_balances = _run_sim_loop(config, net, rois)

    if plot:
        plot_agent(daily_balances, rois=rois, filename=filename, config=config)
    
    if debug:
        pd.DataFrame(daily_balances).to_csv(f'data/debug_actions.tsv', sep='\t')

    if eval_run:
        config.fitness_metric = 'median_roi'
        median_roi = _calc_fitness(config, rois, daily_balances)
        config.fitness_metric = 'avg_roi'
        avg_roi = _calc_fitness(config, rois, daily_balances)
        return {
            'median_roi': median_roi,
            'avg_roi': avg_roi,
            'n_rois': len(rois),
            'win_rate': len([r for r in rois if r > 1]) / len(rois)
        }

    else:
        return _calc_fitness(config, rois, daily_balances)


def eval_genome(genome, config, plot=False, filename=None, debug=False):
    
    if config.training == True:
        config.cv = False
        config.data = config.training_features
        fitness = _run_fitness_sim(genome, config, plot, filename, debug, eval_run=False)
    else:
        config.data = config.eval_features
        fitness = _run_fitness_sim(genome, config, plot, filename, debug, eval_run=True)
    
    config.data = None
    return fitness


def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)