import pickle
import pandas as pd
from scripts.gcs import download_blob, upload_blob
from scripts.evolve import eval_genome, _setup_config
from scripts.utils import _add_data_to_config
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import neat
import os
import json
from dateutil import parser

def eval_agent(config, params, filename='/tmp/genome', print_daily=False):
    with open(filename, 'rb') as f:
        winner = pickle.load(f)
    if not params.get('test_data'):
        config.training = True
        eval_genome(winner, config, plot=True, filename=filename)
    
    config.training = False
    eval_genome(winner, config, plot=True, filename=filename, debug=True)

    config.training = False
    config.cv = True
    config.n_cv = 100
    config.data = config.training_features
    eval_genome(winner, config, plot=True, filename=filename, debug=True)


def download_and_eval_agent(prefix, gen, test_data=None, capital=None, interval=None):

    download_blob(f'data/{prefix}/params.json', '/tmp/params.json')

    with open('/tmp/params.json', 'r') as f:
        params = json.load(f)
    if interval:
        params['interval'] = interval
    if capital:
        params['capital'] = capital
    
    if test_data:
        params['test_data'] = True
        params['data_files'] = test_data
    else:
        params['test_data'] = False

    config = _setup_config(params)
    download_blob(f'data/{prefix}/best_genome_gen{gen}-feedforward', '/tmp/genome')
    eval_agent(config, params)
    
