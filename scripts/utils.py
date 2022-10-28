import pandas as pd
import neat
from scripts.gcs import download_blob
from config.local_settings import firestore_creds
import json

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='eth neat')
	parser.add_argument('--local_debug', action='store_true', help='')
	parser.add_argument('--prefix', help='')
	parser.add_argument('--config', help='')
	parser.add_argument('--data_files', help='')
	parser.add_argument('--capital', help='')
	parser.add_argument('--output_int', help='', type=int)
	parser.add_argument('--interval', help='')
	parser.add_argument('--data_interval', help='')
	parser.add_argument('--pca', action='store_true', help='')
	parser.add_argument('--btc', help='', action='store_true')
	parser.add_argument('--eth', help='', action='store_true')
	parser.add_argument('--pop_size', help='')
	parser.add_argument('--max_stag', help='')
	parser.add_argument('--n_cv', help='', type=int)
	parser.add_argument('--fitness_metric', help='', type=str)
	parser.add_argument('--fitness_threshold', help='', type=float)
	parser.add_argument('--n_inputs', help='', type=int)
	parser.add_argument('--n_outputs', help='', type=int)
	parser.add_argument('--hidden_layers', help='', type=int)
	parser.add_argument('--job-dir',help='',default='')
	args = parser.parse_args()
	return args

def _setup_params(args):
	params = {
        'OUTPUT_INT': args.output_int,
        'n_inputs': args.n_inputs,
        'n_outputs': args.n_outputs,
        'hidden_layers': args.hidden_layers,
        'prefix': args.prefix,
        'data_files': args.data_files,
        'config_file': args.config,
        'fitness_metric': args.fitness_metric,
        'fitness_threshold': args.fitness_threshold,
        'capital': args.capital,
        'pop_size': args.pop_size,
        'max_stag': args.max_stag,
        'interval': args.interval,
		'data_interval': args.data_interval,
        'test_data': False,
        'btc': args.btc,
        'eth': args.eth,
		'pca': args.pca,
        'n_cv': args.n_cv
    }
	return params

def _get_feats(config, data):
	if config.pca:
		pca_cols = [c for c in data.columns if 'PCA' in c or (c == 'datetime')]
		return data.loc[:, pca_cols]
	else:
		return data


def _add_data_to_config(config, params, data):
	if config.interval == 'weekly':
		period_intervals = 7 * 24
		eval_periods = 12
	elif config.interval == 'daily':
		period_intervals = 24
		eval_periods = 30

	eval_offset = period_intervals * eval_periods
	if config.data_interval == '15m':
		eval_offset = eval_offset * 4

	print(f'Full data shape: {data.shape}')
	print(f'First date in data: {data["datetime"].iloc[0]}')
	print(f'Last date in data: {data["datetime"].iloc[-1]}')

	eval_data = data.iloc[-(eval_offset):]
	eval_data = eval_data.reset_index(drop=True)
	config.eval_data = eval_data
	config.eval_data = config.eval_data.sort_values(by="datetime")
	print(f'Eval data shape: {config.eval_data.shape}')
	print(f'First date in eval data: {config.eval_data["datetime"].iloc[0]}')
	print(f'Last date in eval data: {config.eval_data["datetime"].iloc[-1]}')

	if not params.get('test_data'):
		training_data = data.iloc[:-(eval_offset)]
		training_data = training_data.reset_index(drop=True)
		config.training_data = training_data
		config.training_data = config.training_data.sort_values(by="datetime")
		print(f'Training data shape: {config.training_data.shape}')
		print(f'First date in training data: {config.training_data["datetime"].iloc[0]}')
		print(f'Last date in training data: {config.training_data["datetime"].iloc[-1]}')

	
	config.training_features = pd.DataFrame(_get_feats(config, config.training_data))
	config.training_features = config.training_features.rename({0: 'datetime'}, axis=1)
	config.eval_features = pd.DataFrame(_get_feats(config, config.eval_data))
	config.eval_features = config.eval_features.rename({0: 'datetime'}, axis=1)

	print(f'Training features shape: {config.training_features.shape}')
	print(f'Eval features shape: {config.eval_features.shape}')

	config.prices = {dt: {'btc': btc, 'eth': eth} for (dt, btc, eth) in zip(data['datetime'], data['BTC_price'], data['ETH_price'])}

	return config


def _setup_config(params):
	download_blob(params['config_file'], f'/tmp/neat-config')
	config = neat.Config(
		neat.DefaultGenome, 
		neat.DefaultReproduction,
		neat.DefaultSpeciesSet, 
		neat.DefaultStagnation,
		f'/tmp/neat-config'
	)
	
	config.pop_size = params['pop_size']
	config.fitness_metric = params['fitness_threshold']
	config.genome_config.num_inputs = params['n_inputs']
	config.genome_config.num_outputs = params['n_outputs']
	config.genome_config.num_hidden = params['hidden_layers']
	config.stagnation_config.max_stagnation = params['max_stag']
	
	### Update local config so workers will get updates
	config.save('/tmp/neat-config')

	config = neat.Config(
		neat.DefaultGenome, 
		neat.DefaultReproduction,
		neat.DefaultSpeciesSet, 
		neat.DefaultStagnation,
		f'/tmp/neat-config'
	)
	
	if params.get("test_data"):
		print(f'Loading data files: {params.get("test_data")}')

		config.training = False

		datas = []
		for file_idx, file in enumerate(params['data_files'].split(',')):
			file = file.strip()
			tmp_data = pd.read_csv(file)

			if file_idx > 0:
				tmp_data = tmp_data.drop(('datetime'), axis=1)
			datas.append(tmp_data)
		all_data = pd.concat(datas, axis=1)
		
	else:		
		datas = []
		config.training = True
		for file_idx, file in enumerate(params['data_files'].split(',')):
			file = file.strip()
			download_blob(file, '/tmp/tmp_data.csv')
			tmp_data = pd.read_csv('/tmp/tmp_data.csv')
			if file_idx > 0:
				tmp_data = tmp_data.drop(('datetime'), axis=1)
			datas.append(tmp_data)
		all_data = pd.concat(datas, axis=1)
	
	config.interval = params['interval']
	config.data_interval = params['data_interval']
	
	if config.interval == 'daily':
		if config.data_interval == '15m':
			config.data_per_interval = 24 * 4
		else:
			config.data_per_interval = 24
	elif config.interval == 'weekly':
		if config.data_interval == '15m':
			config.data_per_interval = 24 * 4 * 7
		else:
			config.data_per_interval = 24 * 7
	config.pca = params.get('pca')

	all_data.to_csv(f"/tmp/hourly_data_TA.csv", index=False)
	data = pd.read_csv("/tmp/hourly_data_TA.csv")
	valid_dt_mask = [True if i == i else False for i in data['datetime']]
	data = data.loc[valid_dt_mask, :]
	config = _add_data_to_config(config, params, data)
	config.fitness_metric = params['fitness_metric']
	config.capital = int(params['capital'])
	config.eth = params['eth']
	config.btc = params['btc']
	config.cv = False
	config.n_cv = params.get('n_cv', 0)

	print('Config details:')
	print(f'Pop size: {config.pop_size}')
	print(f'Fitness Criterion: {config.fitness_criterion}')
	print(f'Fitness threshold: {config.fitness_threshold}')
	print(f'Number of inputs: {config.genome_config.num_inputs}')
	print(f'Number of hidden layers: {config.genome_config.num_hidden}')
	print(f'Number of outputs: {config.genome_config.num_outputs}')
	print(f'Species fitness function: {config.stagnation_config.species_fitness_func}')
	print(f'Max stagnation: {config.stagnation_config.max_stagnation}')
	print(f'Fitness Metric: {config.fitness_metric}')
	print(f'Taking profit interval: {config.interval}')
	print(f'Data interval: {config.data_interval}')
	print(f'Data per PT interval: {config.data_per_interval}')

	return config