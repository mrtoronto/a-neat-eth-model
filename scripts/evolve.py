from datetime import datetime
import json
import logging
import os
import neat
import multiprocessing
import pickle
import neat

from scripts.gcs import upload_blob
from scripts.reporter import CustomOutReporter
from scripts.utils import parse_args, _setup_config, _setup_params

from scripts.eval_genomes import eval_genome
from scripts.gspread import insert_to_gsheet

def _evolve(args):
    params = _setup_params(args)
    if not os.path.exists('data'):
        os.mkdir('data')
        os.mkdir(f'data/{params["prefix"]}')
    elif not os.path.exists(f'data/{params["prefix"]}'):
        os.mkdir(f'data/{params["prefix"]}')

    config = _setup_config(params)
    
    with open(f'data/{params["prefix"]}/params.json', 'w') as f:
        json.dump(params, f, indent=4)

    upload_blob(f'data/{params["prefix"]}/params.json')

    insert_to_gsheet(
        "runs", 
        "1F_uQGUbPSpjTw0DCzM0Ls7xu5oe22uQju-mYlDgFe9Y", 
        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            params['prefix'],
            args.config,
            config.capital,
            config.pop_size,
            config.fitness_criterion,
            config.fitness_threshold,
            config.genome_config.num_inputs,
            config.genome_config.num_hidden,
            config.genome_config.num_outputs,
            config.stagnation_config.species_fitness_func,
            config.stagnation_config.max_stagnation,
            config.fitness_metric,
            config.interval,
            params['OUTPUT_INT'], 
            f'{config.training_data.shape!r}',
            f'{config.eval_data.shape!r}',
            config.n_cv,
            multiprocessing.cpu_count()
        ]
    )

    pop = neat.Population(config)
    pop.add_reporter(CustomOutReporter(True,  params['OUTPUT_INT'], params['prefix']))
    print(f'Number of CPUs: {multiprocessing.cpu_count()}')
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open(f'data/{params["prefix"]}/winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    upload_blob(f'data/{params["prefix"]}/winner-feedforward')
    logging.info('won')


if __name__ == "__main__":
    args = parse_args()
    _evolve(args)