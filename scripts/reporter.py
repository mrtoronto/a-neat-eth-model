from datetime import datetime
import os
import pickle
import shutil
import time
import logging
from neat.math_util import mean, stdev
from neat.reporting import BaseReporter
from scripts.gcs import upload_blob
from scripts.gspread import insert_to_gsheet
from scripts.eval_genomes import eval_genome


class CustomOutReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""

    def __init__(self, show_species_detail, DEBUG_INT, prefix, OVERWRITE=True):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.run_start_time = time.time()
        self.generation_times = []
        self.num_extinctions = 0
        self.DEBUG_INT = DEBUG_INT
        self.model_dir = f'data/{prefix}'
        if not os.path.exists('data'):
            os.mkdir('data')
        if OVERWRITE:
            shutil.rmtree(self.model_dir, ignore_errors=True)
        os.mkdir(self.model_dir)

    def start_generation(self, generation):
        self.generation = generation
        if (self.generation % self.DEBUG_INT == 0):
            elapsed_run = time.time() - self.run_start_time
            logging.info(f'\n ****** Running generation {generation} ({elapsed_run:.3f}s) ****** \n')
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        if len(self.generation_times) > 1:
            logging.info(f"Generation {self.generation} time: {elapsed:.3f} sec ({average:.3f} average)")
        else:
            logging.info(f"Generation {self.generation} time: {elapsed:.3f} sec")
            
        if self.show_species_detail and (self.generation % self.DEBUG_INT == 0):
            ng = len(population)
            ns = len(species_set.species)
            logging.info('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            logging.info("   ID   age  size   fitness   adj fit  stag")
            logging.info("  ====  ===  ====  =========  =======  ====")
            for sid in sorted(species_set.species):
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else f"{s.fitness:.3f}"
                af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.3f}"
                st = self.generation - s.last_improved
                logging.info(f"  {sid:>4}  {a:>3}  {n:>4}  {f:>9}  {af:>7}  {st:>4}")
            logging.info('Total extinctions: {0:d}'.format(self.num_extinctions))

            

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        if (self.generation % self.DEBUG_INT == 0):
            fitnesses = [c.fitness for c in population.values()]
            fit_mean = mean(fitnesses)
            fit_std = stdev(fitnesses)
            best_species_id = species.get_species_id(best_genome.key)
            logging.info(f'Population\'s average fitness: {fit_mean:3.5f} stdev: {fit_std:3.5f}')
            logging.info(f'Best fitness: {best_genome.fitness:3.5f} - size: {best_genome.size()!r} - species {best_species_id} - id {best_genome.key}')
            config.training = False
            config.cv = False
            eval_fitness_dict = eval_genome(best_genome, config)
            if config.n_cv > 0:
                config.cv = True
                cv_fitness_dict = eval_genome(best_genome, config)
                config.cv = False
                config.training = True
            else:
                cv_fitness_dict = {
                    'median_roi' :0,
                    'avg_roi': 0,
                    'win_rate': 0,
                    'n_rois': 0
                }
            config.training = True

            with open(f'{self.model_dir}/best_genome_gen{self.generation}-feedforward', 'wb') as f:
                pickle.dump(best_genome, f)

            upload_blob(
                f'{self.model_dir}/best_genome_gen{self.generation}-feedforward'
            )
            insert_to_gsheet(
                "generations", 
                "1F_uQGUbPSpjTw0DCzM0Ls7xu5oe22uQju-mYlDgFe9Y", 
                row = [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    self.model_dir,
                    self.generation,
                    f'{fit_mean:3.5f}',
                    f'{fit_std:3.5f}',
                    f'{best_genome.fitness:3.5f}',
                    f'{best_genome.size()!r}',
                    best_species_id,
                    best_genome.key,
                    eval_fitness_dict['median_roi'] - 1,
                    eval_fitness_dict['avg_roi'] - 1,
                    eval_fitness_dict['win_rate'],
                    eval_fitness_dict['n_rois'],
                    cv_fitness_dict['median_roi'] - 1,
                    cv_fitness_dict['avg_roi'] - 1,
                    cv_fitness_dict['win_rate'],
                    cv_fitness_dict['n_rois'],
                    f'{time.time() - self.generation_start_time:.3f}',
                    f'{time.time() - self.run_start_time:.4f}'
                ]
            )

    def complete_extinction(self):
        self.num_extinctions += 1
        logging.info('All species extinct.')

    def found_solution(self, config, generation, best):
        logging.info('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            logging.info("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        logging.info(msg)