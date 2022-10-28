from ast import parse
from scripts.evolve import _evolve
from scripts.utils import parse_args

def run():
    args = parse_args()
    if args.local_debug:
        args.prefix = 'test11_debug'
        args.config = 'data/config-feedforward-1kpop-1.05Goal'
        args.data_files = 'data/ETHUSDT_15m_data_TA_PCA_futures_2021-12-1_2022-9-30.csv,data/BTCUSDT_15m_data_TA_PCA_futures_2021-12-1_2022-9-30.csv'
        args.fitness_metric = 'median_avg_roi'
        args.output_int = 1
        args.n_inputs = 12
        args.n_outputs = 17
        args.interval = 'daily'
        args.data_interval = '15m'
        args.fitness_threshold = 1.05
        args.hidden_layers = 1
        args.pop_size = 50
        args.max_stag = 20
        args.capital = 1000000
        args.n_cv = 10
        args.btc = True
        args.eth = True
        args.pca = True
    _evolve(args)

if __name__ == '__main__':
    run()
