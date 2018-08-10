import pickle
import argparse
from genotype import *
from animat import *
from single_tmaze import *
from evolutionary_search import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("experiment.py")
    parser.add_argument("-s", "--src", help = "src pickle for population")
    parser.add_argument("-d", "--dst", help = "dst pickle for population")
    parser.add_argument("-g", "--ng",  help = "number of generations", default = 100)
    parser.add_argument("-p", "--np",  help = "number of population",  default = 300)
    parser.add_argument("-n", "--nn",  help = "number of neurons",     default =   2)
    args = parser.parse_args()

    pop = None
    if args.src is not None:
        with open(args.src, 'rb') as f:
            pop = pickle.load(f)
            print('Load POPULATION from ' + args.src)

    n_neu = int(args.nn) if pop is None else pop[0].n_neuron
    n_pop = int(args.np) if pop is None else len(pop)
    es = EvolutionarySearch(SingleTMaze, Animat, n_neu, n_pop)
    es.pop = es.pop if pop is None else pop
    pop = es.run(int(args.ng))

    best = sorted(pop, key = lambda g:g.fitness)[-1]
    best_animat = Animat(best)
    print(best.fitness)
    print(es.env.evaluate(best_animat))
    print(best_animat.weight)
    print(best_animat.rule)

    if args.dst != None:
        with open(args.dst, 'wb') as f:
            pickle.dump(pop, f)
            print('Dump POPULATION to ' + args.dst)
