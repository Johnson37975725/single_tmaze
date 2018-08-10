import numpy as np
import copy as cp
from genotype import *

class EvolutionarySearch:
    def __init__(self, env_cls, phe_cls, n_neuron, n_pop, n_group = 5):
        self.env      = env_cls()
        self.phe_cls  = phe_cls
        self.n_input  = self.env.n_input
        self.n_neuron = n_neuron
        self.n_pop    = n_pop
        self.n_group  = n_group
        self.pop      = np.array([Genotype(self.n_input, n_neuron) for i in range(n_pop)])
        self.n_eval   = 2

    def run(self, n_generation, print_stats = True):
        for i in range(n_generation):
            self.evaluation(self.pop, self.n_eval)
            if print_stats: print(', '.join([str(x) for x in [i] + self.stats(self.pop)]))
            if (i+1) == n_generation: break
            self.pop = self.mutation(self.crossover(self.selection(self.pop, self.n_pop, self.n_group)))
        return self.pop

    def stats(self, pop): # return list object not numpy object
        return [round(x, 5) for x in self.fit_stats(pop) + self.best_rule(pop) + self.plastic_utility(pop)]

    def fit_stats(self, pop): # return a list object not numpy object
        fits = [g.fitness for g in pop]
        return [f(fits) for f in [np.mean, np.median, np.std, max, min]]

    def best_rule(self, pop): # return a list object not numpy object
        best = sorted(pop, key = lambda g:g.fitness)[-1]
        return self.phe_cls(best).rule.tolist()

    def plastic_utility(self, pop): # return a list object not numpy object
        best = sorted(pop, key = lambda g:g.fitness)[-1]
        pos  = np.mean([self.env.evaluate(self.phe_cls(best)) for i in range(5)])
        neg  = np.mean([self.env.evaluate(self.phe_cls(best, False)) for i in range(5)])
        return [pos, neg]

    def evaluation(self, pop, n_eval): # I feel that this is an awkward function
        for g in pop: g.fitness = np.mean([self.env.evaluate(self.phe_cls(g)) for i in range(n_eval)])

    def selection(self, pop, n_pop, n_group):
        lpop   = pop.tolist()
        offset = np.random.randint(n_group)
        head   = self.best_proliferate(lpop[(offset - n_group):] + lpop[:offset])
        index  = np.arange(n_pop / n_group - 1, dtype = 'i8') * n_group + offset
        return np.array(head + sum([self.best_proliferate(lpop[i:(i + n_group)]) for i in index], []))

    def best_proliferate(self, group):
        best = sorted(group, key = lambda g:g.fitness)[-1]
        return [cp.deepcopy(best) for i in range(len(group))]

    def crossover(self, pop, c_rate = 0.1):
        G = lambda : pop[np.random.randint(len(pop))]
        M = lambda g: cross_genotype((g, G())) if np.random.rand() < c_rate else cp.deepcopy(g)
        best = sorted(pop, key = lambda g:g.fitness)[-1]
        return np.array([M(g) if g != best else cp.deepcopy(best) for g in pop])

    def mutation(self, pop):
        best = sorted(pop, key = lambda g:g.fitness)[-1]
        return np.array([mutate_genotype(g) if g != best else cp.deepcopy(best) for g in pop])

if __name__ == '__main__':
    '''
    Some tests
    '''

    def assert_evolutionary_search(condition, msg = 'NG in "evolutionary_search.py"'):
        assert condition, msg

    class EnvMock:
        def __init__(self):
            self.n_input = 5
            self.ret_val = None

        def evaluate(self, phenotype):
            assert isinstance(phenotype, PheMock), 'NG in "evolutionary_search.py"'
            if self.ret_val is not None: return self.ret_val
            return (2 if phenotype.plastic else 3) * phenotype.fitness

    class PheMock:
        def __init__(self, genotype_, plastic = True):
            assert isinstance(genotype_, Genotype), 'NG in "evolutionary_search.py"'
            self.plastic = plastic
            self.fitness = genotype_.fitness
            self.rule    = np.ones(5) * genotype_.fitness

    n_neu =  4
    n_pop = 25
    n_grp =  5
    es = EvolutionarySearch(EnvMock, PheMock, n_neu, n_pop, n_grp)

    # tests for fit_stats
    for i in range(n_pop): es.pop[i].fitness = i
    result = es.fit_stats(es.pop)
    assert_evolutionary_search(isinstance(result, list))
    assert_evolutionary_search(len(result) == 5)

    # tests for best_rule
    fits = np.random.rand(n_pop)
    for i in range(n_pop): es.pop[i].fitness = fits[i]
    rule = es.best_rule(es.pop)
    assert_evolutionary_search(isinstance(rule, list))
    assert_evolutionary_search(np.allclose(rule, np.ones(len(rule)) * max(fits)))

    # tests for plastic_utility
    fits = np.random.rand(n_pop)
    for i in range(n_pop): es.pop[i].fitness = fits[i]
    result = es.plastic_utility(es.pop)
    assert_evolutionary_search(isinstance(result, list))
    assert_evolutionary_search(np.allclose(result, [max(fits) * 2, max(fits) * 3]))

    # a test for evaluation
    fits = np.random.rand(n_pop)
    for i in range(n_pop): es.pop[i].fitness = fits[i]
    es.evaluation(es.pop, 3)
    assert_evolutionary_search(np.allclose([g.fitness for g in es.pop], fits * 2))

    # tests for selection
    fits = np.random.rand(n_pop)
    for i in range(n_pop): es.pop[i].fitness = fits[i]
    result = es.selection(es.pop, n_pop, n_grp)

    assert_evolutionary_search(len(np.unique([g.fitness for g in result])) == n_pop / n_grp)

    ids = [id(g) for g in result]
    assert_evolutionary_search(len(np.unique(ids)) == n_pop)
    assert_evolutionary_search(len(list(set(ids) & set([id(g) for g in es.pop]))) == 0)

    # tests for best_proliferate
    fits = np.random.rand(n_grp)
    grp  = es.pop[0:n_grp]
    for i in range(n_grp): grp[i].fitness = fits[i]
    result = es.best_proliferate(grp)
    assert_evolutionary_search(len(result) == n_grp)
    assert_evolutionary_search(np.all(np.array([g.fitness for g in result]) == max(fits)))
    assert_evolutionary_search(not np.all(np.array([id(g) for g in result]) == id(grp[np.argmax(fits)])))

    # tests for crossover
    fits = np.random.rand(n_pop)
    for i in range(n_pop): es.pop[i].fitness = fits[i]
    result = es.crossover(es.pop)

    ids = [id(g) for g in result]
    assert_evolutionary_search(len(list(set(ids) & (set([id(g) for g in es.pop])))) == 0)

    best  = es.pop[np.argmax(fits)]
    best_ = result[np.argmax(fits)]
    assert_evolutionary_search(np.all(best.weight == best_.weight))
    assert_evolutionary_search(np.all(best.rule == best_.rule))

    # tests for mutation
    fits = np.random.rand(n_pop)
    for i in range(n_pop): es.pop[i].fitness = fits[i]
    result = es.mutation(es.pop)

    ids = [id(g) for g in result]
    assert_evolutionary_search(len(list(set(ids) & (set([id(g) for g in es.pop])))) == 0)

    best  = es.pop[np.argmax(fits)]
    best_ = result[np.argmax(fits)]
    assert_evolutionary_search(np.all(best.weight == best_.weight))
    assert_evolutionary_search(np.all(best.rule == best_.rule))
