import numpy as np
import copy as cp

def cross_weight(parents):
    r, c = parents[0].weight.shape
    p1, p2 = np.random.randint(r), np.random.randint(c) # crossing points
    I = np.array([[1 if i < p1 or j < p2 else 0 for j in range(c)] for i in range(r)])
    return I * parents[0].weight + np.abs(I - 1) * parents[1].weight

def cross_rule(parents):
    I = np.random.randint(0, 2, 5)
    return I * parents[0].rule + np.abs(I - 1) * parents[1].rule

def cross_genotype(parents):
    child = Genotype(parents[0].n_input, parents[0].n_neuron)
    child.weight = cross_weight(parents)
    child.rule   = cross_rule(parents)
    return child

def mutate_genotype(subject, m_rate = 0.1, sigma = 0.3):
    mutated = cp.deepcopy(subject)
    r, c = subject.weight.shape
    mutated.weight += (np.random.rand(r, c) < m_rate) * np.random.randn(r, c) * sigma
    mutated.rule   += (np.random.rand(5)    < m_rate) * np.random.randn(5)    * sigma
    return mutated

class Genotype:
    def __init__(self, n_input, n_neuron = False):
        # define the size of neural network
        self.n_input  = n_input
        self.n_neuron = n_neuron or self.rand_n_neuron()

        # define genotypic weight matrix
        r = self.n_neuron
        c = self.n_input + self.n_neuron
        self.sg_rate = 1.0 # sg_rate : synaptogenesis rate
        self.weight  = self.initialise_weight(r, c, self.sg_rate)

        # define genotypic learning rule
        self.rule = np.random.rand(5) * 3.0 - 1.5

        # At first, fitness should be zero
        self.fitness = 0.0

    def rand_n_neuron(self, ulim = 20):
        return int(np.random.rand() * (ulim - 1)) + 1

    def initialise_weight(self, row, col, sg_rate):
        return (np.random.rand(row, col) < sg_rate) * (np.random.rand(row, col) * 2.0 - 1.0)

if __name__ == '__main__':
    '''
    Some tests
    '''

    def assert_genotype(condition, msg = "NG in 'genotype.py'"):
        assert condition, msg

    # tests for attributs of Genotype
    g = Genotype(5, 5)
    assert_genotype(g.n_input == 5)
    assert_genotype(g.n_neuron == 5)
    assert_genotype(g.weight.shape == (5, 10))
    assert_genotype(len(g.rule) == 5)
    assert_genotype(g.fitness == 0.0)

    gs = np.array([Genotype(5) for i in range(100)])
    assert_genotype(min([x.n_neuron for x in gs]) >   0)
    assert_genotype(max([x.n_neuron for x in gs]) <= 20)

    assert_genotype(min([np.min(x.weight) for x in gs]) >= -1.0)
    assert_genotype(max([np.max(x.weight) for x in gs]) <=  1.0)

    assert_genotype(min([x.rule[0] for x in gs]) >= -3.0)
    assert_genotype(max([x.rule[0] for x in gs]) <=  3.0)

    assert_genotype(min([np.min(x.rule[1:5]) for x in gs]) >= -1.0)
    assert_genotype(max([np.max(x.rule[1:5]) for x in gs]) <=  1.0)

    # tests for cross_weight & cross_rule & cross_genotype
    g1, g2 = Genotype(5, 5), Genotype(5, 5)

    w = cross_weight([g1, g2])
    assert_genotype(np.any(w != g1.weight) or np.any(w != g2.weight))
    assert_genotype(np.all((w == g1.weight) + (w == g2.weight)))

    r = cross_rule([g1, g2])
    assert_genotype(np.any(r != g1.rule) or np.any(r != g2.rule))
    assert_genotype(np.all((r == g1.rule) + (r == g2.rule)))

    g = cross_genotype([g1, g2])
    assert_genotype(g.n_input == 5)
    assert_genotype(g.n_neuron == 5)

    # tests for mutate_genotype
    g = Genotype(5, 5)
    assert_genotype(not np.array_equal(g.weight, mutate_genotype(g, m_rate = 1.0).weight))
    assert_genotype(not np.array_equal(g.rule, mutate_genotype(g, m_rate = 1.0).rule))

    assert_genotype(np.array_equal(g.weight, mutate_genotype(g, sigma = 0.0).weight))
    assert_genotype(np.array_equal(g.rule, mutate_genotype(g, sigma = 0.0).rule))

    assert_genotype(np.array_equal(g.weight, mutate_genotype(g, m_rate = 0.0).weight))
    assert_genotype(np.array_equal(g.rule, mutate_genotype(g, m_rate = 0.0).rule))
