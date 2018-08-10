import numpy as np

class Phenotype:
    def __init__(self, genotype):
        self.n_input  = genotype.n_input
        self.n_neuron = genotype.n_neuron
        self.weight   = self.transform_weight(genotype.weight)
        self.rule     = self.transform_rule(genotype.rule)

    def transform_weight(self, genotypic_weight, ulim = 10.0, llim = 0.01):
        return np.vectorize(self.transform)(genotypic_weight, ulim, llim)

    def transform_rule(self, genotypic_rule, ulim = 30.0, llim = 0.1):
        rule = np.vectorize(self.transform)(genotypic_rule, ulim, llim)
        # The followings are to facilitate the evolution of plastic neural networks
        rule[0] += 0.0 if rule[0] != 0.0 else (np.random.rand() + 0.5) * np.sign(np.random.randn())
        if np.all(rule[1:5] == 0.0): rule[1:5] = (np.random.rand(4) + 0.5) * np.sign(np.random.randn(4))
        return rule

    def transform(self, g_x, ulim, llim):
        x = g_x ** 3.0
        return np.sign(x) * ulim if abs(x) > ulim else (x if abs(x) > llim else 0.0)

    def perform(self, input):
        assert False, "'perform' must be implemented"

if __name__ == '__main__':
    '''
    Some tests
    '''

    def assert_phenotype(condition, msg = "NG in 'phenotype.py'"):
        assert condition, msg

    class Mock:
        def __init__(self):
            self.n_input  = 3
            self.n_neuron = 5
            self.weight   = np.random.rand(5, 8) * 2.0 + 1.0
            self.rule     = np.random.rand(5) * 10.0 - 5.0

    genotype = Mock()
    p = Phenotype(genotype)

    # general tests
    assert_phenotype(p.n_input == genotype.n_input)
    assert_phenotype(p.n_neuron == genotype.n_neuron)
    assert_phenotype(p.weight.shape == genotype.weight.shape)
    assert_phenotype(len(p.rule) == len(genotype.rule))

    # tests for transform
    assert_phenotype(p.transform( 2.0, 10.0, 0.01) ==   8.0)
    assert_phenotype(p.transform( 3.0, 10.0, 0.01) ==  10.0)
    assert_phenotype(p.transform(-2.0, 10.0, 0.01) ==  -8.0)
    assert_phenotype(p.transform(-3.0, 10.0, 0.01) == -10.0)
    assert_phenotype(p.transform( 0.1, 10.0, 0.01) ==   0.0)
    assert_phenotype(p.transform(-0.1, 10.0, 0.01) ==   0.0)

    # tests for perform
    flag = False
    try:
        p.perform('input')
    except AssertionError:
        flag = True
    assert_phenotype(flag)
