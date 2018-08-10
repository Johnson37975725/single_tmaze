import numpy as np
from phenotype import *

class Animat(Phenotype):
    def __init__(self, genotype, plastic = True):
        super().__init__(genotype)
        self.state   = np.zeros(self.n_neuron)
        self.plastic = plastic

    def behave(self, input):
        return np.tanh(np.dot(self.weight, np.concatenate((input, self.state))))

    def learn(self, input, output = None):
        updated_weight = self.weight + self.delta_weight(input, output)
        return self.inspect_weight(updated_weight)

    def delta_weight(self, input, output = None):
        u = np.concatenate((input, self.state))
        v = self.behave(input) if output is None else output
        w = np.array([[self.apply_rule(u[j], v[i]) for j in range(len(u))] for i in range(len(v))])
        return w * (self.weight != 0.0)

    def apply_rule(self, x, y):
        return self.rule[0] * np.dot(self.rule[1:5], (x * y, x, y, 1))

    def inspect_weight(self, weight):
        return np.vectorize(lambda x: x if abs(x) < 10.0 else np.sign(x) * 10.0)(weight)

    def perform(self, input):
        output = self.behave(input)
        self.weight = self.learn(input, output) if self.plastic else self.weight
        self.state = output # This timing of updating state is paramount.
        return output

if __name__ == '__main__':
    '''
    Some tests
    '''
    import copy as cp

    def assert_animat(condition, msg = "NG in 'animat.py'"):
        assert condition, msg

    class Mock:
        def __init__(self, n_input, n_neuron):
            self.n_input  = n_input
            self.n_neuron = n_neuron
            self.weight   = np.ones((n_neuron, n_input + n_neuron))
            self.rule     = np.ones(5)

    n_input  =  5
    n_neuron = 10
    genotype = Mock(n_input, n_neuron)
    a = Animat(genotype)

    # a test for constructor
    assert_animat(np.array_equal(a.state, np.zeros(n_neuron)))

    # tests for behave
    input = np.ones(n_input)

    output = np.tanh(np.ones(n_neuron) * n_input)
    a.state = np.zeros(n_neuron)
    assert_animat(np.array_equal(a.behave(input), output))

    output = np.tanh(np.ones(n_neuron) * (n_input + n_neuron))
    a.state = np.ones(n_neuron)
    assert_animat(np.array_equal(a.behave(input), output))

    # tests for delta_weight & learn
    input   = np.ones(n_input)
    output  = np.ones(n_neuron)
    a.state = np.ones(n_neuron)
    delta   = np.ones((n_neuron, n_input + n_neuron)) * 4.0 # a.apply_rule(1.0, 1.0)

    result  = a.delta_weight(input, output)
    assert_animat(result.shape == a.weight.shape)
    assert_animat(np.array_equal(result, delta))

    result = a.learn(input, output)
    assert_animat(result.shape == a.weight.shape)
    assert_animat(np.array_equal(result, delta + 1.0))

    # tests for apply_rule
    assert_animat(a.apply_rule(3, 4) == 20) # 1 * 3 * 4 + 1 * 3 + 1 * 4 + 1 = 20

    # tests for inspect_weight
    input  = np.ones((n_neuron, n_input + n_neuron))
    output = np.ones((n_neuron, n_input + n_neuron))
    assert_animat(np.array_equal(a.inspect_weight(input), output))

    input  = np.ones((n_neuron, n_input + n_neuron)) * 30.0
    output = np.ones((n_neuron, n_input + n_neuron)) * 10.0
    assert_animat(np.array_equal(a.inspect_weight(input), output))

    # tests for perform
    input   = np.ones(n_input)
    output  = np.tanh(np.ones(n_neuron) * (n_input + n_neuron))
    a.state = np.ones(n_neuron)
    b = cp.deepcopy(a) # internal attributes shall be modified
    assert_animat(np.array_equal(b.perform(input), output))
    assert_animat(np.array_equal(b.weight, a.weight * (np.tanh(n_input + n_neuron) + 1.0) * 2.0 + 1.0))
    assert_animat(np.array_equal(b.state, output))
