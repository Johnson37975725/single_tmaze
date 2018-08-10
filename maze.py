import numpy as np
from environment import *

class Maze(Environment):
    '''
    Five Inputs:
        1. Home
        2. Junction
        3. Maze End
        4. Reward
        5. Bias
    '''

    def __init__(self, noise_std = 0.001, n_think = 3):
        self.n_input = 5
        self.noise_std = noise_std
        self.n_think = n_think
        self.debug = False

    def home(self, animat):
        out = self.thinking(animat, np.array([1.0, 0.0, 0.0, 0.0, 1.0]))
        if self.debug: print("MS:" + str(out[0]))
        return True if abs(out[0]) <= 1/3 else False

    def corridor(self, animat):
        out = self.thinking(animat, np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
        if self.debug: print("CO:" + str(out[0]))
        return True if abs(out[0]) <= 1/3 else False

    def junction(self, animat):
        out = self.thinking(animat, np.array([0.0, 1.0, 0.0, 0.0, 1.0]))
        if self.debug: print("JN:" + str(out[0]))
        return out[0]

    def maze_end(self, animat, reward):
        out = self.thinking(animat, np.array([0.0, 0.0, 1.0, reward, 1.0]))
        if self.debug: print("ME:" + str(out[0]))
        return out[0]

    def thinking(self, animat, input):
        return [animat.perform(input + self.noise()) for i in range(self.n_think)][-1]

    def noise(self):
        return np.random.randn(5) * self.noise_std

if __name__ == '__main__':
    '''
    Some tests
    '''

    def assert_maze(condition, msg = "NG in 'maze.py'"):
        assert condition, msg

    class Mock:
        def __init__(self):
            self.count = 0
            self.out = 0.0
            self.cmp = np.zeros(5)
            self.flag = False

        def perform(self, input):
            self.count += 1
            self.flag = np.array_equal(input, self.cmp)
            return np.concatenate((np.array([self.out]), input))

    m = Maze(noise_std = 0.0, n_think = 3)
    animat = Mock()

    # tests for home
    animat.count = 0
    animat.out = 0.0
    animat.cmp = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    assert_maze(m.home(animat))
    animat.out = 1.0
    assert_maze(not m.home(animat))
    animat.out = -1.0
    assert_maze(not m.home(animat))
    assert_maze(animat.count == 9)
    assert_maze(animat.flag)

    # tests for corridor
    animat.count = 0
    animat.out = 0.0
    animat.cmp = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    assert_maze(m.corridor(animat))
    animat.out = 1.0
    assert_maze(not m.corridor(animat))
    animat.out = -1.0
    assert_maze(not m.corridor(animat))
    assert_maze(animat.count == 9)
    assert_maze(animat.flag)

    # tests for junction
    animat.count = 0
    animat.out = 0.0
    animat.cmp = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    assert_maze(m.junction(animat) == 0.0)
    animat.out = 1.0
    assert_maze(m.junction(animat) == 1.0)
    animat.out = -1.0
    assert_maze(m.junction(animat) == -1.0)
    assert_maze(animat.count == 9)
    assert_maze(animat.flag)

    # tests for maze_end
    animat.count = 0
    animat.out = 0.0
    animat.cmp = np.array([0.0, 0.0, 1.0, 0.5, 1.0])
    assert_maze(m.maze_end(animat, 0.5) == 0.0)
    assert_maze(animat.count == 3)
    assert_maze(animat.flag)
    animat.cmp = np.array([0.0, 1.0, 0.0, 0.5, 1.0])
    assert_maze(m.maze_end(animat, 0.0) == 0.0)
    assert_maze(not animat.flag)
