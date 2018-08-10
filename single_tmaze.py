import numpy as np
from maze import *

class SingleTMaze(Maze):
    def __init__(self, noise_std = 0.001, n_think = 3):
        super().__init__(noise_std, n_think)
        self.max_reward =  1.0
        self.min_reward =  0.2
        self.penalty    = -0.3

        self.n_trip = 100
        self.cycle  =  50
        self.region =  30

    def trip(self, animat, target):
        if not self.home(animat):
            return self.penalty

        if not self.corridor(animat):
            return self.penalty

        turn1 = self.junction(animat)
        if abs(turn1) < 1/3:
            return self.penalty

        if not self.corridor(animat):
            return self.penalty

        reward = self.max_reward if np.sign(turn1) == target else self.min_reward
        self.maze_end(animat, reward)

        if not self.corridor(animat):
            return reward + self.penalty

        turn2 = self.junction(animat)
        if abs(turn2) < 1/3 or np.sign(turn2) == np.sign(turn1):
            return reward + self.penalty

        if not self.corridor(animat):
            return reward + self.penalty

        self.home(animat)

        return reward

    def evaluate(self, animat):
        switch = self.switch_points(self.n_trip, self.cycle, self.region)
        target = np.sign(np.random.rand() - 0.5)

        reward_sum = 0.0
        for i in range(self.n_trip):
            target = target * -1.0 if i in switch else target
            reward_sum += self.trip(animat, target)

        return reward_sum

    def switch_points(self, n_trip, cycle, region):
        step = [cycle * (i + 1) for i in range(int(n_trip / cycle - 1))]
        return np.array([int(s + np.random.rand() * region - (region / 2)) for s in step])

if __name__ == '__main__':
    '''
    Some tests
    '''

    def assert_single_tmaze(condition, msg = "NG in 'single_tmaze.py'"):
        assert condition, msg

    class Mock:
        def __init__(self, n_think):
            self.count = 0
            self.n_think = n_think
            self.action = np.array([
                [ 0.0], # Go Straight at Maze Start
                [ 0.0], # Go Straight at Corridor
                [ 1.0], # Turn Right  at Junction
                [ 0.0], # Go Straight at Corridor
                [ 0.0], # Go Straight at Maze End
                [ 0.0], # Go Straight at Corridor
                [-1.0], # Turn Left   at Junction
                [ 0.0], # Go Straight at Corridor
                [ 0.0], # Go Straight at Maze Start
            ])

        def perform(self, input):
            output = self.action[int(self.count / self.n_think)]
            self.count += 1
            self.count %= len(self.action) * self.n_think
            return output

    m = SingleTMaze()
    animat = Mock(m.n_think)

    # tests for trip
    assert_single_tmaze(m.trip(animat,  1.0) == m.max_reward)
    assert_single_tmaze(m.trip(animat, -1.0) == m.min_reward)

    # tests for evaluate (here presume only one swithing point)
    min_region = (m.n_trip - m.region) / 2
    max_region = (m.n_trip + m.region) / 2
    min_reward_sum = min_region * m.max_reward + max_region * m.min_reward
    max_reward_sum = min_region * m.min_reward + max_region * m.max_reward

    reward_sum = [m.evaluate(animat) for i in range(100)]
    assert_single_tmaze(int(min(reward_sum)) >= min_reward_sum)
    assert_single_tmaze(int(max(reward_sum)) <= max_reward_sum)

    # tests for switch_points
    n_trip, cycle, region = 150, 50, 20
    sp = np.array([m.switch_points(n_trip, cycle, region) for i in range(100)])
    assert_single_tmaze(max(np.abs(sp[:,0] - cycle * 1)) <= (region / 2))
    assert_single_tmaze(max(np.abs(sp[:,1] - cycle * 2)) <= (region / 2))
    assert_single_tmaze(np.all(np.apply_along_axis(len, 1, sp) == (n_trip / cycle - 1)))
