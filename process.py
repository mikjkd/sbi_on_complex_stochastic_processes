import typing
from random import randint, random, choice

import numpy as np


class ProbabilisticCharacterization:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma


class Process:
    def __init__(self):
        pass

    # There are several ways to generate a process, maybe we can return a constant value + a noise
    # or something more complex. It depends on what we want to prove.
    def generate(self):
        pass

    def generate_n(self, n) -> typing.List:
        v = []
        for i in range(n):
            v.append(self.generate())
        return v


class RandomWalkProcess(Process):
    def __init__(self, probabilistic_characterization: ProbabilisticCharacterization, start_val, drift=10):
        super().__init__()
        self.probabilistic_characterization = probabilistic_characterization
        self.val = start_val
        self.drift = drift

    def generate(self):
        self.val = self.val + np.random.normal(self.probabilistic_characterization.mu,
                                               self.probabilistic_characterization.sigma) + self.drift
        return self.val


class ExperimentalProcess(Process):
    def __init__(self, pc1: ProbabilisticCharacterization, pc2: ProbabilisticCharacterization, start_val):
        self.val = start_val
        self.pc1 = pc1
        self.pc2 = pc2

    def generate(self):
        k = randint(0, 1)
        if k == 0:
            self.val = np.random.normal(self.pc1.mu,
                                        self.pc1.sigma)
        else:
            self.val = np.random.normal(self.pc2.mu,
                                        self.pc2.sigma)
        return self.val


class ExperimentalProcess(Process):
    def __init__(self, pc1: ProbabilisticCharacterization, pc2: ProbabilisticCharacterization, start_val):
        self.val = start_val
        self.pc1 = pc1
        self.pc2 = pc2

    def generate(self):
        k = randint(0, 1)
        if k == 0:
            self.val = np.random.normal(self.pc1.mu,
                                        self.pc1.sigma)
        else:
            self.val = np.random.normal(self.pc2.mu,
                                        self.pc2.sigma)
        return self.val


class ExperimentalProcess2(Process):
    def __init__(self, pc1: ProbabilisticCharacterization, pc2: ProbabilisticCharacterization, start_val):
        self.val = start_val
        self.pc1 = pc1
        self.pc2 = pc2

    def generate(self):
        k = randint(0, 1)
        if k == 0:
            self.val = np.random.normal(self.pc1.mu,
                                        self.pc1.sigma)
        else:
            self.val = np.random.normal(self.pc2.mu,
                                        self.pc2.sigma)
        return self.val


class SpikeProcess(Process):
    def __init__(self, probabilistic_characterization: ProbabilisticCharacterization, start_val, spike_rate,
                 spike_range: typing.List):
        super().__init__()
        self.probabilistic_characterization = probabilistic_characterization
        self.val = start_val
        self.spike_rate = spike_rate
        self.spike_range = spike_range

    def generate(self):
        # x(t) = x(t-1)+xi
        if random() > self.spike_rate:
            self.val = self.val + np.random.normal(self.probabilistic_characterization.mu,
                                                   self.probabilistic_characterization.sigma)
            return self.val
        else:
            return self.val + choice(self.spike_range)

    def __probabilistic_return(self):
        return choice(self.spike_range) if random() < self.spike_rate else 0


class StrangeProcess(Process):
    def __init__(self, pc1: ProbabilisticCharacterization, pc2: ProbabilisticCharacterization, change_rate=0.7):
        super().__init__()
        self.pc1 = pc1
        self.pc2 = pc2
        self.change_rate = change_rate
        self.val = 10

    def generate(self):
        # x(t) = x(t-1)+xi
        if random() > self.change_rate:
            self.val = self.val + np.random.normal(self.pc1.mu,
                                                   self.pc1.sigma)
        else:
            self.val = self.val + np.random.normal(self.pc2.mu,
                                                   self.pc2.sigma)
        return self.val

class StrangeProcessf(Process):
    def __init__(self, pc1: ProbabilisticCharacterization, pc2: ProbabilisticCharacterization, f,
                 change_rate=0.7):
        super().__init__()
        self.iniitial_conditions = {
            'val': 10
        }
        self.pc1 = pc1
        self.pc2 = pc2
        self.change_rate = change_rate
        self.val = self.iniitial_conditions['val']
        self.f = f

    def generate(self):
        if random() > self.change_rate:
            self.val = self.f(self.val) + np.random.normal(self.pc1.mu,
                                                           self.pc1.sigma)
        else:
            self.val = self.f(self.val) + np.random.normal(self.pc2.mu,
                                                           self.pc2.sigma)
        return self.val

    def reset(self):
        self.val = self.iniitial_conditions['val']
