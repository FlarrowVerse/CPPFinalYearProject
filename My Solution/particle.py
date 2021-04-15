import numpy as np

class Particle:
    def __init__(self, c1=2, c2=2, w=1):
        self._pbest = 0
        self._gbest = 0
        self.velocity = 0
        self.position  = np.zeros(0)
        self._fitness = 0
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def fitness(self):
        pass
    
    def calc_pbest(self):
        pass

    def get_gbest(self):
        pass

    def update_x(self):
        old_x =self.position[len(self.position)]
        self.position.append(old_x + self.velocity)

    def update_v(self):
        pass

    def show(self):
        print('Personal Best:', self._pbest)
        print('Velocity:', self.velocity)
        print('Position:', self.position)