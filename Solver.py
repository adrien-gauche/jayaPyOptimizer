import copy as cp
import time

import numpy as np
#import numba as nb
from tqdm import tqdm


class Solver():
    def __init__(self, f, lb, ub, pop_size=25, generation=1000, debug=False):

        # problem parameters
        self.f: function = f  # return scalar
        self.lb = np.array(lb)
        self.ub = np.array(ub)

        # metaheuristic parameters
        self.pop_size: int = pop_size
        self.generation: int = generation

        # metaheuristic variables
        self.p1 = self.initialPopulation()
        # vector to storage best value at each iteration
        self.p1_f = self.cost_function(self.p1)
        self.new_p1 = self.p1

        self.best   = np.ones(self.generation) * np.inf
        self.xbest  = np.zeros((self.generation, self.lb.shape[0]))
        
        self.debug = debug

    def initialPopulation(self):
        """
        Once the search space is definied and termination criteria is set,
        initialpopuplation """
        #population = []
        population_initial = np.random.random_sample((self.pop_size, len(self.lb)))
        population_initial = self.lb + (self.ub - self.lb) * population_initial

        return population_initial

    def updatePopulation(self):
        """update population position based on equation and update position
        Args:
            p1 (np.array): all swarm position
            dim (int): problem dimension (x objectives values)

        Returns:
            np.array: position of all particles updated
        """
        best_x = self.p1[self.p1_f.argmin()]
        worst_x = self.p1[self.p1_f.argmax()]
        new_x = cp.deepcopy(self.new_p1)    # create a new variable at new memory address
        for i in range(self.p1.shape[0]):
            old_x = self.p1[i]
            r = np.random.random_sample((self.lb.shape[0]))
            new_x[i] = old_x+r[0]*(best_x-abs(old_x))-r[1]*(worst_x-abs(old_x))
        self.new_p1 = new_x

        return self.new_p1

    def greedySelector(self):
        """Function 'eat' only goods solutions. For each generation, keeps only 
        goods solutions and completely discard poors or inferiors solutions.

        Args:
            p1 ([type]): current solutions
            new_p1 ([type]): new generated solutions

        Returns:
            [type]: updated solutions with greedy selector
        """
        datamask = self.p1_f > self.new_p1_f # True were new position have better fitness
        self.p1[datamask] = self.new_p1[datamask] # update positions where fitness is better
        
        #for i in range(self.p1.shape[0]):
        #    if self.f(self.p1[i]) > self.f(self.new_p1[i]):
        #        self.p1[i] = self.new_p1[i]
        #assert((p1_updated == self.p1).all())
        return self.p1

    def trimming(self, lb, ub):
        """Adjustment of the particles position by bounding them in the search space.
        Positions outside search spaces are put equal to boudaries

        Args:
            new_p1 ([type]): new particles positions
            lb ([type]):  lower bounds
            ub ([type]):  upper bounds

        Returns:
            [type]: particles positions bounded inside lb and up
        """
        for i in range(self.new_p1.shape[0]):
            for j in range(self.lb.shape[0]):
                if self.new_p1[i][j] > ub[j]:
                    self.new_p1[i][j] = ub[j]
                elif self.new_p1[i][j] < lb[j]:
                    self.new_p1[i][j] = lb[j]
        return self.new_p1

    
    def cost_function(self, position_matrix: np.array):
        fitness_vector = np.zeros(position_matrix.shape[0])
        for i in range(position_matrix.shape[0]):
            fitness_vector[i] = self.f(position_matrix[i])
        return fitness_vector

    # @nb.jit(nopython=True, parallel=True)
    def jaya(self):
        start = time.time()

        gen = 0    # initialise current generation
        best = []  # initialise best position
        pbar = tqdm(total = self.generation)
        while (gen < self.generation):
            self.new_p1 = self.updatePopulation()           # move population with equation
            # check if particles are within bounds
            self.new_p1 = self.trimming(self.lb, self.ub)
            # determine new fitness for all particles
            self.new_p1_f = self.cost_function(self.new_p1)
            #self.p1 = self.greedySelector()                 # update only better positions
            self.greedySelector()                 # update only better positions
            self.p1_f = self.cost_function(self.p1)   # update fitness on current positions
            gen = gen + 1
            self.best[-1] = self.p1_f.min()
            self.xbest[-1] = self.p1[self.p1_f.argmin()]
            if self.debug and gen % (self.generation * 0.1) == 0.0: # plot progress every 10% of total generation
                print("DEBUG \n------")
                print("Generation:", gen)
                print('Best={}'.format(self.best[-1]))
                print('xbest={}'.format(self.xbest[-1]))
                print("END DEBUG \n----------")
            pbar.update()
        end = time.time()
        pbar.close()
        print("time:", end - start)

        return self.best, self.xbest
