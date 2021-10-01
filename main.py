#!/usr/bin/env python
# -*- coding: utf-8 -*-

# python -m cProfile -o output.file main.py <args>

import cProfile
import pstats

import Solver as sol
from testFunctions import *


def main():
    """ Main program """
    # Code goes over here.
    
    lb: list = [-5, -5]
    ub: list = [5, 5]

    
    problem = sol.Solver(ackley, lb, ub, generation=10000)
    best,xbest = problem.jaya()
    print('The objective function value = {}'.format(best[-1]))
    print('The optimum values of variables = {}'.format(xbest[-1]))
    return 0

if __name__ == "__main__":
    # Initialize profile class and call regression() function
    profiler = cProfile.Profile() # https://www.machinelearningplus.com/python/cprofile-how-to-profile-your-python-code/
    profiler.enable()
    
    main()
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(10)   
    
