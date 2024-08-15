#! /usr/bin/env python

from pathos import multiprocessing
from itertools import product
import numpy as np
import traceback
import os
import sys
sys.path.append('..')

from lib.problems import get_problem, get_tri_problem


TM_DIR = '../traffic-matrices/site-traffic-matrices'


SCALE_FACTORS = [1.]


MODELS = ['uniform']


NUM_SAMPLES = 1


def generate_traffic_matrix(args):
    prob_short_name, model, scale_factor = args
    tm_model_dir = os.path.join(TM_DIR, model)

    for _ in range(NUM_SAMPLES):
        print(prob_short_name, model, scale_factor)
        if model =='uniform':
            print('ll')
        problem = get_problem(prob_short_name,
                              model,
                              scale_factor=scale_factor,
                              seed=np.random.randint(2**31 - 1))
        problem.print_stats()

        try:
            problem.traffic_matrix.serialize(tm_model_dir)
        except Exception:
            print('{}, model {}, scale factor {} failed'.format(
                problem.name, model, scale_factor))
            traceback.printexc()

def generate_test_traffic_matrix():
    print('generating triangle tm')
    tm_test_dir = os.path.join(TM_DIR, 'triangle')
    problem = get_tri_problem()
    print('got')
    
    try:
        problem.traffic_matrix.serialize(tm_test_dir)
    except Exception:
        print('failed')
        traceback.printexc()



if __name__ == '__main__':
    PROBLEM_SHORT_NAMES = [        
        'b4',
        'delta',
        'uninett',
        'us-carrier',
        'cogentco'        
    ]
    
    
    if len(sys.argv) == 2 and sys.argv[1] == '--holdout':
        TM_DIR += '/holdout'

    if not os.path.exists(TM_DIR):
        os.makedirs(TM_DIR)
        
    '''if len(sys.argv) == 2 and sys.argv[1] == 'triangle':
        tm_test_dir = os.path.join(TM_DIR, 'triangle')
        if not os.path.exists(tm_test_dir):
            os.makedirs(tm_test_dir)
        combination = 'triangle-problem', 'uniform', 1.0
        generate_traffic_matrix(combination)'''
    
    
    for model in MODELS:
        tm_model_dir = os.path.join(TM_DIR, model)
        if not os.path.exists(tm_model_dir):
            os.makedirs(tm_model_dir)
    #pool = multiprocessing.ProcessPool(14)
    #pool.map(generate_traffic_matrix,
             #product(PROBLEM_SHORT_NAMES, MODELS, SCALE_FACTORS))
             
    for combination in product(PROBLEM_SHORT_NAMES, MODELS, SCALE_FACTORS):
        generate_traffic_matrix(combination)
