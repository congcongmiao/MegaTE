import os
import traceback

import sys
sys.path.append('..')

#from lib.lp_top import *
from lp_all import lp_all_benchmark
from ncflow import ncflow_benchmark
from mega import mega_top_benchmark, mega_benchmark
from benchmark_consts import get_args_and_problems, print_, PATH_FORM_HYPERPARAMS, NCFLOW_HYPERPARAMS




if __name__ == '__main__':
    #if not os.path.exists(TOP_DIR):
    #    os.makedirs(TOP_DIR)
        
    args, size, algo, problems = get_args_and_problems()

    if algo == 'mega':
        if args.dry_run:   
            print('Problems to run:')
            for problem in problems:
                print(problem)
        else:
            mega_benchmark(problems, size, top=False)
    elif algo == 'megatop':
        if args.dry_run:   
            print('Problems to run:')
            for problem in problems:
                print(problem)
        else:
            print(problems)
            mega_top_benchmark(problems, size, top=True)
    elif algo == 'ncflow':
        if args.dry_run:   
            print('Problems to run:')
            for problem in problems:
                print(problem)
        else:
            print(problems)
            ncflow_benchmark(problems, size)
    elif algo == 'lpall':
        if args.dry_run:   
            print('Problems to run:')
            for problem in problems:
                print(problem)
        else:
            print(problems)
            lp_all_benchmark(problems, size, top=False)
        
    
    
    