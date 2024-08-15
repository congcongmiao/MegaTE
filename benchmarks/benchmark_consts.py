from collections import defaultdict
from glob import iglob

import argparse
import os

import sys
sys.path.append('..')

from lib.partitioning import FMPartitioning, SpectralClustering
from lib.problems import get_tri_problem
from lib.config import TL_DIR, TOPOLOGIES_DIR, TEST_PROBLEM_NAMES


PROBLEM_NAMES = [
    #'triangle.json',
    'b4.json',
    #'Uninett2010.graphml',
    'Deltacom.graphml',
    #'UsCarrier.graphml',
    #'Cogentco.graphml'
    #'Deltacom_2_0.graphml',
    #'Deltacom_2_1.graphml',
    #'Deltacom_2_2.graphml',
    #'Deltacom_2_3.graphml',
    #'Deltacom_2_4.graphml',
    #'Deltacom_5_0.graphml',
    #'Deltacom_5_1.graphml',
    #'Deltacom_5_2.graphml',
    #'Deltacom_5_3.graphml',
    #'Deltacom_5_4.graphml',
    #'Deltacom_10_0.graphml',
    #'Deltacom_10_1.graphml',
    #'Deltacom_10_2.graphml',
    #'Deltacom_10_3.graphml'  
]

TOP_HYPERPARAMS = 0.1

PATH_FORM_HYPERPARAMS = (4, True, 'inv-cap')

MEGA_HYPERPARAMS = (4, True, 'inv-cap')


NCFLOW_HYPERPARAMS = {
    'b4.json': (4, True, 'inv-cap', FMPartitioning, 3),
    'Meganet.json': (4, True, 'inv-cap', FMPartitioning, 3),
    'Cogentco.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Uninett2010.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'UsCarrier.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'triangle.json': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_2_0.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_2_1.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_2_2.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_2_3.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_2_4.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_5_0.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_5_1.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_5_2.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_5_3.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_5_4.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_10_0.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_10_1.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_10_2.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_10_3.graphml': (4, True, 'inv-cap', FMPartitioning, 3),
    'Deltacom_10_4.graphml': (4, True, 'inv-cap', FMPartitioning, 3)    
}

'''
TM_MODELS = [
    'uniform', 'gravity', 'bimodal', 'poisson-high-intra', 'poisson-high-inter'
]
'''

TM_MODELS = ['uniform']
PROBLEM_NAMES_AND_TM_MODELS = [(prob_name, tm_model)
                               for prob_name in PROBLEM_NAMES
                               for tm_model in TM_MODELS]

PROBLEMS = []


def formulate(level, size):
    GROUPED_BY_PROBLEMS = defaultdict(list)
    for problem_name in PROBLEM_NAMES:
        if level == 'site':
            topo_fname = os.path.join(TOPOLOGIES_DIR, 'site_topologies',
                                  problem_name)
            tmdir = 'site-traffic-matrices'
        elif level == 'srv':
            topo_fname = os.path.join(TOPOLOGIES_DIR, 'srv_topologies-{}'.format(size),
                                  problem_name)
            tmdir = 'srv-traffic-matrices-{}'.format(size)
        #if problem_name.endswith('.graphml'):   
        #else:
        #    topo_fname = os.path.join('..', 'topologies', 'readytopologies',problem_name)
        for model in TM_MODELS:
            for tm_fname in iglob(
                    '../traffic-matrices/'+tmdir+'/{}/{}*_traffic-matrix.pkl'.format(
                        model, problem_name)):                        
                vals = os.path.basename(tm_fname)[:-4].split('_')
                _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(
                    vals[3])
                
                print(problem_name)
                GROUPED_BY_PROBLEMS[(problem_name, model, scale_factor)].append(
                    (topo_fname, tm_fname))
                PROBLEMS.append((problem_name, topo_fname, tm_fname))
                
                
            '''for tm_fname in iglob(
                    '../traffic-matrices/'+tmdir+'/holdout/{}/{}*_traffic-matrix.pkl'.format(
                    model, problem_name)):
                vals = os.path.basename(tm_fname)[:-4].split('_')
                _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(
                    vals[3])
                GROUPED_BY_HOLDOUT_PROBLEMS[(problem_name, model,
                                             scale_factor)].append(
                                                 (topo_fname, tm_fname))
                HOLDOUT_PROBLEMS.append((problem_name, topo_fname, tm_fname))'''

    GROUPED_BY_PROBLEMS = dict(GROUPED_BY_PROBLEMS)
    for key, vals in GROUPED_BY_PROBLEMS.items():
        GROUPED_BY_PROBLEMS[key] = sorted(vals)

    '''GROUPED_BY_HOLDOUT_PROBLEMS = dict(GROUPED_BY_HOLDOUT_PROBLEMS)
    for key, vals in GROUPED_BY_HOLDOUT_PROBLEMS.items():
        GROUPED_BY_HOLDOUT_PROBLEMS[key] = sorted(vals)'''
    return GROUPED_BY_PROBLEMS
    
    
    
def formulate_demo(level, size):
    GROUPED_BY_PROBLEMS = defaultdict(list)
    for problem_name in TEST_PROBLEM_NAMES:
        if level == 'site':
            topo_fname = os.path.join(TOPOLOGIES_DIR, 'site_topologies',
                                  problem_name)
            tmdir = 'site-traffic-matrices'
        elif level == 'srv':
            topo_fname = os.path.join(TOPOLOGIES_DIR, 'srv_topologies-{}'.format(size),
                                  problem_name)
            tmdir = 'srv-traffic-matrices-{}'.format(size)
        for model in TM_MODELS:
            for tm_fname in iglob(
                    '../traffic-matrices/'+tmdir+'/{}/{}*_traffic-matrix.pkl'.format(
                        model, problem_name)):                        
                vals = os.path.basename(tm_fname)[:-4].split('_')
                _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(
                    vals[3])
                
                print(problem_name)
                GROUPED_BY_PROBLEMS[(problem_name, model, scale_factor)].append(
                    (topo_fname, tm_fname))
                PROBLEMS.append((problem_name, topo_fname, tm_fname))
                

    GROUPED_BY_PROBLEMS = dict(GROUPED_BY_PROBLEMS)
    for key, vals in GROUPED_BY_PROBLEMS.items():
        GROUPED_BY_PROBLEMS[key] = sorted(vals)
    return GROUPED_BY_PROBLEMS


def get_problems(args):
    problems = []
    algo = args.algo
    size = args.server
    if args.algo == 'ncflow' or args.algo == 'lpall':
        level = 'srv'
    elif args.algo == 'mega' or args.algo == 'megatop':
        level = 'site'
    GROUPED_BY_PROBLEMS = formulate(level, size)
    
    for (
            problem_name,
            _,
            _,
    ), topo_and_tm_fnames in GROUPED_BY_PROBLEMS.items():
        for slice in args.slices:
            topo_fname, tm_fname = topo_and_tm_fnames[slice]
            problems.append((problem_name, topo_fname, tm_fname))
    return size, algo, problems
    

def get_demo_problems(algo, size):
    problems = []
    if algo == 'ncflow' or algo == 'lpall':
        if size == 1:
            level = 'site'
        else:
            level = 'srv'
    elif algo == 'mega' or algo == 'megatop':
        level = 'site'
    GROUPED_BY_PROBLEMS = formulate_demo(level, size)
    
    for (
            problem_name,
            _,
            _,
    ), topo_and_tm_fnames in GROUPED_BY_PROBLEMS.items():
        topo_fname, tm_fname = topo_and_tm_fnames[0]
        problems.append((problem_name, topo_fname, tm_fname))
    return problems

    


def get_tri_problems(args):
    problems = []
    #print(GROUPED_BY_PROBLEMS.items())
    if args.test == 'triangle':
        return get_tri_problem()
    return problems
    


def get_args_and_problems():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run',
                        dest='dry_run',
                        action='store_true',
                        default=False)
    parser.add_argument('--slices',
                        type=int,
                        choices=range(5),
                        nargs='+',
                        required=False)
    parser.add_argument('--algo',type=str,required=True)
    parser.add_argument('--server',type=int,required=True)
    args = parser.parse_args()
    size, algo, problems = get_problems(args)
    return args, size, algo, problems
    
    

def get_demo_args_and_problems(algo, size):
    problems = get_demo_problems(algo, size)
    return problems    
    

    
def get_tri_args_and_problems():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',type=str,required=False)
    parser.add_argument('--dry-run',
                        dest='dry_run',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    return args, get_tri_problems(args)
    
    
    
#def get_demo_args_and_problems():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--test',type=str,required=False)
#    parser.add_argument('--dry-run',
#                        dest='dry_run',
#                        action='store_true',
#                        default=False)
#    args = parser.parse_args()
#    return args, get_demo_problems(args)


def print_(*args, file=None):
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()
