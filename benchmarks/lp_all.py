#! /usr/bin/env python

from benchmark_consts import get_args_and_problems, print_, PATH_FORM_HYPERPARAMS, TOP_HYPERPARAMS
#from SSP import *

import os
import pickle
import traceback

import sys
sys.path.append('..')

from lib.algorithms import ServerLP_Formulation
from lib.problem import Problem
import time
# from pyomo.environ import *
from lib.config import TL_DIR, TOPOLOGIES_DIR


LP_ALL_DIR = 'LP-all-logs'
SITE_PATH_DIR = os.path.join(TOPOLOGIES_DIR, 'paths', 'site-paths')


HEADERS = [
    'problem', 'num_nodes', 'num_edges', 'traffic_seed', 'scale_factor',
    'tm_model', 'num_commodities', 'total_demand', 'algo', 'num_paths',
    'edge_disjoint', 'dist_metric', 'total_flow', 'Server_LP_satisfied_demand(%)', 
    'runtime'
]
PLACEHOLDER = ','.join('{}' for _ in HEADERS)


def lp_all_benchmark(problems, size, top=False):
    num_paths, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS 
    SD_list = {}
    runtime_list = {}   
    with open('LP-all.csv', 'a') as results:
        print_(','.join(HEADERS), file=results)
        for problem_name, topo_fname, tm_fname in problems:               
            print(problem_name, topo_fname, tm_fname)
            problem = Problem.from_file(topo_fname, tm_fname, top=False)
            print_(problem.name, tm_fname)
            traffic_seed = problem.traffic_matrix.seed
            total_demand = problem.total_demand
            print_('traffic seed: {}'.format(traffic_seed))
            print_('traffic scale factor: {}'.format(
                problem.traffic_matrix.scale_factor))
            print_('traffic matrix model: {}'.format(
                problem.traffic_matrix.model))
            print_('total demand: {}'.format(total_demand))
            
            run_dir = os.path.join(
                LP_ALL_DIR, problem.name,
                '{}-{}'.format(traffic_seed, problem.traffic_matrix.model), '{}-per_site'.format(size))
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)            
          

            try:
                print_(
                    '\nServer level Linear programming, {} paths, edge disjoint {}, dist metric {}'
                    .format(num_paths, edge_disjoint, dist_metric))
                with open(
                        os.path.join(
                            run_dir,
                            '{}-LP-all_{}-paths_edge-disjoint-{}_dist-metric-{}.txt'
                            .format(problem.name, num_paths, edge_disjoint,
                                    dist_metric)), 'w') as log:
                    lp = ServerLP_Formulation.new_max_flow(  
                        num_paths,
                        edge_disjoint=edge_disjoint,
                        dist_metric=dist_metric,
                        out=log)
                    lp.solve(problem,server=size)
                    lp_sol_dict = lp.extract_sol_as_dict()
                    ssp_dict = lp.ssp_output()
                    with open(
                            os.path.join(
                                run_dir,
                                '{}-LP-all_{}-paths_edge-disjoint-{}_dist-metric-{}_sol-dict.pkl'
                                .format(problem.name, num_paths, edge_disjoint,
                                        dist_metric)), 'wb') as w:
                        pickle.dump(lp_sol_dict, w)
                    
                    '''with open(
                            os.path.join(
                                run_dir,
                                '{}-linear-programming_{}-paths_edge-disjoint-{}_dist-metric-{}_ssp_dict.pkl'
                                .format(problem.name, num_paths, edge_disjoint,
                                        dist_metric)), 'wb') as w:
                        # pickle.dump(pf_sol_dict, w)
                        pickle.dump(ssp_dict, w)'''
                    
                print_('Satisfied demand(%):', lp.obj_val / total_demand)
                print_('Runtime:', lp.runtime)
            
                    
                
                result_line = PLACEHOLDER.format(
                    problem.name,
                    len(problem.G.nodes),
                    len(problem.G.edges),
                    traffic_seed,
                    problem.traffic_matrix.scale_factor,
                    problem.traffic_matrix.model,
                    len(problem.commodity_list),
                    total_demand,
                    'lp_all',
                    num_paths,
                    edge_disjoint,
                    dist_metric,
                    lp.obj_val,
                    lp.obj_val / total_demand,
                    lp.runtime,
                )
                print_(result_line, file=results)
            
                SD = lp.obj_val / total_demand
                time = lp.runtime
                
                if problem.name not in SD_list:
                    SD_list[problem.name] = 0.0
                SD_list[problem.name] = SD
                
                if problem.name not in runtime_list:
                    runtime_list[problem.name] = 0.0
                runtime_list[problem.name] = time
            
            except Exception:
                print_(
                    'Server level Linear programming {} paths, edge disjoint {}, dist metric {}, Problem {}, traffic seed {}, traffic model {} failed'
                    .format(num_paths, edge_disjoint, dist_metric,
                            problem.name, traffic_seed,
                            problem.traffic_matrix.model))
                traceback.print_exc(file=sys.stdout)

    return (SD_list, runtime_list)  


if __name__ == '__main__':
    if not os.path.exists(MEGA_DIR):
        os.makedirs(MEGA_DIR)
    
    #if len(sys.argv) == 2 and sys.arv[2] == 'triangle':
    #problems = get_tri_problem()
    #benchmarks_tri(problems)

    args, size, problems = get_args_and_problems()
    #print(size)
    #print(problems)
    if args.dry_run:   # no actual move
        print('Problems to run:')
        for problem in problems:
            print(problem)
    else:
        benchmark(problems,size)
        
        
