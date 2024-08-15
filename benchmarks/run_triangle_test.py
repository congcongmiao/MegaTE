#! /usr/bin/env python

from benchmark_consts import get_test_args_and_problems, print_, PATH_FORM_HYPERPARAMS
from SSP_v1 import *
import os
import pickle
import traceback
import multiprocessing
import sys
sys.path.append('..')

from lib.algorithms import PathFormulation
from lib.problem import Problem
import time
from pyomo.environ import *
from lib.problems import get_tri_problem,get_srv_tri_problem

#TOP_DIR = 'path-form-logs'
TOP_DIR = 'linear-programming-logs'
HEADERS = [
    'problem', 'num_nodes', 'num_edges', 'traffic_seed', 'scale_factor',
    'tm_model', 'num_commodities', 'total_demand', 'algo', 'num_paths',
    'edge_disjoint', 'dist_metric', 'total_flow', 'runtime'
]
PLACEHOLDER = ','.join('{}' for _ in HEADERS)


def benchmarks_tri(problems):
    num_paths, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS   
    problem_name = 'triangle-problem'
    topo_fname = '../topologies/readytopologies/triangle.json'
    #topo_fname = '../topologies/srv_topologies/triangle.json'
    tm_fname = '../traffic-matrices/ready-traffic-matrices/triangle.json_uniform_480528682_1.0_1500.0_traffic-matrix.pkl'
    #tm_fname = '../traffic-matrices/srv-traffic-matrices/triangle.json_uniform_480528682_1.0_1500.0_traffic-matrix.pkl'
    print(problems.G[0][1]['capacity'],problems.G[0][2]['capacity'],problems.G[2][1]['capacity'])
    with open('triangle_test.csv', 'a') as results:
        print_(','.join(HEADERS), file=results)
        problem = Problem.from_file(topo_fname, tm_fname)
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
                TOP_DIR, problem.name,
                '{}-{}'.format(traffic_seed, problem.traffic_matrix.model))
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        #print(problem.traffic_matrix._tm)
        try:
            print_(
                    #'\nPath formulation, {} paths, edge disjoint {}, dist metric {}'
                    '\nLinear programming, {} paths, edge disjoint {}, dist metric {}'
                    .format(num_paths, edge_disjoint, dist_metric))
            with open(
                        os.path.join(
                            run_dir,
                            #'{}-path-formulation_{}-paths_edge-disjoint-{}_dist-metric-{}.txt'
                            '{}-linear-programming_{}-paths_edge-disjoint-{}_dist-metric-{}.txt'
                            .format(problem.name, num_paths, edge_disjoint,
                                    dist_metric)), 'w') as log:
                    #pf = PathFormulation.new_max_flow(
                lp = PathFormulation.new_max_flow(  
                        num_paths,
                        edge_disjoint=edge_disjoint,
                        dist_metric=dist_metric,
                        out=log)
                    # pf.solve(problem)
                lp.solve(problem)
                    # pf_sol_dict = pf.extract_sol_as_dict()
                lp_sol_dict = lp.extract_sol_as_dict()
                ssp_dict = lp.ssp_output()
                
                with open(
                            os.path.join(
                                run_dir,
                                #'{}-path-formulation_{}-paths_edge-disjoint-{}_dist-metric-{}_sol-dict.pkl'
                                '{}-linear-programming_{}-paths_edge-disjoint-{}_dist-metric-{}_sol-dict.pkl'
                                .format(problem.name, num_paths, edge_disjoint,
                                        dist_metric)), 'wb') as w:
                        # pickle.dump(pf_sol_dict, w)
                    pickle.dump(lp_sol_dict, w)
                with open(
                            os.path.join(
                                run_dir,
                                #'{}-path-formulation_{}-paths_edge-disjoint-{}_dist-metric-{}_sol-dict.pkl'
                                '{}-linear-programming_{}-paths_edge-disjoint-{}_dist-metric-{}_ssp-dict.pkl'
                                .format(problem.name, num_paths, edge_disjoint,
                                        dist_metric)), 'wb') as w:
                        # pickle.dump(pf_sol_dict, w)
                    pickle.dump(ssp_dict, w)
                
            print(value(lp.obj_val))
            #print(ssp_dict)
            
            
            #ssp_run_time, ssp_run_time_per_path = run_tri_ssp()
            print(value(lp.runtime))
            #print(ssp_run_time,ssp_run_time_per_path) 
            #print(value(lp.runtime)+ssp_run_time)           
            result_line = PLACEHOLDER.format(
                    problem.name,
                    len(problem.G.nodes),
                    len(problem.G.edges),
                    traffic_seed,
                    problem.traffic_matrix.scale_factor,
                    problem.traffic_matrix.model,
                    len(problem.commodity_list),
                    total_demand,
                    #'path_formulation',
                    'linear_programming',
                    num_paths,
                    edge_disjoint,
                    dist_metric,
                    # pf.obj_val,
                    # pf.runtime,
                    value(lp.obj_val),
                    lp.runtime,
                )
            print_(result_line, file=results)

        except Exception:
            print_(
                    # 'Path formulation {} paths, edge disjoint {}, dist metric {}, Problem {}, traffic seed {}, traffic model {} failed'
                    'Linear programming {} paths, edge disjoint {}, dist metric {}, Problem {}, traffic seed {}, traffic model {} failed'
                    .format(num_paths, edge_disjoint, dist_metric,
                            problem.name, traffic_seed,
                            problem.traffic_matrix.model))
            traceback.print_exc(file=sys.stdout)

if __name__ == '__main__':
    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)
    args, problems = get_test_args_and_problems()
    problems = get_tri_problem()
    #print(problems.G[0][1]['capacity']) 
    
    benchmarks_tri(problems)