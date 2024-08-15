#! /usr/bin/env python

from benchmark_consts import get_args_and_problems, print_, NCFLOW_HYPERPARAMS, get_tri_args_and_problems

import os
import pickle
import traceback
import numpy as np
import multiprocessing
import sys
sys.path.append('..')
import time
from lib.algorithms import NcfEpi
from lib.problem import Problem
from lib.problems import get_tri_problem, get_srv_tri_problem

TOP_DIR = 'ncflow-logs'
OUTPUT_CSV = 'ncflow.csv'

# Sweep topos and traffic matrices for that topo. For each combo, record the
# runtime and total flow for each algorithm
HEADERS = [
    'problem', 'num_nodes', 'num_edges', 'traffic_seed', 'tm_model',
    'scale_factor', 'num_commodities', 'total_demand', 'algo',
    'clustering_algo', 'num_partitions', 'size_of_largest_partition',
    'partition_runtime', 'num_paths', 'edge_disjoint', 'dist_metric',
    'iteration', 'total_flow', 'runtime', 'r1_runtime', 'r2_runtime',
    'recon_runtime', 'r3_runtime', 'kirchoffs_runtime'
]
PLACEHOLDER = ','.join('{}' for _ in HEADERS)


def benchmark_tri(problems):   
    problem_name = 'triangle.json'
    topo_fname = '../topologies/srv_topologies-2/triangle.json'  
    tm_fname = '../traffic-matrices/srv-traffic-matrices-2/triangle.json_uniform_480528682_1.0_1500.0_traffic-matrix.pkl'
    print(problems.G[0][1]['capacity'],problems.G[0][2]['capacity'],problems.G[2][1]['capacity'])

    with open(OUTPUT_CSV, 'a') as results:
        print_(','.join(HEADERS), file=results)
        
        problem = Problem.from_file(topo_fname, tm_fname, top=False)
        print_(problem.name, tm_fname)
        traffic_seed = problem.traffic_matrix.seed
        total_demand = problem.total_demand
        print_('traffic seed: {}'.format(traffic_seed))
        print_('traffic matrix model: {}'.format(
                problem.traffic_matrix.model))
        print_('traffic matrix scale factor: {}'.format(
                problem.traffic_matrix.scale_factor))
        print_('total demand: {}'.format(total_demand))

        num_paths, edge_disjoint, dist_metric, partition_cls, num_parts_scale_factor = NCFLOW_HYPERPARAMS[
                problem_name]
        #num_partitions_to_set = num_parts_scale_factor * int(
        #        np.sqrt(len(problem.G.nodes)))
        num_partitions_to_set = 3
        partitioner = partition_cls(num_partitions_to_set)
        partition_algo = partitioner.name

        run_dir = os.path.join(
                TOP_DIR, problem.name,
                '{}-{}'.format(traffic_seed, problem.traffic_matrix.model))
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        try:
            print_(
                    '\nNCFlow, {} partitioner, {} partitions, {} paths, edge disjoint {}, dist metric {}'
                    .format(partition_algo, num_partitions_to_set, num_paths,
                            edge_disjoint, dist_metric))
            run_nc_dir = os.path.join(
                    run_dir, 'ncflow', partition_algo,
                    '{}-partitions'.format(num_partitions_to_set),
                    '{}-paths'.format(num_paths),
                    'edge_disjoint-{}'.format(edge_disjoint),
                    'dist_metric-{}'.format(dist_metric))
            if not os.path.exists(run_nc_dir):
                os.makedirs(run_nc_dir)
            with open(
                        os.path.join(
                            run_nc_dir,
                            '{}-ncflow-partitioner_{}-{}_partitions-{}_paths-edge_disjoint_{}-dist_metric_{}.txt'
                            .format(problem.name, partition_algo,
                                    num_partitions_to_set, num_paths,
                                    edge_disjoint, dist_metric)), 'w') as log:
                ncflow = NcfEpi.new_max_flow(num_paths,
                                                 edge_disjoint=edge_disjoint,
                                                 dist_metric=dist_metric,
                                                 out=log)
                    
                
                start_time = time.time()
                with open("/home/ubuntu/ncflow/benchmarks/ncflow-logs/runtime_log.txt",'a') as t:
                    t.write(str(len(problem.G.nodes)))
                    t.write('\n')
                    t.write(str(num_partitions_to_set))
                    t.write('\nr2 path computing,r1,r2,recon,kirchoffs,r3,iteration time,total time\n')
                    
                ncflow.solve(problem, partitioner)
                end_time = time.time()
                print('solve total time:', end_time-start_time)
                with open("/home/ubuntu/ncflow/benchmarks/ncflow-logs/runtime_log.txt",'a') as t:
                    t.write('{}\n'.format(end_time-start_time))
                    
                
                obj = 0.0
                    
                for i, nc in enumerate(ncflow._ncflows):
                    with open(
                                log.name.replace(
                                    '.txt',
                                    '-runtime-dict-iter-{}.pkl'.format(i)),
                                'wb') as w:
                        pickle.dump(nc.runtime_dict, w)
                    with open(
                                log.name.replace(
                                    '.txt', '-sol-dict-iter-{}.pkl'.format(i)),
                                'wb') as w:
                        pickle.dump(nc.sol_dict, w)
                num_partitions = len(np.unique(ncflow._partition_vector))

                for iter in range(ncflow.num_iters):
                    nc = ncflow._ncflows[iter]

                    r1_runtime, r2_runtime, recon_runtime, \
                                r3_runtime, kirchoffs_runtime = nc.runtime_est(14, breakdown = True)
                    runtime = r1_runtime + r2_runtime + recon_runtime + r3_runtime + kirchoffs_runtime
                    total_flow = nc.obj_val
                    print(runtime , ' s')
                    result_line = PLACEHOLDER.format(
                            problem.name, len(problem.G.nodes),
                            len(problem.G.edges), traffic_seed,
                            problem.traffic_matrix.model,
                            problem.traffic_matrix.scale_factor,
                            len(problem.commodity_list), total_demand,
                            'ncflow_edge_per_iter', partition_algo,
                            num_partitions,
                            partitioner.size_of_largest_partition,
                            partitioner.runtime, num_paths, edge_disjoint,
                            dist_metric, iter, total_flow, runtime, r1_runtime,
                            r2_runtime, recon_runtime, r3_runtime,
                            kirchoffs_runtime)
                    print_(result_line, file=results)
                    obj += nc.obj_val
        
        
            print_('total demand', total_demand)
            print_('Satisfied Demand(%):', obj / total_demand)
        except:
            print_(
                    'NCFlow partitioner {}, {} paths, Problem {}, traffic seed {}, traffic model {} failed'
                    .format(partition_algo, num_paths, problem.name,
                            traffic_seed, problem.traffic_matrix.model))
            traceback.print_exc(file=sys.stdout)

def run():
    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)

    args, problems = get_tri_args_and_problems()
    problems = get_tri_problem()
    '''if args.dry_run:
        print('Problems to run:')
        for problem in problems:
            print(problem)
    else:'''
    print(problems)
    #start_time = time.time()
    benchmark_tri(problems)


if __name__ == '__main__':
    cpu_core = 0
    cpu_count = multiprocessing.cpu_count()
    print('cpu number: {}', cpu_count)
    '''process = multiprocessing.Process(target=run)
    
    process.cpu_affinity([cpu_core])
    
    process.start()'''
    run()
    
    
