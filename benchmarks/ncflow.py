#! /usr/bin/env python

from benchmark_consts import get_args_and_problems, print_, NCFLOW_HYPERPARAMS

import os
import pickle
import traceback
import numpy as np

import sys
sys.path.append('..')

from lib.algorithms import NcfEpi
from lib.problem import Problem
import time
from lib.runtime_utils import parallelized_rt
from lib.constants import NUM_CORES

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
    'recon_runtime', 'r3_runtime', 'kirchoffs_runtime', 
    'synctime',
    "r1_synctime",
    "r2_synctime",
    'recon_synctime',
    'r3_synctime',
    'kirchoffs_synctime',
    'itertime',
]
PLACEHOLDER = ','.join('{}' for _ in HEADERS)


def ncflow_benchmark(problems, size):
    SD_list = {}
    runtime_list = {}

    with open(OUTPUT_CSV, 'a') as results:
        print_(','.join(HEADERS), file=results)
        for problem_name, topo_fname, tm_fname in problems:
            #tm_fname = '/home/ubuntu/ncflow/traffic-matrices/srv-traffic-matrices-10/uniform/Deltacom.graphml_uniform_1071951153_1.0_2.5_traffic-matrix.pkl'
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
            num_partitions_to_set = num_parts_scale_factor * int(
                np.sqrt(len(problem.G.nodes)))
                
            '''if problem_name == 'triangle.json':
                num_partitions_to_set = 3
            elif problem_name == 'b4.json':
                num_partitions_to_set = 12
            elif problem_name == 'Uninett2010.graphml':
                num_partitions_to_set = 74
            elif problem_name == 'Deltacom.graphml':
                num_partitions_to_set = 113
            elif problem_name == 'UsCarrier.graphml':
                num_partitions_to_set = 158
            elif problem_name == 'Cogentco.graphml':
                num_partitions_to_set = 197'''
            
                
            partitioner = partition_cls(num_partitions_to_set)
            partition_algo = partitioner.name

            run_dir = os.path.join(
                TOP_DIR, problem.name,
                '{}-{}'.format(traffic_seed, problem.traffic_matrix.model), '{}-per_site'.format(size))
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
                    with open(TOP_DIR + '/runtime_log.txt','a') as t:
                        t.write('\n------------------------------\n')
                        t.write(problem_name)
                        t.write('\n')
                        t.write('{}-{}\n'.format(len(problem.G.nodes),size))
                        t.write('{} partitions\n'.format(num_partitions_to_set))                       
                                                 
                    ncflow.solve(problem, partitioner)
                    
                    soldict = ncflow.sol_dict
                    
                    obj = 0.0
                    total_time = 0.0
                    
                    
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
                        
                        #print('sol as path:', nc._sol_dict_as_paths)
                        
                        r1_synctime = nc._synctime_dict['r1']
                        r2_synctime = parallelized_rt(
                            [t for _, t in nc._synctime_dict['r2'].items()], 
                            NUM_CORES)
                        recon_synctime = parallelized_rt(
                            [t for _, t in nc._synctime_dict['reconciliation'].items()], 
                            NUM_CORES)
                        r3_synctime = nc._synctime_dict['r3']
                        kirchoffs_synctime = parallelized_rt(
                            [t for _, t in nc._synctime_dict['kirchoffs'].items()], 
                            NUM_CORES)
                        synctime  = r1_synctime + r2_synctime + recon_synctime + r3_synctime + kirchoffs_synctime
                        
                        r1_runtime, r2_runtime, recon_runtime, \
                                r3_runtime, kirchoffs_runtime = nc.runtime_est(NUM_CORES, breakdown = True)
                        runtime = r1_runtime + r2_runtime + recon_runtime + r3_runtime + kirchoffs_runtime
                        total_flow = nc.obj_val
                        
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
                            kirchoffs_runtime,
                            synctime,
                            r1_synctime,
                            r2_synctime,
                            recon_synctime,
                            r3_synctime,
                            kirchoffs_synctime,
                            ncflow.iter_time[iter],
                            )
                        print_(result_line, file=results)
                        obj += nc.obj_val
                        total_time += runtime
                        
                print_('Satisfied Demand(%):', obj / total_demand)
                
                SD = obj / total_demand
                time = total_time
                
                if problem.name not in SD_list:
                    SD_list[problem.name] = 0.0
                SD_list[problem.name] = SD
                
                if problem.name not in runtime_list:
                    runtime_list[problem.name] = 0.0
                runtime_list[problem.name] = time
            except:
                print_(
                    'NCFlow partitioner {}, {} paths, Problem {}, traffic seed {}, traffic model {} failed'
                    .format(partition_algo, num_paths, problem.name,
                            traffic_seed, problem.traffic_matrix.model))
                traceback.print_exc(file=sys.stdout)
    return (SD_list, runtime_list)  
    
    
    

if __name__ == '__main__':
    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)

    args, problems = get_args_and_problems()

    if args.dry_run:
        print('Problems to run:')
        for problem in problems:
            print(problem)
    else:
        print(problems)
        benchmark(problems)
