import os
import pickle
import traceback
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json

import sys
sys.path.append('..')
import random
from lib.config import TL_DIR, TOPOLOGIES_DIR, edge_disjoint, num_paths


SOL_DIR = os.path.join(TL_DIR, 'benchmarks')
NCFLOW_SOL_DIR = os.path.join(SOL_DIR, 'ncflow-logs')

PROBLEM_NAME = ['triangle.json',
                'b4.json', 
                'Uninett2010.graphml',
                'Deltacom.graphml',
                'UsCarrier.graphml',
                'Cogentco.graphml'
]


SIZE_LIST = [2,
             10, 
             50, 
             100, 
             200
]


def read_graph_json(fname):
    assert fname.endswith('.json')
    with open(fname) as f:
        data = json.load(f)
    return len(json_graph.node_link_graph(data).nodes())



def read_graph_graphml(fname):
    assert fname.endswith('.graphml')
    file_G = nx.read_graphml(fname).to_directed()
    if isinstance(file_G, nx.MultiDiGraph):
        file_G = nx.DiGraph(file_G)
        
    G = []
    # Pick largest strongly connected component
    for scc_ids in nx.strongly_connected_components(file_G):
        scc = file_G.subgraph(scc_ids)
        if len(scc) > len(G):
            G = scc
        
    return len(G.nodes())


if __name__ == '__main__':
    with open(os.path.join(SOL_DIR,'Delay-result.csv'),'a') as t:
        t.write('------------------------------------------------\n')
        t.write('topo, server, chosen pair, ncflow hops, mega hops ')
    
    
    for topo in PROBLEM_NAME:
        topo_fname = os.path.join(TOPOLOGIES_DIR, 'site_topologies', topo)
        if topo_fname.endswith('.json'):
            topo_size = read_graph_json(topo_fname)
        else:
            topo_size = read_graph_graphml(topo_fname)
        
        
        
        # specific site pair
        if topo.startswith('b4.json'):
            chosen_pair = (3, 7)
        elif topo.startswith('Cogen'):
            chosen_pair = (15,175)
        elif topo.startswith('Delta'):
            chosen_pair = (10, 65)
        elif topo.startswith('Uni'):
            chosen_pair = (6,44)
        elif topo.startswith('Us'):
            chosen_pair = (83,151)
        elif topo.startswith('tri'):
            chosen_pair = (0,1)
        else:
            continue
        
        
        # ncflow solution extraction
        for size in SIZE_LIST:
            nc_fname = os.path.join(NCFLOW_SOL_DIR, topo)
            for folder in os.listdir(nc_fname):
                nc_fname = os.path.join(nc_fname, folder, 
                                        '{}-per_site'.format(size), 
                                        'ncflow', 
                                        'fm_partitioning', 
                                        '{}-partitions'.format(topo_size),
                                        '{}-paths'.format(num_paths),
                                        'edge_disjoint-{}'.format(str(edge_disjoint)),
                                        'dist_metric-inv-cap',
                                        )
                print(nc_fname)
                nc_solutions = []
                iteration = 0
                while(iteration < 10000):
                    nc_sol_fname = topo + '-ncflow-partitioner_fm_partitioning-' 
                    nc_sol_fname += '{}_partitions-{}_paths-edge_disjoint_{}-dist_metric_inv-cap_iter_{}-r3-sol-dict.pkl'.format(topo_size, 
                                                                                                                   num_paths, 
                                                                                                                   str(edge_disjoint), 
                                                                                                                   iteration)
                    print(nc_sol_fname)
                    if nc_sol_fname not in os.listdir(nc_fname):
                        break
                    else:
                        nc_path = os.path.join(nc_fname, nc_sol_fname)
                        with open(nc_path, 'rb') as f:
                            try:
                                solution = pickle.load(f)
                                nc_solutions.append(solution)
                            except EOFError:
                                print("File is empty or incomplete.")
                    
                    iteration += 1
                
                #print('all the solutions:', nc_solutions)
                
                final_solution = {}
                for solution in nc_solutions:
                    #print('iteration:', nc_solutions.index(solution))
                    for (_, (s, t, demand)), flow_list in solution.items():
                        if (s, t) not in final_solution:
                            final_solution[(s, t)] = [0.0,[]]
                            final_solution[(s, t)][0] = demand
                            final_solution[(s, t)][1] = flow_list
                        else:
                            final_solution[(s, t)][0] += demand
                            cur_flow_list = final_solution[(s, t)][1]
                            same_flag = False
                            # only when edge disjoint is true
                            for (u1, v1), f1 in cur_flow_list:
                                for (u2, v2), f2 in flow_list:
                                    if (u1, v1) == (u2, v2):
                                        same_flag = True                                    
                                        list(final_solution[(s, t)][1][cur_flow_list.index(((u1, v1), f1))])[1] += f2
                                        final_solution[(s, t)][1][cur_flow_list.index(((u1, v1), f1))] = tuple(final_solution[(s, t)][1][cur_flow_list.index(((u1, v1), f1))])
                                    
                            if not same_flag:
                                final_solution[(s, t)][1] += flow_list
                
                pair_hops_dict = {}
                for (s, t), [demand, flow_list] in final_solution.items():
                    flows_hops = {}
                    for (si, ti), flow in flow_list:
                        if flow not in flows_hops:
                            flows_hops[flow] = 1
                        else:
                            flows_hops[flow] += 1
                    
                    #print('flow_hops:', flows_hops)
                    
                    pair_hops = 0.0
                    for flow, hops in flows_hops.items():
                        if flow / demand != 1.0:
                            print((s, t), flow / demand, hops)
                            print(pair_hops, (flow / demand) * hops)
                        pair_hops += (flow / demand) * hops
                    
                    pair_hops_dict[(s, t)] = pair_hops
                    
                
                        
                # average hops of the specific site pair
                av_hops = pair_hops_dict[chosen_pair]
                
                print('{},{},{},{}'.format(topo, size, chosen_pair, av_hops))
                        
                t.write('{},{},{},{}'.format(topo, str(size), str(chosen_pair), str(av_hops)))
                            
        
        











    
    
    
    