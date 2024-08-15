import os
import pickle
import traceback
import numpy as np

import sys
sys.path.append('..')
import random

TL_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
SOL_DIR = os.path.join(TL_DIR, 'LP-solution')
#DELAY_DIR = os.path.join(SOL_DIR, 'Delay-result.csv')
PROBLEM_NAME = [#'triangle.json','b4.json', 
               #'Uninett2010.graphml',
    'Deltacom.graphml',
    #'UsCarrier.graphml',
    #'Cogentco.graphml','b4test.json'
]

if __name__ == '__main__':
    with open(os.path.join(SOL_DIR,'Delay-result.csv'),'a') as t:
        t.write('-------------------------------------------')
    
    
    for topo in PROBLEM_NAME:
        filename = '{}-linear-programming_4-paths_edge-disjoint-True_dist-metric-inv-cap_ssp_dict.pkl'.format(topo)
        filename = 'Deltacom.graphml-10-LP-in-mega_ssp_input.pkl'
        path = os.path.join(SOL_DIR, filename)
        print(filename)
        
        with open(path, 'rb') as f:
            try:
                solution = pickle.load(f)
            except EOFError:
                print("File is empty or incomplete.")
        
        
        
        #print(solution)
        pair_delay = dict()
        portion = {} 
        selected_pair = [] 
        if topo.startswith('b4.json'):
            chosen_pair = (3,7)
        elif topo.startswith('Cogen'):
            chosen_pair = (15,175)
        elif topo.startswith('Delta'):
            chosen_pair = (43,85)
            chosen_pair = (10, 65)
        elif topo.startswith('Uni'):
            chosen_pair = (6,44)
        elif topo.startswith('Us'):
            chosen_pair = (83,151)
        else:
            continue
        
        for (s, t), flow_seq in solution.items():
            site_pair = (s,t)
            total_flow = sum(flow for flow, _ in flow_seq)
            #if len(flow_seq) > 1:
            #    selected_pair.append(site_pair)
                #print(site_pair)
            if site_pair == chosen_pair:
                for flow, path in flow_seq:
                    l = len(path) - 1
                    delay = 1 * l
                    #portion[site_pair][tuple(path)] = flow / total_flow
                    if site_pair not in pair_delay:
                        pair_delay[site_pair] = (flow / total_flow)* delay
                    else:
                        pair_delay[site_pair] += (flow / total_flow)*delay
        
        
        total_pair_num = len(chosen_pair)
        
        
        if 1>0:
            DELAY = sum(delay for _, delay in pair_delay.items())
        
            print(DELAY)
            with open(os.path.join(SOL_DIR,'Delay-result.csv'),'a') as t:
                t.write('\n{}\ntotal delay:{}\n'.format(topo, DELAY))
                
                
                
                

