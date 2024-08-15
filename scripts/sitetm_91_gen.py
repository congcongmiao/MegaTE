import networkx as nx
import numpy as np
import pickle
import sys
import os

sys.path.append('..')
TL_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
TOPOLOGIES_DIR = os.path.join(TL_DIR, 'topologies')

TM_DIR = '../traffic-matrices/ready-traffic-matrices'
nineone_TM_DIR = '../traffic-matrices/srv-traffic-matrices'

size = [10,50,100,200,300,500,1000]


def process_tm(tm, seed):
    np.random.seed(seed)
    rows, cols = np.triu_indices_from(tm, k=1)
    pos_indices = np.random.choice(rows.shape[0], int(rows.shape[0] / 10), replace=False)
    original_sum = np.sum(tm)
    picked_rows = np.hstack([rows[pos_indices], cols[pos_indices]])
    picked_cols = np.hstack([cols[pos_indices], rows[pos_indices]])
    picked_sum = np.sum(tm[picked_rows, picked_cols])
    tm[picked_rows, picked_cols] *= 9 * (original_sum-picked_sum) / picked_sum
    tm *= original_sum / np.sum(tm)
    return tm

tm = np.ones((5, 5))
print(tm)
tm = process_tm(tm, 1)
print(tm)





if __name__ == '__main__':
    for folder in os.listdir(TM_DIR):
        folder_path = os.path.join(TM_DIR,folder)
        for filename in os.listdir(folder_path):
            vals = os.path.basename(fname)[:-4].split('_')
            model, seed, scale_factor = vals[1], int(vals[2]), float(vals[3])
            vals = vals[4:]
            
        
        
    


    