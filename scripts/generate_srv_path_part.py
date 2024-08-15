import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import pickle
import sys
sys.path.append('..')
import os
from lib.config import TL_DIR, TOPOLOGIES_DIR, edge_disjoint, num_paths
from benchmarks.benchmark_consts import PROBLEM_NAMES
import argparse

PART = True

PATH_DIR = os.path.join(TOPOLOGIES_DIR, 'paths', 'site-paths')
SITE_TOPO_DIR = os.path.join(TOPOLOGIES_DIR, 'site_topologies')



def get_topo_and_server():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topo',type=str,required=True)
    parser.add_argument('--server',type=int,required=True)
    args = parser.parse_args()
    topo, n = args.topo, args.server
    return topo, n
    

class GenerateSrvLevel:
    def __init__(self, toponame, topofname, pathname, pathfname, n_servers):      
        self.toponame = toponame
        self.topofname = topofname
        self.pathname = pathname
        self.pathfname = pathfname
        self.n_servers = n_servers  
        

    def read_graph_graphml(self):
    
        fstyle = self.topofname.split('.')[-1]
    
        if fstyle == "graphml":
            file_G = nx.read_graphml(self.topofname).to_directed()
            G = []
            # Pick largest strongly connected component
            if isinstance(file_G, nx.MultiDiGraph):
                file_G = nx.DiGraph(file_G)
    
            for scc_ids in nx.strongly_connected_components(file_G):
                scc = file_G.subgraph(scc_ids)
                if len(scc) > len(G):
                    G = scc
            G = nx.convert_node_labels_to_integers(G)
            for u, v in G.edges():
                G[u][v]['capacity'] = 1000.0
    
            return G
    
        elif fstyle == 'json':
            f = open(self.topofname, 'r')
            content = f.read()
            a = json.loads(content)
    
            G = nx.DiGraph()
            for tep in a['nodes']:
                G.add_nodes_from([tep['id']])
    
            for tep in a['links']:
                G.add_edges_from([(tep['source'], tep['target'])], capacity=tep['capacity'])
    
            return G
    # nx.draw(G, with_labels=True)
    # plt.show()
    
    def add_server(self, G, site_num, server_num):
        G.add_nodes_from([server_num])
        G.add_edges_from([(site_num, server_num), (server_num, site_num)], capacity = 10000.0)
        return 0


    def add_path(self, P, pair_matrix):
        tep_P = P.copy()
        for key in P:
            for i in pair_matrix[key[0]]:
                for j in pair_matrix[key[1]]:
                    tep_P[(i, j)] = []
                    for o_path in P[key]:
                        tep_path = o_path.copy()
                        tep_path.insert(0, i)
    
                        tep_path.append(j)
    
                        tep_P[(i,j)].append(tep_path)
    
        return tep_P
    
    def add_servers(self):
    
        G = self.read_graph_graphml()
        P = pd.read_pickle(self.pathfname)
    
        site_num = G.number_of_nodes()
        pair_matrix = []
    
        for i in range(site_num):
            pair_matrix.append([])
    
        total_number_of_servers = 0
    
        N_list = [self.n_servers] * site_num
    
        print("The number of servers corresponding to each site: ")
        print(N_list)
        for tep in N_list:
            total_number_of_servers = total_number_of_servers + tep
        server2site = -np.ones(site_num + total_number_of_servers, dtype=np.int32)
        
    
        t = 0
        for i, ele in enumerate(N_list):
            for j in range(ele):
                server_num = site_num + t
                site_tep = i
                # print(site_tep)
                # site_tep = 0
                # site_tep = site_squence[i]
                self.add_server(G, site_tep, server_num)
                #G[site_tep][server_num]['id'] = 'e' + str(i*2 + edge_num)
                #G[site_tep][server_num]['id'] = 'e' + str(i*2 + edge_num + 1)
                server2site[server_num] = site_tep
                #pair_matrix[site_tep].append(server_num)
    
                if ( t + 1 ) % 1000 == 0:
                    print(str(t+1) + " servers have been added, but the file is not written until the end of the program")
                t = t + 1
        
        
        ser2site_fname = os.path.join(TOPOLOGIES_DIR, 'server2site-{}'.format(self.n_servers))
        
        if not os.path.exists(ser2site_fname):
            os.makedirs(ser2site_fname)
        
        awname = os.path.join(TOPOLOGIES_DIR, 'server2site-{}'.format(self.n_servers), '{}_server2site.pkl'.format(self.toponame))
        with open(awname, "wb") as aw:
            pickle.dump(server2site, aw)
        
        
        P = self.add_path(P, pair_matrix)
        
        if self.topofname.endswith('.graphml'):
            fwname = os.path.join(SRV_TOPO_DIR, '{}.graphml'.format(self.toponame[0:-8]))
            if not os.path.exists(fwname):
                os.makedirs(fwname)
            nx.write_graphml(G, fwname)
    
        elif self.topofname.endswith('.json'):
            fwname = os.path.join(SRV_TOPO_DIR, '{}.json'.format(self.toponame[0:-5]))
            if not os.path.exists(fwname):
                os.makedirs(fwname)
            json.dump(dict(nodes=[dict(id=n) for n in G.nodes()],
                           links=[dict(capacity=G[u][v]['capacity'], source=u, target=v) for u, v in G.edges()],
                           multigraph=False,
                           directed = True),
                      open(fwname, 'w'), indent=2)
        
        pwname = SRV_PATH_DIR + '/{}.pkl'.format(self.pathname[0:-4])

        with open(pwname, "wb") as pw:
            pickle.dump(P, pw)
        
    
    def make_only_s2s(self):
        G = self.read_graph_graphml()
        site_num = G.number_of_nodes()
        total_number_of_servers = site_num * self.n_servers
        server2site = np.ones(site_num + total_number_of_servers, dtype=np.int32)
        
        for i in range(site_num):
            server2site[i] = -1
        for i in range(site_num):
            for j in range(self.n_servers):
                server2site[site_num + i * self.n_servers + j] = i
        
        ser2site_fname = os.path.join(TOPOLOGIES_DIR, 'server2site-{}'.format(self.n_servers))
        
        if not os.path.exists(ser2site_fname):
            os.makedirs(ser2site_fname)
        
        awname = os.path.join(TOPOLOGIES_DIR, 'server2site-{}'.format(self.n_servers), '{}_server2site.pkl'.format(self.toponame))
        with open(awname, "wb") as aw:
            pickle.dump(server2site, aw)

if __name__ == '__main__':
    # fanme: The path of the graph file which can be in .graphml or .json format
    # pklname: The path of the graph file
    # N：the number of servers which need to be added
    # The multi-line comment section is for debugging, and you can try it
    topo, n_servers = get_topo_and_server()
        
    SRV_TOPO_DIR =  os.path.join(TOPOLOGIES_DIR, 'srv_topologies-{}'.format(n_servers))
    SRV_PATH_DIR =  os.path.join(TOPOLOGIES_DIR, 'paths', 'srv_paths-{}'.format(n_servers))

    if not os.path.exists(SRV_PATH_DIR):
        os.makedirs(SRV_PATH_DIR)
    if not os.path.exists(SRV_TOPO_DIR):
        os.makedirs(SRV_TOPO_DIR)

        
    for toponame in PROBLEM_NAMES:
        topofname = os.path.join(SITE_TOPO_DIR, toponame)
        pathname = '{}-{}-paths_edge-disjoint-{}_dist-metric-inv-cap-dict.pkl'.format(toponame, num_paths, edge_disjoint)
        pathfname = os.path.join(PATH_DIR, pathname)
        problem = GenerateSrvLevel(toponame, topofname, pathname, pathfname, n_servers)
        if not PART: 
            problem.add_servers()

        else:
            problem.make_only_s2s()