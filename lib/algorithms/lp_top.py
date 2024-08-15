import math
import numpy as np
import pickle
import time
import os
import networkx as nx
from networkx.readwrite import json_graph
import json
from benchmarks.benchmark_consts import TOP_HYPERPARAMS

class TOP:
    def __init__(self, tm, path, G):
        self.top = TOP_HYPERPARAMS
        self.tm = tm
        self.path = path
        self.G = G
        self.site_num = len(self.tm[:,0])
        start = time.time()
        self.create_tm_list()
        self.runtime = time.time() - start

    def create_tm_list(self):
        self.tm_list = []
        for i in range(self.site_num):
            for j in range(self.site_num):
                if i == j: continue
                self.tm_list.append((self.tm[i,j], (i,j)))
        self.tm_list.sort(reverse=True)

    def is_available(self,flow,path):
        for i in range(0, len(path)-1):
            if flow > self.G[path[i]][path[i + 1]]['capacity']:
                return False
        return True

    def shortest_path(self, site_pair, flow):
        for path in self.path[site_pair]:
            if self.is_available(flow, path):
                s_p = path
                break
        for path in self.path[site_pair]:
            if self.weight(s_p) > self.weight(path) and self.is_available(flow, path):
                s_p = path
        return s_p
    
    def weight(self, path):
        w = 0
        for i in range(0, len(path)-1):
            if self.G[path[i]][path[i + 1]]['capacity'] == 0:
                w += 1/1000000000000.0
            else:
                w += 1/self.G[path[i]][path[i + 1]]['capacity']
        return w
    
    def cut_path_in_tm(self, path,flow):
        for i in range(0, len(path)-1):
            self.G[path[i]][path[i + 1]]['capacity'] -= flow

    def solve(self):
        for flow, site_pair in self.tm_list[int(len(self.tm_list)*self.top):]:
            if site_pair not in self.path:  continue
            full_flag = 0
            for path in self.path[site_pair]:
                if self.is_available(flow, path):
                    full_flag = 1
            if not full_flag: continue
            
            shortest_p = self.shortest_path(site_pair, flow)
            self.cut_path_in_tm(shortest_p,flow)
            self.tm[site_pair[0],site_pair[1]] = 0
            del self.path[site_pair]
        print("top time:", self.runtime)

    def save(self, file_name_graph, file_name_tm, file_name_path):
        if file_name_graph.endswith(".graphml"):
            nx.write_graphml(self.G, file_name_graph)
        elif file_name_graph.endswith(".json"):
            with open(file_name_graph, "w") as f:
                json.dump(json_graph.node_link_data(self.G), f)
        else:
            print("Invalid Graph Output Format")
            return
        with open(file_name_tm, "wb") as f:
            pickle.dump(self.tm, f)
        with open(file_name_path, "wb") as f:
            pickle.dump(self.path, f)


class TOP_MAIN(TOP):
    def __init__(self, topo_fname, tm_fname, path_fname):
        self.topo_fname = topo_fname
        self.tm_fname = tm_fname
        self.path_fname = path_fname
        self.top_main(topo_fname, tm_fname, path_fname)
        
    
    def top_main(self, topo_fname,tm_fname,path_fname):
        #input
        with open(tm_fname, 'rb') as f:
            tm = pickle.load(f)
        with open(path_fname, 'rb') as f:
            path = pickle.load(f)
        graph_file_name = topo_fname
        
        if graph_file_name.endswith(".graphml"):
            file_G = nx.read_graphml(graph_file_name).to_directed()
            G = []       
            file_G = nx.DiGraph(file_G)
            for scc_ids in nx.strongly_connected_components(file_G):            
                scc = file_G.subgraph(scc_ids)            
                if len(scc) > len(G):                
                    G = scc        
                    G = nx.convert_node_labels_to_integers(G) 
            for u, v in G.edges():            
                G[u][v]['capacity'] = 1000.0
        elif graph_file_name.endswith(".json"):
            with open(graph_file_name) as f:
                data = json.load(f)
            G = json_graph.node_link_graph(data)
        else:
            print("Invalid Graph Input Format")
            return
        
        super().__init__(tm = tm,
                               path = path,
                               G = G)

        self.solve()
        
        sub_list = topo_fname.split('/')
        new_path = '/'.join(sub_list[:-1] + ['Top_version'] + sub_list[-1:])
        topo_fname = new_path
                    
        sub_list = tm_fname.split('/')
        new_path = '/'.join(sub_list[:-1] + ['Top_version'] + sub_list[-1:])
        tm_fname = new_path  
        
        sub_list = path_fname.split('/')
        new_path = '/'.join(sub_list[:-1] + ['Top_version'] + sub_list[-1:])
        path_fname = new_path  
        
    
        #output
        self.save(
            file_name_graph=topo_fname,
            file_name_tm = tm_fname,
            file_name_path= path_fname
        )
    
if __name__ == '__main__':
    main()      






        
                    

        
