#Add lp-top on v3.1
#Add progress bar
import math
import numpy as np
import pickle
import time
import os
from alive_progress import alive_bar
from heapq import heappush, heappop
import os
from lib.config import TL_DIR, TOPOLOGIES_DIR, TM_DIR, edge_disjoint, num_paths
from benchmarks.benchmark_consts import TOP_HYPERPARAMS


class SSP():
    def __init__(self, tm, clas, path,  ref_table, delay = 10, epsilon = 0.05, clas_mode = False, top_mode = True, debug = False):
        self.epsilon = epsilon
        self.threads_num = 14
        self.pf = []
        self.top_lp = TOP_HYPERPARAMS 
        self.top_flag = top_mode
        self.tm = tm
        self.clas_mode = clas_mode
        if delay == 0:
            self.delay_flag = False
        else:
            self.delay_flag = True
        self.clas = clas
        self.delay = delay
        self.dpt = 0
        self.ct = 0
        self.gt = 0
        self.out = {}
        self.lrts = {
            'dp':[],
            'ga':[],
            'tt':[]
        }
        self.path = path
        self.path_list = []
        self.flow_list = []
        self.ref_table = ref_table
        self.debug = debug
        self.node_num = len(ref_table)
        self.result_table = np.zeros((self.node_num, self.node_num))
        np.ndarray.fill(self.result_table, -1)
        self.inv_table = []
        self.inv_cluster = []
        self.init_inverse_table()
        self.init_path_demand_list()
        self.flow_sum = sum(self.flow_list)
        
    def other_top(self):
        for index, (flow, path) in enumerate(self.pf[int(len(self.pf)*self.top_lp):]):
            ind = index + int(len(self.pf)*self.top_lp)
            for i in self.inv_table[path[0]]:
                for j in self.inv_table[path[-1]]:
                    if self.result_table[i,j] != -1:
                        continue
                    if (self.ref_table[i],self.ref_table[j]) not in self.path:
                        continue
                    if self.tm[i,j] < flow:
                        flow -= self.tm[i,j]
                        self.result_table[i,j] = ind

    def init_inverse_table(self):
        for srv_id, site_id in enumerate(self.ref_table):
            if site_id == -1:
                self.inv_table.append([])
                continue
            else:
                self.inv_table[site_id].append(srv_id)
    
    def init_path_demand_list(self):

        for path_data in self.path.values():
            for (flow, path) in path_data:
                self.pf.append((flow, path))

        self.pf.sort()
        for (flow, path) in self.pf:
            self.path_list.append(path)
            self.flow_list.append(flow)
        
    def single_ssp(self, site_flow, site_pair, path_num):
        def next_tm(srv_pair):
            if srv_pair[1] == self.inv_table[site_pair[1]][-1]:
                return (srv_pair[0] + 1, self.inv_table[site_pair[1]][0])
            else:
                return (srv_pair[0], srv_pair[1] + 1)

        act = time.time()
        #tm_table = np.zeros((self.node_num, self.node_num, 3))
        cluster_board = self.epsilon * site_flow /3
        delta = cluster_board * (self.epsilon /3)
        if self.debug:
            print(f"------------ SSP {path_num} ------------")
            print("site pair:", site_pair)
            print("srv board:" + str(cluster_board))
            print("site flow:" + str(site_flow))
            '''
            srv_sum = 0
            for i in self.inv_table[site_pair[0]]:
                for j in self.inv_table[site_pair[1]]:
                    pass#srv_sum += self.tm[i,j]
            print("srv mean:" + str(srv_sum/(len(self.inv_table[site_pair[0]])*len(self.inv_table[site_pair[1]]))))
            '''
        
        cluster_srv = []
        cluster_srv_sum = []
        sum = 0
        for i in self.inv_table[site_pair[0]]:
            for j in self.inv_table[site_pair[1]]:
                if self.result_table[i,j] != -1: 
                    continue
                if (self.ref_table[i],self.ref_table[j]) not in self.path:
                    continue
                sum += self.tm[i,j]
                if sum >= cluster_board:
                    cluster_srv.append((i,j))
                    cluster_srv_sum.append(sum)
                    sum = 0
                else:
                    pass
                    #cluster_srv[len(cluster_srv_sum)].append((i,j))
        bct = time.time()

            #tm_table[i,j,0] = len(cluster_srv)
            #cluster_srv.append(sum)
        if cluster_srv_sum == []:
            if self.debug:
                print("No demand left")
            return
        if self.debug:
            print("cluster srv num:", len(cluster_srv_sum))
            #print("cluster srv mean:", np.mean(cluster_srv))

        res_srv = []
        dp_srv =[]
        scaled_dp_srv = []
        dp2cluster = []
        res2cluster = []
        for index, c_srv in enumerate(cluster_srv_sum):
            if c_srv <= cluster_board:
                #tm_table[i,j,1] = len(res_srv)
                res_srv.append(c_srv)
                res2cluster.append(index)
            else:
                #tm_table[i,j,2] = len(dp_srv)
                dp_srv.append(c_srv)
                scaled_dp_srv.append(math.ceil(c_srv/delta))
                dp2cluster.append(index)
        
        
        f = np.zeros((len(scaled_dp_srv) + 1,math.floor(site_flow/delta) + 1))
        s = np.zeros((math.floor(site_flow/delta) + 1,len(scaled_dp_srv) + 1))
        adp = time.time()
        for item in range(len(scaled_dp_srv)):
            for bagg in range(math.floor(site_flow/delta)):
                i = item + 1
                j = bagg + 1
                if j < scaled_dp_srv[item]:
                    f[i][j] = f[i - 1][j]
                else:
                    if f[i - 1][j] >= f[i - 1][j - scaled_dp_srv[item]] + scaled_dp_srv[item]:
                        f[i][j] = f[i - 1][j]
                        s[j][i] = 0
                    else:
                        s[j][i] = 1
                        s[j][:i] = s[j - scaled_dp_srv[item]][:i]
                        f[i][j] = f[i - 1][j - scaled_dp_srv[item]] + scaled_dp_srv[item]
        bdp = time.time()
        agt = time.time()
        dp_srv_sum = 0
        add_tm = 0

        for index in range(len(s[math.floor(site_flow/delta)][1:])):
            if s[math.floor(site_flow/delta)][1:][index] == 1:
                dp_srv_sum += dp_srv[index]
                (endi, endj) = next_tm(cluster_srv[dp2cluster[index]])
                if dp2cluster[index] == 0:
                    (starti, startj) = (self.inv_table[site_pair[0]][0], self.inv_table[site_pair[1]][0])
                else:
                    (starti, startj) = next_tm(cluster_srv[dp2cluster[index] - 1])
                for i in range(starti, endi + 1):
                    if i == starti and i == endi:
                        var_startj = startj
                        var_end = endj
                    elif i == starti:
                        var_startj = startj
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    elif i == endi:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = endj
                    else:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    for j in range(var_startj, var_end):
                        self.result_table[i,j] = path_num
                        self.flow_list[path_num] -= self.tm[i,j]
                        add_tm += 1
        

        bgt = time.time()
        res_demand = site_flow - dp_srv_sum
        res_srv.sort(reverse=True)
        for index, r_srv in enumerate(res_srv):
            if r_srv < res_demand:
                #rr += 1
                (endi, endj) = next_tm(cluster_srv[res2cluster[index]])
                if res2cluster[index] == 0:
                    (starti, startj) = (self.inv_table[site_pair[0]][0], self.inv_table[site_pair[1]][0])
                else:
                    (starti, startj) = next_tm(cluster_srv[dp2cluster[index] - 1])
                for i in range(starti, endi + 1):
                    if i == starti and i == endi:
                        var_startj = startj
                        var_end = endj
                    elif i == starti:
                        var_startj = startj
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    elif i == endi:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = endj
                    else:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    for j in range(var_startj, var_end):
                        self.result_table[i,j] = path_num
                        self.flow_list[path_num] -= self.tm[i,j]
                        add_tm += 1

        for index in range(len(s[math.floor(site_flow/delta)][1:])):
            if s[math.floor(site_flow/delta)][1:][index] == 0:
                dp_srv_sum += dp_srv[index]
                (endi, endj) = cluster_srv[dp2cluster[index]]
                endj += 1
                if dp2cluster[index] == 0:
                    (starti, startj) = (self.inv_table[site_pair[0]][0], self.inv_table[site_pair[1]][0])
                else:
                    (starti, startj) = cluster_srv[dp2cluster[index] - 1]
                    startj += 1
                for i in range(starti, endi + 1):
                    if i == starti and i == endi:
                        var_startj = startj
                        var_end = endj
                    elif i == starti:
                        var_startj = startj
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    elif i == endi:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = endj
                    else:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    for j in range(var_startj, var_end):
                        if self.tm[i,j] < self.flow_list[path_num]:
                            self.result_table[i,j] = path_num
                            self.flow_list[path_num] -= self.tm[i,j]
                            add_tm += 1

        self.dpt += bdp - adp
        self.ct += bct - act
        self.gt += bgt - agt
        self.lrts['dp'].append(bdp - adp)
        self.lrts['ga'].append(bgt - agt)
        self.lrts['tt'].append(bdp - adp + bct - act + bgt - agt)
        if self.debug:
            print('DP time:', bdp - adp)
            print('Cluster time:', bct - act)
            print('GA time:', bgt - agt)
            print('Singal time:', bdp - adp + bct - act + bgt - agt)
            print('Added server demand:', add_tm)
            print(f'Total flow:{site_flow}, res flow:{res_demand}, srv num:{(len(self.inv_table[site_pair[0]]), len(self.inv_table[site_pair[0]]))}')

    def class_greedy(self, clas_num):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.ref_table[i] == -1 or self.ref_table[j] == -1 or self.ref_table[i] == self.ref_table[j]:
                    continue
                if (self.ref_table[i],self.ref_table[j]) not in self.path:
                    continue
                if self.clas[i,j] == clas_num:
                    m_path = ''
                    if self.delay_flag == True:
                        mini_delay = 1000000
                        for _, site_p in self.path[(self.ref_table[i],self.ref_table[j])]:
                            if mini_delay > self.delay(str(site_p)) and self.flow_list[self.path_list.index(site_p)] < self.tm[i,j]:
                                m_path = site_p
                    else:
                        for (_, site_p) in self.path[(self.ref_table[i],self.ref_table[j])]:
                            if self.flow_list[self.path_list.index(site_p)] > self.tm[i,j]:
                                m_path = site_p
                                break
                    path_pos = self.path_list.index(m_path)
                    self.flow_list[path_pos] -= self.tm[i,j]
                    self.result_table[i,j] = path_pos
    
    
    def show_class_delay(self):
        class_average_delay = {}
        class_delay_num = {}
        for i in range(self.node_num[0]):
            for j in range(self.node_num[1]):
                if self.result_table[i,j] != -1:
                    if str(self.clas[i,j]) in class_average_delay:
                        class_average_delay[str(self.clas[i,j])] += len(self.path_list[int(self.result_table[i,j])])
                        class_delay_num[str(self.clas[i,j])] += 1
                    else:
                        class_average_delay[str(self.clas[i,j])] = len(self.path_list[int(self.result_table[i,j])])
                        class_delay_num[str(self.clas[i,j])] = 1
        for key in class_average_delay.keys:
            class_average_delay[key] /= class_delay_num[key]
        print('delay:', class_average_delay)
    
    
    

    def solve(self):
        print('------------ SSP Enable ------------')
        if self.debug:
            print('path list:', self.path_list)
        print('Total path num:', len(self.path_list))
        print('Total node num:', self.node_num)
        if self.clas_mode == True:
            for i in range(1):
                print(f'------------ Class {i} ------------')
                self.class_greedy(i)
        if self.top_flag == True:
            with alive_bar(len(self.flow_list[:int(len(self.pf)*self.top_lp)])) as bar:
                for path_num, flow in enumerate(self.flow_list[:int(len(self.pf)*self.top_lp)]):
                        self.single_ssp(flow, (self.path_list[path_num][0],self.path_list[path_num][-1]), path_num)
                        bar()
        else:
            with alive_bar(len(self.flow_list)) as bar:
                for path_num, flow in enumerate(self.flow_list):
                        self.single_ssp(flow, (self.path_list[path_num][0],self.path_list[path_num][-1]), path_num)
                        bar()
        if self.top_flag == True:
            self.other_top()
        #self.show_class_delay()

    def output(self, output_dict = {}):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.result_table[i,j] == -1:
                    continue
                srv_pair = (i,j)
                srv_path = [i] + self.path_list[int(self.result_table[i,j])] + [j]
                output_dict[srv_pair] = srv_path
            self.out = output_dict
        if self.debug:
            print('------------ output ------------')
            #print(output_dict)
    
    def save(self, file_path, filename):
        if filename.endswith('ssp_solution.pkl'):
            print('WARN: The file may be overwritten with the same name')
        if self.out == {}:
            print('ERROR: No output produced')
        else:
            with open(file_path + '/' + filename, 'wb') as f:
                pickle.dump(self.out, f)
    
    def pathnum(self):
        return len(self.flow_list)
    
    def total_demand(self):
        td = 0
        for i in range(self.node_num):
            for j in range(self.node_num):
                if (self.ref_table[i],self.ref_table[j]) not in self.path:
                    continue
                if self.result_table[i,j] != -1:
                    td += self.tm[i,j]
        return td

    def tm_sum(self):
        ts = 0
        for num in range(self.node_num):
            if self.ref_table[num] != -1:
                index = num
                break
        for i in range(index, self.node_num):
            for j in range(index, self.node_num):
                ts += self.tm[i,j]
        return ts

    def runtime(self):
    
        def part_runtime(part):
            self.lrts[part].sort(reverse=True)
            h = []
            for rt in self.lrts[part][:self.threads_num]:
                heappush(h, rt)
            curr_rt = 0
            for rt in self.lrts[part][self.threads_num:]:
                curr_rt = heappop(h)
                heappush(h, rt + curr_rt)
            while len(h) > 0:
                curr_rt = heappop(h)
            return curr_rt
        
        dp_rt = part_runtime('dp')
        ga_rt = part_runtime('ga')
        if self.debug:
            print('DP est time:', dp_rt)
            print('GA est time:', ga_rt)
        total_rt = part_runtime('tt')
        return total_rt

# to deal with the problem of Memory overflow, use SSP_PART to read and
# calculate a series of partitial TM 
        
class SSP_PART:
    def __init__(self,
                 tm_path,
                 path_path,
                 ref_table_path,
                 clas_path = "",
                 delay_path = "",
                 epsilon = 0.1,
                 delay_mode = False,
                 clas_mode = False,
                 top_mode = True,
                 debug = False,
                 start_site_pair = (0,0)):
        
        self.threads_num = 16
        self.top_lp = 0.1
        self.top_flag = top_mode
        self.clas_mode = clas_mode
        self.delay_flag = delay_mode
        self.epsilon = epsilon

        self.dpt = 0
        self.ct = 0
        self.gt = 0
        self.pf = []
        self.out = {}
        self.lrts = {
            "dp":[],
            "ga":[],
            "tt":[]
        }
        self.path_list = []
        self.flow_list = []
        self.part_previous_path = 0
        self.debug = debug
        self.inv_table = []
        self.inv_cluster = []
        if clas_mode:
            self.clas = self.from_file(clas_path)
        if delay_mode:
            self.delay = self.from_file(delay_path)
        self.ref_table = self.from_file(ref_table_path)
        self.tm = self.from_file(tm_path)
        self.node_num = (len(self.tm[:,0]),len(self.tm[0,:]))
        self.path = self.from_file(path_path)
        self.result_table = np.zeros((self.node_num[0] + 1, self.node_num[1] + 1))
        np.ndarray.fill(self.result_table, -1)
        self.init_inverse_table()
        self.init_path_demand_list()
        self.start_site_pair = start_site_pair
        self.start_srv_pair = (self.inv_table[self.start_site_pair[0]][0], self.inv_table[self.start_site_pair[1]][0])

    def other_top(self):
        with alive_bar(len(self.pf[int(len(self.pf)*self.top_lp):])) as bar:
            for index, (flow, path) in enumerate(self.pf[int(len(self.pf)*self.top_lp):]):
                ind = index + int(len(self.pf)*self.top_lp)
                for i in self.inv_table[path[0]]:
                    for j in self.inv_table[path[-1]]:
                        if self.result_table[i, j] != -1:
                            continue
                        if (self.ref_table[i],self.ref_table[j]) not in self.path:
                            continue
                        if self.tm[i, j] < flow:
                            flow -= self.tm[i, j]
                            self.result_table[i, j] = ind
            bar()

    def init_inverse_table(self):
        for srv_id, site_id in enumerate(self.ref_table):
            if site_id == -1:
                self.inv_table.append([])
                continue
            else:
                self.inv_table[site_id].append(srv_id)

    def flow_sum(self):
        f_s = 0
        for path_data in self.path.values():
            for (flow, _) in path_data:
                f_s += flow
        return f_s

    def init_path_demand_list(self):

        for path_data in self.path.values():
            for (flow, path) in path_data:
                self.pf.append((flow, path))

        self.pf.sort()
        for (flow, path) in self.pf:
            self.path_list.append(path)
            self.flow_list.append(flow)
        
    def single_ssp(self, site_flow, site_pair, path_num):
        def next_tm(srv_pair):
            if srv_pair[1] == self.inv_table[site_pair[1]][-1]:
                return (srv_pair[0] + 1, self.inv_table[site_pair[1]][0])
            else:
                return (srv_pair[0], srv_pair[1] + 1)

        act = time.time()
        #tm_table = np.zeros((self.node_num, self.node_num, 3))
        cluster_board = self.epsilon * site_flow /3
        delta = cluster_board * (self.epsilon /3)
        if self.debug:
            print(f"------------ Path ID {path_num} ------------")
            print("site pair:", site_pair)
            print("srv board:" + str(cluster_board))
            print("site flow:" + str(site_flow))
            '''
            srv_sum = 0
            for i in self.inv_table[site_pair[0]]:
                for j in self.inv_table[site_pair[1]]:
                    pass#srv_sum += self.tm[i,j]
            print("srv mean:" + str(srv_sum/(len(self.inv_table[site_pair[0]])*len(self.inv_table[site_pair[1]]))))
            '''
        
        cluster_srv = []
        cluster_srv_sum = []
        sum = 0
        
        for i in self.inv_table[site_pair[0]]:
            for j in self.inv_table[site_pair[1]]:
                
                if self.result_table[i - self.start_srv_pair[0], j - self.start_srv_pair[1]] != -1: 
                    continue
                if (self.ref_table[i],self.ref_table[j]) not in self.path:
                    continue
                sum += self.tm[i - self.start_srv_pair[0], j - self.start_srv_pair[1]]
                if sum >= cluster_board:
                    cluster_srv.append((i,j))
                    cluster_srv_sum.append(sum)
                    sum = 0
                else:
                    pass
                    #cluster_srv[len(cluster_srv_sum)].append((i,j))
        bct = time.time()

            #tm_table[i,j,0] = len(cluster_srv)
            #cluster_srv.append(sum)
        if cluster_srv_sum == []:
            if self.debug:
                print("No demand left")
            return
        if self.debug:
            print("cluster srv num:", len(cluster_srv_sum))
            #print("cluster srv mean:", np.mean(cluster_srv))

        res_srv = []
        dp_srv =[]
        scaled_dp_srv = []
        dp2cluster = []
        res2cluster = []
        for index, c_srv in enumerate(cluster_srv_sum):
            if c_srv <= cluster_board:
                #tm_table[i,j,1] = len(res_srv)
                res_srv.append(c_srv)
                res2cluster.append(index)
            else:
                #tm_table[i,j,2] = len(dp_srv)
                dp_srv.append(c_srv)
                scaled_dp_srv.append(math.ceil(c_srv/delta))
                dp2cluster.append(index)
        
        
        f = np.zeros((len(scaled_dp_srv) + 1,math.floor(site_flow/delta) + 1))
        s = np.zeros((math.floor(site_flow/delta) + 1,len(scaled_dp_srv) + 1))
        adp = time.time()
        for item in range(len(scaled_dp_srv)):
            for bagg in range(math.floor(site_flow/delta)):
                i = item + 1
                j = bagg + 1
                if j < scaled_dp_srv[item]:
                    f[i][j] = f[i - 1][j]
                else:
                    if f[i - 1][j] >= f[i - 1][j - scaled_dp_srv[item]] + scaled_dp_srv[item]:
                        f[i][j] = f[i - 1][j]
                        s[j][i] = 0
                    else:
                        s[j][i] = 1
                        s[j][:i] = s[j - scaled_dp_srv[item]][:i]
                        f[i][j] = f[i - 1][j - scaled_dp_srv[item]] + scaled_dp_srv[item]
        bdp = time.time()
        agt = time.time()
        dp_srv_sum = 0
        add_tm = 0

        for index in range(len(s[math.floor(site_flow/delta)][1:])):
            if s[math.floor(site_flow/delta)][1:][index] == 1:
                dp_srv_sum += dp_srv[index]
                (endi, endj) = next_tm(cluster_srv[dp2cluster[index]])
                if dp2cluster[index] == 0:
                    (starti, startj) = (self.inv_table[site_pair[0]][0], self.inv_table[site_pair[1]][0])
                else:
                    (starti, startj) = next_tm(cluster_srv[dp2cluster[index] - 1])
                for i in range(starti, endi + 1):
                    if i == starti and i == endi:
                        var_startj = startj
                        var_end = endj
                    elif i == starti:
                        var_startj = startj
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    elif i == endi:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = endj
                    else:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    for j in range(var_startj, var_end):
                        self.result_table[i - self.start_srv_pair[0],j - self.start_srv_pair[1]] = path_num
                        self.flow_list[path_num] -= self.tm[i - self.start_srv_pair[0],j - self.start_srv_pair[1]]
                        add_tm += 1
        

        bgt = time.time()
        res_demand = site_flow - dp_srv_sum
        res_srv.sort(reverse=True)
        for index, r_srv in enumerate(res_srv):
            if r_srv < res_demand:
                #rr += 1
                (endi, endj) = next_tm(cluster_srv[res2cluster[index]])
                if res2cluster[index] == 0:
                    (starti, startj) = (self.inv_table[site_pair[0]][0], self.inv_table[site_pair[1]][0])
                else:
                    (starti, startj) = next_tm(cluster_srv[dp2cluster[index] - 1])
                for i in range(starti, endi + 1):
                    if i == starti and i == endi:
                        var_startj = startj
                        var_end = endj
                    elif i == starti:
                        var_startj = startj
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    elif i == endi:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = endj
                    else:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    for j in range(var_startj, var_end):
                        self.result_table[i,j] = path_num
                        self.flow_list[path_num] -= self.tm[i,j]
                        add_tm += 1

        for index in range(len(s[math.floor(site_flow/delta)][1:])):
            if s[math.floor(site_flow/delta)][1:][index] == 0:
                dp_srv_sum += dp_srv[index]
                (endi, endj) = cluster_srv[dp2cluster[index]]
                endj += 1
                if dp2cluster[index] == 0:
                    (starti, startj) = (self.inv_table[site_pair[0]][0], self.inv_table[site_pair[1]][0])
                else:
                    (starti, startj) = cluster_srv[dp2cluster[index] - 1]
                    startj += 1
                for i in range(starti, endi + 1):
                    if i == starti and i == endi:
                        var_startj = startj
                        var_end = endj
                    elif i == starti:
                        var_startj = startj
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    elif i == endi:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = endj
                    else:
                        var_startj = self.inv_table[site_pair[1]][0]
                        var_end = self.inv_table[site_pair[1]][-1] + 1
                    for j in range(var_startj, var_end):
                        if self.tm[i - self.start_srv_pair[0],j - self.start_srv_pair[1]] < self.flow_list[path_num]:
                            self.result_table[i - self.start_srv_pair[0],j - self.start_srv_pair[1]] = path_num
                            self.flow_list[path_num] -= self.tm[i - self.start_srv_pair[0],j - self.start_srv_pair[1]]
                            add_tm += 1

        self.dpt += bdp - adp
        self.ct += bct - act
        self.gt += bgt - agt
        self.lrts["dp"].append(bdp - adp)
        self.lrts["ga"].append(bgt - agt)
        self.lrts["tt"].append(bdp - adp + bct - act + bgt - agt)
        if self.debug:
            print("DP time:", bdp - adp)
            print("Cluster time:", bct - act)
            print("GA time:", bgt - agt)
            print("Singal time:", bdp - adp + bct - act + bgt - agt)
            print("Added server demand:", add_tm)
            print(f"Total flow:{site_flow}, res flow:{res_demand}, srv num:{(len(self.inv_table[site_pair[0]]), len(self.inv_table[site_pair[0]]))}")

    def class_greedy(self, clas_num):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.ref_table[i] == -1 or self.ref_table[j] == -1 or self.ref_table[i] == self.ref_table[j]:
                    continue
                if (self.ref_table[i],self.ref_table[j]) not in self.path:
                    continue
                if self.clas[i,j] == clas_num:
                    m_path = ""
                    if self.delay_flag == True:
                        mini_delay = 1000000
                        for _, site_p in self.path[(self.ref_table[i],self.ref_table[j])]:
                            if mini_delay > self.delay(str(site_p)) and self.flow_list[self.path_list.index(site_p)] < self.tm[i,j]:
                                m_path = site_p
                    else:
                        for (_, site_p) in self.path[(self.ref_table[i],self.ref_table[j])]:
                            if self.flow_list[self.path_list.index(site_p)] > self.tm[i,j]:
                                m_path = site_p
                                break
                    path_pos = self.path_list.index(m_path)
                    self.flow_list[path_pos] -= self.tm[i,j]
                    self.result_table[i,j] = path_pos

    def solve(self):
        print("------------ SSP Enable ------------")
        if self.debug:
            print("path list:", self.path_list)
        print("Total path num:", len(self.path_list))
        print("Total node num:", self.node_num)
        if self.clas_mode == True:
            for i in range(1):
                print(f"------------ Class {i} ------------")
                self.class_greedy(i)
        if self.top_flag == True:
            with alive_bar(len(self.flow_list[:int(len(self.pf)*self.top_lp)])) as bar:
                for path_num, flow in enumerate(self.flow_list[:int(len(self.pf)*self.top_lp)]):
                        self.single_ssp(flow, (self.path_list[path_num][0],self.path_list[path_num][-1]), path_num)
                        bar()
        else:
            with alive_bar(len(self.flow_list)) as bar:
                for path_num, flow in enumerate(self.flow_list):
                        self.single_ssp(flow, (self.path_list[path_num][0],self.path_list[path_num][-1]), path_num)
                        bar()
        if self.top_flag == True:
            self.other_top()

    def output(self):
        output_dict = {}
        for i in range(self.node_num[0]):
            for j in range(self.node_num[1]):
                if self.result_table[i,j] == -1:
                    continue
                srv_pair = (i + self.start_srv_pair[0],j + self.start_srv_pair[1])
                srv_path = [i + self.start_srv_pair[0]] + self.path_list[int(self.result_table[i,j])] + [j + self.start_srv_pair[1]]
                output_dict[srv_pair] = srv_path
            self.out = output_dict
        if self.debug:
            print("------------ output ------------")
            #print(output_dict)
    
    def save(self, file_path, filename):
        if filename == "ssp_solution.pkl":
            print("WARN: The file may be overwritten with the same name")
        if self.out == {}:
            print("ERROR: No output produced")
        else:
            with open(file_path + filename, 'wb') as f:
                pickle.dump(self.out, f)
    
    def pathnum(self):
        return len(self.flow_list)
    
    def total_demand(self):
        td = 0
        for i in range(self.node_num[0]):
            for j in range(self.node_num[1]):
                #if (self.ref_table[i + self.start_srv_pair[0]],self.ref_table[j + self.start_srv_pair[1]]) not in self.path:
                    #continue
                if self.result_table[i,j] != -1:
                    td += self.tm[i,j]
                    
        return td

    def tm_sum(self):
        ts = 0
        for i in range(self.node_num[0]):
            for j in range(self.node_num[1]):
                ts += self.tm[i,j]
        return ts

    def runtime(self):
    
        def part_runtime(part):
            self.lrts[part].sort(reverse=True)
            h = []
            for rt in self.lrts[part][:self.threads_num]:
                heappush(h, rt)
            curr_rt = 0
            for rt in self.lrts[part][self.threads_num:]:
                curr_rt = heappop(h)
                heappush(h, rt + curr_rt)
            while len(h) > 0:
                curr_rt = heappop(h)
            return curr_rt
        
        dp_rt = part_runtime("dp")
        ga_rt = part_runtime("ga")
        if self.debug:
            print("DP est time:", dp_rt)
            print("GA est time:", ga_rt)
        total_rt = part_runtime("tt")
        return total_rt
    
    def from_file(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
        
    def add_nums2path(self, num, path):
        return path[:-4] + "_" + str(num) + ".pkl"
    
    def write_log(self, path, index, log):
        log_list = self.from_file(path)
        if index < len(log_list):
            log_list[index] = log
        else:
            log_list.append(log)
        with open(path, 'wb') as f:
            pickle.dump(log_list, f)




class SSP_TOP_MAIN(SSP):
    def __init__(self, size, problem_name, model, traffic_seed, scale_factor, tmuse):
        self.size = size
        self.problem_name = problem_name
        self.model = model
        self.traffic_seed = traffic_seed 
        self.scale_factor = scale_factor
        self.tmuse = tmuse
        self.total_run_time = None
        self.ssp_run_time_per_path = None
        self.ssp_runtime = None
        self.LP_satisfied_flow = None
        self.SSP_satisfied_flow = None
        self.ssp_top_main(size, problem_name, model, traffic_seed,scale_factor, tmuse)
        
        
    def ssp_top_main(self, size, problem_name, model, traffic_seed,scale_factor, tmuse):
        ## ref_table: a srv-stie reflection table
        ## tm: srv-level traffic matrix
        ## path: the solution of site-level TE
        ## The output will be saved in PKL format.
    
       
        ## Run the site-level TE:
        PATHS_DIR = os.path.join(TL_DIR, 'benchmarks','Top-Mega-logs')
        
        vals = os.path.basename(tmuse)[:-4].split('_')
        param = vals[4]
        toponame = problem_name + '_server2site.pkl'
        ref_table_fname = os.path.join(TOPOLOGIES_DIR, 'server2site-{}'.format(size),toponame)
        tm_fname = os.path.join(TM_DIR, 'srv-traffic-matrices-{}'.format(size), model, 
        '{}_{}_{}_{}_{}_traffic-matrix.pkl'.format(problem_name, model, traffic_seed, scale_factor, param))
        
        path_fname = os.path.join(PATHS_DIR, problem_name, '{}-{}'.format(traffic_seed, model), '{}-per_site'.format(size))
        save_fname = path_fname
        filename = '{}-{}-ssp_solution.pkl'.format(problem_name, size)
        path_fname = os.path.join(path_fname, '{}-{}-LP-in-mega_ssp_input.pkl'.format(problem_name, size))
    
        with open(ref_table_fname, 'rb') as f:
            ref_table = pickle.load(f)
            #print('ref,', size, ref_table)
        with open(tm_fname, 'rb') as f:
            tm = pickle.load(f)
        with open(path_fname, 'rb') as f:
            path = pickle.load(f)
            
        clas = []
        
        super().__init__(tm, 
                  clas, 
                  path, 
                  ref_table, 
                  delay = 0, 
                  epsilon = 0.1,
                  clas_mode=False,
                  top_mode=True,
                  debug=False)
                  
        start_time = time.time()
        self.solve()
        end_time = time.time()
        self.output()
        self.save(save_fname, filename)
        if self.debug == True:
            print('DP Total time:', self.dpt)
            print('Cluster Total time:', self.ct)
            print('GA Total time:', self.gt)
            print('Total Time:',end_time - start_time)
            print('Average time:', (end_time - start_time) / self.pathnum())
        print('------------ SSP results ------------')
        print('SSP Time:', self.runtime())
        
        
        self.total_run_time = end_time - start_time
        self.ssp_run_time_per_path = (end_time - start_time)/self.pathnum()
        self.ssp_runtime = self.runtime()
        self.LP_satisfied_flow = self.flow_sum
        self.SSP_satisfied_flow = self.total_demand()
        
    
class SSP_MAIN(SSP):
    def __init__(self, size, problem_name, model, traffic_seed,scale_factor, tmuse):
        self.size = size
        self.problem_name = problem_name
        self.model = model
        self.traffic_seed = traffic_seed 
        self.scale_factor = scale_factor
        self.tmuse = tmuse
        self.total_run_time = None
        self.ssp_run_time_per_path = None
        self.ssp_runtime = None
        self.LP_satisfied_flow = None
        self.SSP_satisfied_flow = None
        self.ssp_main(size, problem_name, model, traffic_seed,scale_factor, tmuse)
        
        
    def ssp_main(self, size, problem_name, model, traffic_seed,scale_factor, tmuse):
        ## ref_table: a srv-stie reflection table
        ## tm: srv-level traffic matrix
        ## path: the solution of site-level TE
        ## The output will be saved in PKL format.
    
       
        ## Run the site-level TE:
        PATHS_DIR = os.path.join(TL_DIR, 'benchmarks','Mega-logs')
        
        vals = os.path.basename(tmuse)[:-4].split('_')
        param = vals[4]
        toponame = problem_name + '_server2site.pkl'
        ref_table_fname = os.path.join(TOPOLOGIES_DIR, 'server2site-{}'.format(size),toponame)
        tm_fname = os.path.join(TM_DIR, 'srv-traffic-matrices-{}'.format(size), model, 
        '{}_{}_{}_{}_{}_traffic-matrix.pkl'.format(problem_name, model, traffic_seed, scale_factor, param))
        
        path_fname = os.path.join(PATHS_DIR, problem_name, '{}-{}'.format(traffic_seed, model), '{}-per_site'.format(size))
        save_fname = path_fname
        filename = '{}-{}-ssp_solution.pkl'.format(problem_name, size)
        path_fname = os.path.join(path_fname, '{}-{}-LP-in-mega_ssp_input.pkl'.format(problem_name, size))
    
        with open(ref_table_fname, 'rb') as f:
            ref_table = pickle.load(f)
            
        with open(tm_fname, 'rb') as f:
            tm = pickle.load(f)
            print('tm,', sum(tm))
        with open(path_fname, 'rb') as f:
            path = pickle.load(f)
            #print('path,', path)
            
        clas = []
        
        super().__init__(tm, 
                  clas, 
                  path, 
                  ref_table, 
                  delay = 0, 
                  epsilon = 0.1,
                  clas_mode=False,
                  top_mode=True,
                  debug=False)
                  
        start_time = time.time()
        self.solve()
        end_time = time.time()
        self.output()
        self.save(save_fname, filename)
        if self.debug == True:
            print('DP Total time:', self.dpt)
            print('Cluster Total time:', self.ct)
            print('GA Total time:', self.gt)
            print('Total Time:',end_time - start_time)
            print('Average time:', (end_time - start_time) / self.pathnum())
        print('------------ SSP results ------------')
        print('SSP Time:', self.runtime())
        
        
        self.total_run_time = end_time - start_time
        self.ssp_run_time_per_path = (end_time - start_time)/self.pathnum()
        self.ssp_runtime = self.runtime()
        self.LP_satisfied_flow = self.flow_sum
        self.SSP_satisfied_flow = self.total_demand()
        
        
class SSP_PART_MAIN(SSP_PART):
    def __init__(self, size, problem_name, model, traffic_seed, scale_factor, tmuse):
        self.size = size
        self.problem_name = problem_name
        self.model = model
        self.traffic_seed = traffic_seed 
        self.scale_factor = scale_factor
        self.tmuse = tmuse
        self.total_run_time = None
        self.ssp_run_time_per_path = None
        self.ssp_runtime = None
        self.LP_satisfied_flow = None
        self.SSP_satisfied_flow = None
        self.ssp_top_part_main(size, problem_name, model, traffic_seed,scale_factor, tmuse)
        
    def ssp_top_part_main(self, size, problem_name, model, traffic_seed,scale_factor, tmuse):
        ## ref_table: a srv-stie reflection table
        ## tm: srv-level traffic matrix
        ## path: the solution of site-level TE
        ## The output will be saved in PKL format.
    
       
        ## Run the site-level TE
        PATHS_DIR = os.path.join(TL_DIR, 'benchmarks','Top-Mega-logs')
        
        vals = os.path.basename(tmuse)[:-4].split('_')
        param = vals[4]
        toponame = problem_name + '_server2site.pkl'
        ref_table_fname = os.path.join(TOPOLOGIES_DIR, 'server2site-{}'.format(size), toponame)
        
        path_fname = os.path.join(PATHS_DIR, problem_name, '{}-{}'.format(traffic_seed, model), '{}-per_site'.format(size))
        save_fnames = path_fname
        log_path = os.path.join(path_fname, 'log.pkl')
        
        filename = '{}-{}-ssp_solution.pkl'.format(problem_name, size)
        path_fname = os.path.join(path_fname, '{}-{}-LP-in-mega_ssp_input.pkl'.format(problem_name, size))
        
        tm_list_fname = os.path.join(TM_DIR, 'srv-traffic-matrices-{}'.format(size), model, 'split_version', '{}_split_list.pkl'.format(problem_name))    
        with open(tm_list_fname, 'rb') as f:
            tm_list = pickle.load(f)
            
        
        tm_paths = os.path.join(TM_DIR, 'srv-traffic-matrices-{}'.format(size), model, 'split_version',
        '{}_{}_{}_{}_{}_traffic-matrix.pkl'.format(problem_name, model, traffic_seed, scale_factor, param))
        
        ssp_one = 1
        ssp_sum = 0
        ssp_tm_sum = 0
        ssp_time = 0
        start_point = 0
        if start_point != 0:
            logs = SSP.from_file(log_path)
            (ssp_sum, ssp_tm_sum, ssp_time) = logs[start_point]
        start_time = time.time()
        for index, (start_site_pair, end_site_pair) in enumerate(tm_list):
            if ssp_one and index != 0:
                print('DONE')
                break
            if index < start_point: continue
            if index == 0:
                with open(log_path,"wb") as f:pickle.dump([], f)
            print("Part SSP ", index,"|ID:", start_site_pair, "-", end_site_pair)
            
            print(index, tm_paths)
            
            tm_path = self.add_nums2path(index, tm_paths)
            #save_fname = self.add_nums2path(index, save_fnames)
            save_fname = os.path.join(save_fnames, filename)[:-4] + "_" + str(index) + ".pkl"
            super().__init__(
                tm_path,
                path_fname,
                ref_table_fname,
                clas_path = "",
                delay_path = "",
                epsilon = 0.1, 
                delay_mode = False,
                clas_mode = False,
                top_mode = False, 
                debug = False,
                start_site_pair = start_site_pair
            )
            
            
            if 1:
            #with alive_bar((end_site_pair[0] - start_site_pair[0])) as bar:
                for i in range(start_site_pair[0], end_site_pair[0]):
                    for j in range(start_site_pair[1], end_site_pair[1]):
                            if (i,j) not in self.path:continue
                            paths = self.path[(i,j)]
                            for flow, path in paths:
                                path_num = self.path_list.index(path)
                                self.single_ssp(flow, (path[0],path[-1]), path_num)
                    #bar()
            
            if self.debug == True:
                print("DP Total time:", self.dpt)
                print("Cluster Total time:", self.ct)
                print("GA Total time:", self.gt)
                print("Total Time:",end_time - start_time)
                print("Average time:", (end_time - start_time)/self.pathnum())
            ssp_sum += self.total_demand()
            ssp_tm_sum += self.tm_sum()
            ssp_time += self.runtime()
            log = (ssp_sum, ssp_tm_sum, ssp_time)
            self.write_log(log_path,index,log)
            print(self.runtime())
            
            
        end_time = time.time()
        print("------------ SSP results ------------")
        print("SSP time:", ssp_time)
        print("total time:",end_time - start_time)
        print("Satisfied demand:", ssp_sum/ssp_tm_sum)
        print("MCF Flow / Demand:",self.flow_sum()/ ssp_tm_sum)
        #print(ssp_sum, self.flow_sum(),ssp_tm_sum)
       
        self.output()
        self.save(save_fname, '')        
        
        self.total_run_time = end_time - start_time
        self.ssp_run_time_per_path = (end_time - start_time)/self.pathnum()
        self.ssp_runtime = self.runtime()
        self.LP_satisfied_flow = self.flow_sum
        self.SSP_satisfied_flow = self.total_demand()
    
        
            
    
        



if __name__ == '__main__':
    main()      





        
                    

        
