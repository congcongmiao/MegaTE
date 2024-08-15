import numpy as np
import pickle
import sys
sys.path.append('..')
from lib.config import TL_DIR, TOPOLOGIES_DIR, TM_DIR, edge_disjoint, num_paths
from benchmarks.benchmark_consts import PROBLEM_NAMES
import argparse
#import glob
import os

PATH_DIR = os.path.join(TOPOLOGIES_DIR, 'paths', 'site-paths')
SITE_TOPO_DIR = os.path.join(TOPOLOGIES_DIR, 'site_topologies')

SITE_TM_DIR = os.path.join(TM_DIR, 'site-traffic-matrices')

MODELS = 'uniform'


class trafficMatrix:
    def __init__(self):      
        self.siteNum = None                         #site number
        self.serverNum = 0                          #server number
        self.demands = None                         #site TM
        self.demandLambda = 0                       #server TM poisson parameter
        self.trafficMat = None                      #server TM
        self.serverDistribution = None              #server per site
        self.serverSite = None                      #server's corresponding site
        self.splittedTM = None                      #split TM list
    def sitenum_update(self, newNum):
        if self.siteNum == None:
            self.siteNum = newNum
        else:
            try:
                assert self.siteNum == newNum
            except AssertionError:
                print('Error: number of sites %d does not match with input %d' % (self.siteNum, newNum))
                exit(1)
                
    def set_demands(self, demands):
        self.sitenum_update(demands.shape[0])        
        self.demands = demands
        
    def read_demands(self, path):                   
        with open(path,'rb') as f:
            demands = pickle.load(f)
            self.set_demands(demands)         

    def set_server_distribution(self, distribute):  
        self.sitenum_update(np.where(distribute != -1)[0][0])       
        self.serverSite = distribute[self.siteNum:]
        self.serverDistribution = np.bincount(self.serverSite)
        self.serverNum = self.serverSite.shape[0]

    def set_lambda(self, poissonLambda):
        self.demandLambda = poissonLambda
    def generate_traffic(self):       
        serverTrafficMat = np.random.poisson(self.demandLambda, (self.serverNum, self.serverNum)).astype(np.float64)
        serverTrafficMat[range(self.serverNum), range(self.serverNum)] = 0
        TXcurrentPos = 0
        groupBeginning = np.zeros(self.siteNum).astype(np.int32)
        for TXid, TXservern in enumerate(self.serverDistribution):
            RXcurrentPos = 0
            for RXid, RXservern in enumerate(self.serverDistribution):
                if TXservern == 0 or RXservern == 0:
                    continue
                if TXid == RXid:
                    serverTrafficMat[TXcurrentPos:TXcurrentPos+TXservern, RXcurrentPos:RXcurrentPos+RXservern] = 0#-1     #-1 for teal, 0 for regular TM
                else:
                    trafficSum = np.sum(serverTrafficMat[TXcurrentPos:TXcurrentPos+TXservern, RXcurrentPos:RXcurrentPos+RXservern])
                    norm = 0
                    if trafficSum != 0:
                        norm = self.demands[TXid, RXid] / trafficSum
                    serverTrafficMat[TXcurrentPos:TXcurrentPos+TXservern, RXcurrentPos:RXcurrentPos+RXservern] *= norm              
                RXcurrentPos += RXservern
            groupBeginning[TXid] = TXcurrentPos
            TXcurrentPos += TXservern
        #serverTrafficMat[range(self.serverNum), range(self.serverNum)] = -1             #-1 for teal
        totalNum = self.siteNum + self.serverNum
        #self.trafficMat = np.zeros((totalNum, totalNum))
        #self.trafficMat[self.siteNum:, self.siteNum:] = serverTrafficMat
        self.trafficMat = serverTrafficMat
        return self.trafficMat
        '''
        serverid_to_Groupid = np.zeros(self.serverNum).astype(np.int32)
        filledNum = np.zeros(self.siteNum).astype(np.int32)
        for serverid, siteid in enumerate(self.serverSite):
            serverid_to_Groupid[serverid] = filledNum[siteid] + groupBeginning[siteid]
            filledNum[siteid] += 1
        totalNum = self.siteNum + self.serverNum
        self.trafficMat = np.zeros((totalNum, totalNum))       
        for TXid in range(self.siteNum, totalNum):
            for RXid in range(self.siteNum, totalNum):                  
                self.trafficMat[TXid, RXid] = serverTrafficMat[serverid_to_Groupid[TXid - self.siteNum], serverid_to_Groupid[RXid - self.siteNum]]

        return self.trafficMat
        '''
    def generate_splitted_traffic(self, sitepair):
        top_left, bottom_right = sitepair
        row_ids = np.arange(top_left[0], bottom_right[0] + 1)
        col_ids = np.arange(top_left[1], bottom_right[1] + 1)
        row_distribution = self.serverDistribution[row_ids]
        col_distribution = self.serverDistribution[col_ids]
        demands = self.demands[row_ids[0]:row_ids[-1]+1, col_ids[0]:col_ids[-1]+1]
        #print(np.sum(demands))
        split_tm = np.random.poisson(self.demandLambda, (np.sum(row_distribution), np.sum(col_distribution))).astype(np.float64)
        tx_idsum = 0
        rx_idsum = 0
        for txid, txnum in enumerate(row_distribution):
            rx_idsum = 0
            for rxid, rxnum in enumerate(col_distribution):
                gen_sum = np.sum(split_tm[tx_idsum:tx_idsum + txnum, rx_idsum:rx_idsum + rxnum])
                norm = 0
                if gen_sum != 0:
                    norm = demands[txid, rxid] / gen_sum
                split_tm[tx_idsum:tx_idsum + txnum, rx_idsum:rx_idsum + rxnum] *= norm
                rx_idsum += rxnum
            tx_idsum += txnum
        #print(np.sum(split_tm))
        return split_tm
        
    def generate_site_tm(self, sitepair, split_tm):
        top_left, bottom_right = sitepair
        row_ids = np.arange(top_left[0], bottom_right[0] + 1)
        col_ids = np.arange(top_left[1], bottom_right[1] + 1)
        row_distribution = self.serverDistribution[row_ids]
        col_distribution = self.serverDistribution[col_ids]
        demands = np.zeros((row_ids.shape[0], col_ids.shape[0]))
        tx_idsum = 0
        rx_idsum = 0
        for txid, txnum in enumerate(row_distribution):
            rx_idsum = 0
            for rxid, rxnum in enumerate(col_distribution):
                gen_sum = np.sum(split_tm[tx_idsum:tx_idsum + txnum, rx_idsum:rx_idsum + rxnum])
                demands[txid, rxid] = gen_sum
                rx_idsum += rxnum
            tx_idsum += txnum
        self.demands[row_ids[0]:row_ids[-1]+1, col_ids[0]:col_ids[-1]+1] = demands

    def get_server_distribution(self):
        return self.serverDistribution
    def get_traffic(self):
        return self.trafficMat
    
        
    def write_traffic(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.trafficMat, f)




def get_topo_and_server():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topo',type=str,required=True)
    parser.add_argument('--server',type=int,required=True)
    args = parser.parse_args()
    topo, n = args.topo, args.server
    return topo, n


n_file_path="/home/ubuntu/ncflow/topologies/server2site-100/Cogentco.graphml_server2site.pkl"
site_file_path= "/home/ubuntu/ncflow/traffic-matrices/site-traffic-matrices/uniform/Cogentco.graphml_uniform_14149633_1.0_0.6_traffic-matrix.pkl"
output_list_file_path = '/home/ubuntu/ncflow/traffic-matrices/srv-traffic-matrices-100/uniform/split_list.pkl'
output_tm_file_path= '/home/ubuntu/ncflow/traffic-matrices/srv-traffic-matrices-100/uniform/splittest/Cogentcotm_'

def tm_generate_part(n_file_path, site_file_path, output_list_file_path, output_tm_file_path):
    n = None
    with open(n_file_path,'rb') as f:
        n = pickle.load(f)  
        #while n[0]==-1 :
        #    del n[0]
        print(n.shape)

    demands_list = None
    with open(site_file_path,'rb') as f:
        demands_list = pickle.load(f)

    TM = trafficMatrix()
    TM.set_lambda(2)
    TM.set_server_distribution(n)
    TM.set_demands(demands_list)

    split_size = 2
    split_num = int((TM.siteNum-1)/split_size + 1)
    rowsplit_begin = [0]
    rowsplit_end = [split_size]
    for i in range(split_num -1):
        rowsplit_begin.append(split_size*(i + 1))
        rowsplit_end.append(split_size*(i + 2))
    colsplit_begin = rowsplit_begin
    colsplit_end = rowsplit_end

    split_list = []
    for i in range(split_num):
        for j in range(split_num):
            coords = [(i*split_size,j*split_size),(min(TM.siteNum-1,i*split_size+split_size - 1),min(TM.siteNum-1,j*split_size+split_size - 1))]
            split_list.append(coords)
    with open(output_list_file_path, 'wb') as f:
        pickle.dump(split_list, f)
        


    tm_list = []
    
    for index,sitepair in enumerate(split_list):
        tm = TM.generate_splitted_traffic(sitepair)
        #tm_list.append(tm)
        with open(output_tm_file_path+str(index)+'.pkl','wb')as f:
            pickle.dump(tm, f)

    #for i, sitepair in enumerate(split_list):
    #    tm = tm_list[i]
    #    TM.generate_site_tm(sitepair, tm)


if __name__ == '__main__':
    topo, n_servers = get_topo_and_server()
    
    SRV_TM_DIR = os.path.join(TM_DIR, 'srv-traffic-matrices-{}'.format(n_servers)) 
        
    for folder in os.listdir(SITE_TM_DIR):
        folder_path = os.path.join(SITE_TM_DIR, folder)
        srv_folder_path = os.path.join(SRV_TM_DIR, folder)
        if not os.path.exists(srv_folder_path):
            os.makedirs(srv_folder_path)
        for filename in os.listdir(folder_path):
            for topo in PROBLEM_NAMES:
                if filename.startswith(topo):
                    n_file_path = os.path.join(TOPOLOGIES_DIR, 'server2site-{}'.format(n_servers), '{}_server2site.pkl'.format(topo))
                    site_file_path = os.path.join(folder_path, filename)
                    output_list_file_path = os.path.join(SRV_TM_DIR, folder, 'split_version')
                    
                    if not os.path.exists(output_list_file_path):
                        os.makedirs(output_list_file_path)
                    
                    output_list_file_path = os.path.join(SRV_TM_DIR, folder, 'split_version', '{}_split_list.pkl'.format(topo))
                    output_tm_file_path =  os.path.join(SRV_TM_DIR, folder, 'split_version', '{}_'.format(filename[:-4]))
                    tm_generate_part(n_file_path, site_file_path, output_list_file_path, output_tm_file_path)   
