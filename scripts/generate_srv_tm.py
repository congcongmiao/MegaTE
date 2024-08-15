import numpy as np
import pickle
import sys
sys.path.append('..')
import os
from lib.config import TL_DIR, TOPOLOGIES_DIR, TM_DIR, edge_disjoint, num_paths
from benchmarks.benchmark_consts import PROBLEM_NAMES
import argparse

SITE_TM_DIR = os.path.join(TM_DIR, 'site-traffic-matrices')


def get_topo_and_server():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topo',type=str,required=True)
    parser.add_argument('--server',type=int,required=True)
    args = parser.parse_args()
    topo, n = args.topo, args.server
    return topo, n
    

class trafficMatrix:
    def __init__(self):      
        self.siteNum = None                         #site amount
        self.serverNum = 0                          #server amount
        self.demands = None                      #site TM
        self.demandLambda = 0                       #server TM poisson
        self.trafficMat = None                      #server TM
        self.serverDistribution = None              #every site server num
        self.serverSite = None                      #every server's site
        
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
                    serverTrafficMat[TXcurrentPos:TXcurrentPos+TXservern, RXcurrentPos:RXcurrentPos+RXservern] = 0
                else:
                    trafficSum = np.sum(serverTrafficMat[TXcurrentPos:TXcurrentPos+TXservern, RXcurrentPos:RXcurrentPos+RXservern])
                    norm = 0
                    if trafficSum != 0:
                        norm = self.demands[TXid, RXid] / trafficSum
                    serverTrafficMat[TXcurrentPos:TXcurrentPos+TXservern, RXcurrentPos:RXcurrentPos+RXservern] *= norm              
                RXcurrentPos += RXservern
            groupBeginning[TXid] = TXcurrentPos
            TXcurrentPos += TXservern
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
    def get_server_distribution(self):
        return self.serverDistribution
    def get_traffic(self):
        return self.trafficMat
    def write_traffic(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.trafficMat, f)



if __name__ == '__main__':

    topology, n_servers = get_topo_and_server()      
    
    SRV_TM_DIR = os.path.join(TM_DIR, 'srv-traffic-matrices-{}'.format(n_servers)) 
    
    if not os.path.exists(SRV_TM_DIR):
        os.makedirs(SRV_TM_DIR)
    
        
    for folder in os.listdir(SITE_TM_DIR):
        '''if folder == 'triangle.json_uniform_480528682_1.0_1500.0_traffic-matrix.pkl' or folder == 'triangle.json_generic_0_1.0__traffic-matrix.pkl':
            print(folder)
            fname = os.path.join(TM_DIR, folder)
            srv_fname = SRV_TM_DIR +'/'+ folder
            with open(TOPOLOGIES_DIR+'/server2site-1/'+'triangle.json'+'_server2site.pkl','rb') as f:
                n = pickle.load(f)
            print(n)
            TM = trafficMatrix()
            
            TM.read_demands(fname)
            TM.set_server_distribution(n)
            TM.set_lambda(2)
            tm = TM.generate_traffic()
            #print(tm)
            TM.write_traffic(srv_fname)
        else:'''
        folder_path = os.path.join(SITE_TM_DIR, folder)
        srv_folder_path = os.path.join(SRV_TM_DIR, folder)
        if not os.path.exists(srv_folder_path):
            os.makedirs(srv_folder_path)
        for filename in os.listdir(folder_path):
            if filename.startswith('Top'):
                continue
            topo = filename.split("_")[0]
            #if topo != topology:
            #    continue
            
            npath = os.path.join(TOPOLOGIES_DIR, 'server2site-{}'.format(n_servers), '{}_server2site.pkl'.format(topo))
            
            with open(npath,'rb') as f:
                n = pickle.load(f)
            print(n)
            
            if filename.endswith(".pkl"):
                #if filename.startswith('b4'):
                fname = os.path.join(folder_path, filename)
                srv_fname = os.path.join(srv_folder_path, filename)
                #srv_fname = srv_folder_path + '/'+ filename
                TM = trafficMatrix()
                TM.read_demands(fname)
                TM.set_server_distribution(n)
                TM.set_lambda(2)
                tm = TM.generate_traffic()
                #print(tm)
                TM.write_traffic(srv_fname)
        
        
#n = np.array([-1, -1, -1, 1, 2, 1, 2, 2, 2])
#demands = np.array([[0, 2, 1], [3, 0, 2], [1, 4.0, 0]])
#TM = trafficMatrix()
#TM.set_server_distribution(n)
#TM.set_demands(demands)
#TM.set_lambda(2)