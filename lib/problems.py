import os
import numpy as np
import networkx as nx
from .config import TOPOLOGIES_DIR, TM_DIR
from .problem import Problem
from .traffic_matrix import TrafficMatrix
import pickle
################
# Topology Zoo #
################
class TopologyZooProblem(Problem):
    def __init__(self, fname, *, model='gravity', seed=0, scale_factor=1.0, **kwargs):
        self._fname = fname
        G = Problem._read_graph_graphml(
            os.path.join(TOPOLOGIES_DIR, 'topology-zoo', fname))
        super().__init__(G, model=model, seed=seed, scale_factor=scale_factor, **kwargs)

    @property
    def name(self):
        return self._fname


##################
# Topology Ready #
##################
class TopologyReadyProblem(Problem):
    def __init__(self, fname, *, model='uniform', seed=0, scale_factor=1.0, **kwargs):
        self._fname = fname
        if fname.endswith('.json'):
            G = Problem._read_graph_json(
                os.path.join(TOPOLOGIES_DIR, 'site_topologies', fname))
            super().__init__(G, model=model, seed=seed, scale_factor=scale_factor, **kwargs)
        
        elif fname.endswith('.graphml'):
            G = Problem._read_graph_graphml(
                os.path.join(TOPOLOGIES_DIR, 'site_topologies', fname), False)
            super().__init__(G, model=model, seed=seed, scale_factor=scale_factor, **kwargs)

    @property
    def name(self):
        return self._fname
        



#################
# Fake problems #
#################
class ToyProblem(Problem):
    def __init__(self, *, zero_edges=[]):
        G = Problem._read_graph_json(
            os.path.join(TOPOLOGIES_DIR, 'toy-network.json'))
        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 1] = 5.0
        traffic_matrix[0, 3] = 10.0
        traffic_matrix[0, 5] = 11.0
        traffic_matrix[3, 2] = 13.0
        traffic_matrix[2, 5] = 12.0
        traffic_matrix[3, 4] = 7.0
        super().__init__(G, traffic_matrix)
        # 18 is the min capacity needed on every edge to satisfy all the demands
        super()._change_capacities(min_cap=18,
                                   max_cap=18,
                                   fixed_caps=[(u, v, 0.0)
                                               for u, v in zero_edges])

    @property
    def name(self):
        return 'toy-problem'
        
        
class TriProblem(Problem):
    def __init__(self, *, zero_edges=[]):
        G = Problem._read_graph_json(
            os.path.join(TOPOLOGIES_DIR, 'srv_topologies-2', 'triangle.json'))
        num_nodes = len(G.nodes)
        #traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        #traffic_matrix[0, 1] = 1700.0
        tmfname = "/home/ubuntu/ncflow/traffic-matrices/srv-traffic-matrices-2/uniform/triangle.json_uniform_480528682_1.0_1500.0_traffic-matrix.pkl"
        with open(tmfname,'rb') as f:
            traffic_matrix = pickle.load(f)
            print(traffic_matrix)
        super().__init__(G, traffic_matrix)
        #print(traffic_matrix)
        # 18 is the min capacity needed on every edge to satisfy all the demands
        super()._change_capacities(min_cap=0,
                                   max_cap=1000,
                                   fixed_caps=[(u, v, 0.0)
                                               for u, v in zero_edges])

    @property
    def name(self):
        return 'triangle'

        
class BiProblem(Problem):
    def __init__(self, *, zero_edges=[]):
        G = Problem._read_graph_json(
            os.path.join(TOPOLOGIES_DIR, 'bilink.json'))
        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 1] = 1700.0
        super().__init__(G, traffic_matrix)
        #print(traffic_matrix)
        # 18 is the min capacity needed on every edge to satisfy all the demands
        super()._change_capacities(min_cap=0,
                                   max_cap=1000,
                                   fixed_caps=[(u, v, 0.0)
                                               for u, v in zero_edges])

    @property
    def name(self):
        return 'triangle-problem'

        
        
        
class TriSrvProblem(Problem):
    def __init__(self, *, zero_edges=[]):
        G = Problem._read_graph_json(
            os.path.join(TOPOLOGIES_DIR, 'srv_topologies-10','triangle.json'))
        num_nodes = len(G.nodes)
        tmfname = '../traffic-matrices/srv-traffic-matrices/triangle-problem_generic_0_1.0__traffic-matrix.pkl'
        with open(tmfname,'rb') as f:
            traffic_matrix = pickle.load(f)
        print(traffic_matrix)
        #traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        #traffic_matrix[0, 1] = 1700.0
        super().__init__(G, traffic_matrix)
        #print(traffic_matrix)
        # 18 is the min capacity needed on every edge to satisfy all the demands
        #super()._change_capacities(min_cap=0,
        #                           max_cap=1000,
        #                           fixed_caps=[(u, v, 0.0)
        #                                       for u, v in zero_edges])

    @property
    def name(self):
        return 'srv-triangle-problem'


class FeasibilityProblem1(Problem):
    def __init__(self, *, zero_edges=[]):
        G = Problem._read_graph_json(
            os.path.join(TOPOLOGIES_DIR, 'feasible1.json'))
        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 6] = 10.0
        traffic_matrix[1, 7] = 10.0
        super().__init__(G, traffic_matrix)
        super()._change_capacities(min_cap=100,
                                   max_cap=100,
                                   fixed_caps=[(u, v, 0.0)
                                               for u, v in zero_edges],
                                   same_both_ways=False)

    @property
    def name(self):
        return 'FeasibilityProblem1'

class ToyProblem2(Problem):
    def __init__(self, *, zero_edges=[]):
        G = Problem._read_graph_json(
            os.path.join(TOPOLOGIES_DIR, 'toy-network-2.json'))
        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 1] = 5
        traffic_matrix[1, 3] = 10
        #traffic_matrix[0, 5] = 11
        traffic_matrix[0, 11] = 6
        #traffic_matrix[3, 2] = 13
        traffic_matrix[3, 4] = 7
        traffic_matrix[3, 8] = 4
        traffic_matrix[6, 8] = 9
        traffic_matrix[8, 4] = 12
        traffic_matrix[9, 11] = 8
        traffic_matrix[10, 2] = 13

        super().__init__(G, traffic_matrix)
        super()._change_capacities(min_cap=13,
                                   max_cap=14,
                                   fixed_caps=[(u, v, 0.0)
                                               for u, v in zero_edges])

    @property
    def name(self):
        return 'toy-problem-2'


class ReconciliationProblem(Problem):
    # Use partition_vector = [0, 0, 1, 1] to test
    # Should get 0 flow
    def __init__(self):
        G = nx.DiGraph()
        G.add_node(0, label='0', pos=(-2, 1))
        G.add_node(1, label='1', pos=(-2, 2))
        G.add_node(2, label='2', pos=(0, 2))
        G.add_node(3, label='3', pos=(0, 1))

        G.add_edge(0, 1, capacity=0)
        G.add_edge(1, 2, capacity=10)
        G.add_edge(2, 3, capacity=0)

        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 2] = 10
        traffic_matrix[1, 3] = 10

        super().__init__(G, traffic_matrix)

    @property
    def name(self):
        return 'reconciliation-problem'

class ReconciliationProblem2(Problem):
    # Use partition_vector = [0, 0, 1, 1] to test
    # Should get 0 flow
    def __init__(self):
        G = nx.DiGraph()
        G.add_node(0, label='0', pos=(-2, 1))
        G.add_node(1, label='1', pos=(-2, 2))
        G.add_node(2, label='2', pos=(0, 2))
        G.add_node(3, label='3', pos=(0, 1))

        G.add_edge(0, 1, capacity=0)
        G.add_edge(1, 2, capacity=40)
        G.add_edge(2, 3, capacity=0)

        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 2] = 10
        traffic_matrix[0, 3] = 10
        traffic_matrix[1, 2] = 10
        traffic_matrix[1, 3] = 10

        super().__init__(G, traffic_matrix)

    @property
    def name(self):
        return 'reconciliation-problem-2'

class Recon3(Problem):
    def __init__(self):
        G = nx.DiGraph()
        G.add_node(0, label='0', pos=(-2, 0))
        G.add_node(1, label='1', pos=(-1, 0.5))
        G.add_node(2, label='2', pos=(-1, -0.5))
        G.add_node(3, label='3', pos=(0, 0.5))
        G.add_node(4, label='4', pos=(0, -0.5))
        G.add_node(5, label='5', pos=(1, 0))

        G.add_edge(0, 1, capacity=1)
        G.add_edge(0, 2, capacity=5)
        G.add_edge(3, 5, capacity=5)
        G.add_edge(4, 5, capacity=1)
        G.add_edge(1, 3, capacity=100)
        G.add_edge(2, 4, capacity=100)
        G.add_edge(2, 3, capacity=3)

        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 5] = 10

        super().__init__(G, traffic_matrix)

    @property
    def name(self):
        return 'recon3'

class OptGapC3(Problem):
    def __init__(self):
        G = nx.DiGraph()
        G.add_node(0, label='0', pos=(-2, 1))
        G.add_node(1, label='1', pos=(-1, 1))
        G.add_node(2, label='2', pos=(-1.5, -1))
        G.add_node(3, label='3', pos=(0, 0))
        G.add_node(4, label='4', pos=(1, 0))
        G.add_node(5, label='5', pos=(2, 1))
        G.add_node(6, label='6', pos=(2, -1))

        G.add_edge(0, 1, capacity=1)
        G.add_edge(1, 3, capacity=100)
        G.add_edge(2, 3, capacity=100)
        G.add_edge(3, 4, capacity=8)
        G.add_edge(4, 5, capacity=100)
        G.add_edge(4, 6, capacity=100)

        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 5] = 10
        traffic_matrix[2, 6] = 10

        super().__init__(G, traffic_matrix)

    @property
    def name(self):
        return 'optgapc3'

class OptGapC1(Problem):
    def __init__(self):
        G = nx.DiGraph()
        G.add_node(0, label='0', pos=(-2, 0))
        G.add_node(1, label='1', pos=(-1, 1))
        G.add_node(2, label='2', pos=(-1, -1))
        G.add_node(3, label='3', pos=(0, 1))
        G.add_node(4, label='4', pos=(0, -1))
        G.add_node(5, label='5', pos=(1, 0))

        G.add_edge(0, 1, capacity=5)
        G.add_edge(0, 2, capacity=2)
        G.add_edge(1, 2, capacity=4)
        G.add_edge(1, 3, capacity=100)
        G.add_edge(2, 4, capacity=100)
        G.add_edge(3, 5, capacity=3)
        G.add_edge(4, 5, capacity=4)

        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 5] = 20

        super().__init__(G, traffic_matrix)

    @property
    def name(self):
        return 'optgapc1'

class OptGapC2(Problem):
    def __init__(self):
        G = nx.DiGraph()
        G.add_node(0, label='0', pos=(-2, 0))
        G.add_node(1, label='1', pos=(-1.5, 0))
        G.add_node(2, label='2', pos=(-0.5, 1))
        G.add_node(3, label='3', pos=(0, 1))
        G.add_node(4, label='4', pos=(-0.5, -1))
        G.add_node(5, label='5', pos=(0, -1))
        G.add_node(6, label='6', pos=(1, 0))
        G.add_node(7, label='7', pos=(1.5, 0))

        G.add_edge(0, 1, capacity=100)
        G.add_edge(1, 2, capacity=100)
        G.add_edge(1, 4, capacity=100)
        G.add_edge(2, 3, capacity=3)
        G.add_edge(4, 5, capacity=5)
        G.add_edge(3, 6, capacity=100)
        G.add_edge(5, 6, capacity=100)
        G.add_edge(6, 7, capacity=100)

        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 7] = 20

        super().__init__(G, traffic_matrix)

    @property
    def name(self):
        return 'optgapc2'


class OptGap4(Problem):
    def __init__(self):
        G = nx.DiGraph()
        G.add_node(0, label='0', pos=(-2, 1))
        G.add_node(1, label='1', pos=(-1, 1))
        G.add_node(2, label='2', pos=(-1, 0))
        G.add_node(3, label='3', pos=(0, 1))
        G.add_node(4, label='4', pos=(0, 0))
        G.add_node(5, label='5', pos=(1, 0.5))

        G.add_edge(0, 1, capacity=100)
        G.add_edge(1, 3, capacity=10)
        G.add_edge(2, 3, capacity=10)
        G.add_edge(2, 4, capacity=10)
        G.add_edge(3, 5, capacity=1)
        G.add_edge(4, 5, capacity=1)

        G.add_edge(1, 0, capacity=100)
        G.add_edge(3, 1, capacity=10)
        G.add_edge(3, 2, capacity=10)
        G.add_edge(4, 2, capacity=10)
        G.add_edge(5, 3, capacity=1)
        G.add_edge(5, 4, capacity=1)

        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 5] = 40

        super().__init__(G, traffic_matrix)

    @property
    def name(self):
        return 'optgap4'

class WeNeedToFixThis(Problem):
    def __init__(self, EPS=1e-3):
        G = nx.DiGraph()
        G.add_node(0, label='0', pos=(-2, 1))
        G.add_node(1, label='1', pos=(-2, 0))

        G.add_node(2, label='2', pos=(-1, 1))
        G.add_node(3, label='3', pos=(-1, 0))
        G.add_node(4, label='4', pos=(-1.5, 0.5))
        G.add_node(5, label='5', pos=(-0.5, 0.5))

        G.add_node(6, label='6', pos=(0, 1))
        G.add_node(7, label='7', pos=(0, 0))

        G.add_edge(0, 2, capacity=1)
        G.add_edge(2, 3, capacity=1)
        G.add_edge(3, 7, capacity=1)

        G.add_edge(1, 4, capacity=1)
        G.add_edge(4, 5, capacity=1)
        G.add_edge(5, 6, capacity=1)

        G.add_edge(1, 3, capacity=EPS)
        G.add_edge(2, 6, capacity=EPS)


        num_nodes = len(G.nodes)
        tm = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        tm[0, 6] = 100
        tm[1, 7] = 100

        super().__init__(G, tm)

    @property
    def name(self):
        return 'we-need-to-fix-this'

class SingleEdgeB(Problem):
    def __init__(self, EPS=1e-3):
        G = nx.DiGraph()
        G.add_node(0, label='0', pos=(-2, 1))
        G.add_node(1, label='1', pos=(-2, 0))

        G.add_node(2, label='2', pos=(-1, 1))
        G.add_node(3, label='3', pos=(-1, 0))
        G.add_node(4, label='4', pos=(-1.5, 0.5))
        G.add_node(5, label='5', pos=(-0.5, 0.5))

        G.add_node(6, label='6', pos=(0, 1))
        G.add_node(7, label='7', pos=(0, 0))

        G.add_edge(0, 2, capacity=1)
        G.add_edge(0, 1, capacity=1.2)
        G.add_edge(1, 0, capacity=1.2)
        G.add_edge(2, 3, capacity=1)
        G.add_edge(3, 7, capacity=1)

        G.add_edge(1, 4, capacity=1)
        G.add_edge(4, 5, capacity=1)
        G.add_edge(5, 6, capacity=1)

        G.add_edge(1, 3, capacity=EPS)
        G.add_edge(2, 6, capacity=EPS)
        G.add_edge(6, 7, capacity=1.2)
        G.add_edge(7, 6, capacity=1.2)

        num_nodes = len(G.nodes)
        tm = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        tm[0, 6] = 2
        tm[1, 7] = 2

        super().__init__(G, tm)

    @property
    def name(self):
        return 'SingleEdgeB'

class FlowPathConstruction(Problem):
    def __init__(self):
        G = nx.DiGraph()
        G.add_node(0, label='0', pos=(-2, 1))
        G.add_node(1, label='1', pos=(-2, 0))
        G.add_node(2, label='2', pos=(-1, 1))
        G.add_node(3, label='3', pos=(-1, 0))
        G.add_node(4, label='4', pos=(0, 1))
        G.add_node(5, label='5', pos=(0, 0))

        G.add_edge(0, 1, capacity=4)
        G.add_edge(1, 0, capacity=4)
        G.add_edge(0, 2, capacity=25)
        G.add_edge(1, 3, capacity=25)
        G.add_edge(2, 3, capacity=1)
        G.add_edge(3, 2, capacity=0)
        G.add_edge(2, 4, capacity=25)
        G.add_edge(3, 5, capacity=25)
        G.add_edge(4, 5, capacity=4)
        G.add_edge(5, 4, capacity=4)
        G.add_edge(2, 5, capacity=4)

        num_nodes = len(G.nodes)
        tm = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        tm[0, 1] = 3
        tm[0, 2] = 5
        tm[0, 3] = 6
        tm[1, 3] = 7

        tm[2, 3] = 3
        tm[2, 4] = 8
        tm[2, 5] = 9
        tm[3, 5] = 10

        tm[4, 5] = 3
        tm[0, 4] = 11
        tm[0, 5] = 12
        tm[1, 5] = 13

        super().__init__(G, tm)

    @property
    def name(self):
        return 'flow-path-construction'

class BottleneckProblem(Problem):
    def __init__(self):
        G = Problem._read_graph_json(
            os.path.join(TOPOLOGIES_DIR, 'bottleneck.json'))
        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 3] = G[1][3]['capacity']
        super().__init__(G, traffic_matrix)

    @property
    def name(self):
        return 'bottleneck'


class DumbellBottleneckProblem(Problem):
    def __init__(self, *, intra_cap=1e3, inter_cap=15):
        G = Problem._read_graph_json(
            os.path.join(TOPOLOGIES_DIR, 'dumbell-bottleneck.json'))
        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 7] = 35.0
        super().__init__(G, traffic_matrix)
        super()._change_capacities(min_cap=inter_cap,
                                   max_cap=inter_cap,
                                   fixed_caps=[(0, 1, intra_cap),
                                               (2, 3, intra_cap),
                                               (4, 5, intra_cap), (6, 7, 1.0),
                                               (7, 6, 1.0)])

    @property
    def name(self):
        return 'dumbell'


class TwoSrcsFromMetaNodeProblem(Problem):
    def __init__(self):
        G = Problem._read_graph_json(
            os.path.join(TOPOLOGIES_DIR, 'two-srcs.json'))
        num_nodes = len(G.nodes)
        traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        traffic_matrix[0, 3] = 20.0
        traffic_matrix[1, 4] = 10.0
        super().__init__(G, traffic_matrix)
        super()._change_capacities(min_cap=15,
                                   max_cap=15,
                                   fixed_caps=[(2, 3, 10.0), (4, 3, 5.0)])

    @property
    def name(self):
        return 'two-srcs'


PROBLEM_ARGS_FAILURE = {
    'cogentco': {
        'uniform': {
            'max_demand': 0.6
        },
        'fname': 'Cogentco.graphml'
    },
    'uninett': {
        'uniform': {
            'max_demand': 4.6
        },
        'fname': 'Uninett2010.graphml'
    },
    'us-carrier': {
        'uniform': {
            'max_demand': 0.5
        },
        'fname': 'UsCarrier.graphml'
    },
    'b4':{
        'uniform': {
            'max_demand': 150.0
        },
        'fname': 'b4.json'    
    },
    'delta': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom.graphml'
    },
    'triangle': {
        'uniform': {
            'max_demand': 1500.0
        },
        'fname': 'triangle.json'
    },
    'delta20': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_2_0.graphml'
    },
    'delta21': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_2_1.graphml'
    },
    'delta22': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_2_2.graphml'
    },
    'delta23': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_2_3.graphml'
    },
    'delta24': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_2_4.graphml'
    },
    'delta50': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_5_0.graphml'
    },
    'delta51': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_5_1.graphml'
    },
    'delta52': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_5_2.graphml'
    },
    'delta53': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_5_3.graphml'
    },
    'delta54': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_5_4.graphml'
    },
    'delta100': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_10_0.graphml'
    },
    'delta101': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_10_1.graphml'
    },
    'delta102': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_10_2.graphml'
    },
    'delta103': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_10_3.graphml'
    },
    'delta104': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_10_4.graphml'
    },
    'delta1004': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom_100_4.graphml'
    }
    
}


PROBLEM_ARGS = {
    'cogentco': {
        'uniform': {
            'max_demand': 0.6
        },
        'fname': 'Cogentco.graphml'
    },
    'uninett': {
        'uniform': {
            'max_demand': 4.6
        },
        'fname': 'Uninett2010.graphml'
    },
    'us-carrier': {
        'uniform': {
            'max_demand': 0.5
        },
        'fname': 'UsCarrier.graphml'
    },
    'b4':{
        'uniform': {
            'max_demand': 150.0
        },
        'fname': 'b4.json'    
    },
    'delta': {
        'uniform': {
            'max_demand': 2.5
        },
        'fname': 'Deltacom.graphml'
    }
    
}




### Get problem using a single string argument ###
def get_problem(prob_name, model='uniform', seed=0, scale_factor=1.0, **kwargs):

    all_kwargs = PROBLEM_ARGS[prob_name][model]
    for k, v in kwargs.items():
        all_kwargs[k] = v

    if model.startswith('poisson'):
        model = 'poisson'
        fname = PROBLEM_ARGS[prob_name]['fname']
        return TopologyReadyProblem(fname, model=model, scale_factor=scale_factor, seed=seed, **all_kwargs)
    elif prob_name == 'toy':
        return ToyProblem()
    elif prob_name in PROBLEM_ARGS:
        # From Topology Zoo
        fname = PROBLEM_ARGS[prob_name]['fname']
        return TopologyReadyProblem(fname, model=model, scale_factor=scale_factor, seed=seed, **all_kwargs)
        
    else:
        raise Exception('{} is not a valid topology'.format(prob_name))
        
        
        
def get_tri_problem():
    return TriProblem()
    
        
        
def get_srv_tri_problem():
    return TriSrvProblem()
