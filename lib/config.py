import os

TL_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
TOPOLOGIES_DIR = os.path.join(TL_DIR, 'topologies')
TM_DIR = os.path.join(TL_DIR, 'traffic-matrices')
DEMO_RESULT_DIR = os.path.join(TL_DIR, 'benchmarks', 'demo_result')


edge_disjoint = True
num_paths = 4

PART = False

TEST_PROBLEM_NAMES  = [
    #'triangle.json',
    'b4.json',
    'Uninett2010.graphml',
    'Deltacom.graphml',
    'UsCarrier.graphml',
    'Cogentco.graphml'  
]
