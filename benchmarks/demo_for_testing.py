import os
import traceback
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from lp_all import lp_all_benchmark
from ncflow import ncflow_benchmark
from mega import mega_top_benchmark, mega_benchmark
from benchmark_consts import get_demo_args_and_problems, print_, PATH_FORM_HYPERPARAMS, NCFLOW_HYPERPARAMS
from lib.config import DEMO_RESULT_DIR


SERVER_SIZE = [1,
               10]

ALGORITHMS_LIST = ['megatop', 'ncflow', 'lpall']


if __name__ == '__main__':
    if not os.path.exists(DEMO_RESULT_DIR):
        os.makedirs(DEMO_RESULT_DIR)      
        
    
    for algo in ALGORITHMS_LIST:
        for size in SERVER_SIZE:
            problems = get_demo_args_and_problems(algo, size)
            print(problems)
            if algo == 'megatop':
                print(problems)
                if size == 1:
                    mega_satisfied_demand_1, mega_runtime_1 = mega_top_benchmark(problems, size, top=True)
                if size == 10:
                    mega_satisfied_demand_10, mega_runtime_10 = mega_top_benchmark(problems, size, top=True)
            
            
            elif algo == 'ncflow':
                print(problems)
                if size == 1:
                    ncflow_satisfied_demand_1, ncflow_runtime_1 = ncflow_benchmark(problems, size)
                    
                if size == 10:
                    ncflow_satisfied_demand_10, ncflow_runtime_10 = ncflow_benchmark(problems, size)
                         
            elif algo == 'lpall':
                print(problems)
                if size == 1:
                    lpall_satisfied_demand_1, lpall_runtime_1 = lp_all_benchmark(problems, size, top=False)
                if size == 10:
                    lpall_satisfied_demand_10, lpall_runtime_10 = lp_all_benchmark(problems, size, top=False)
                    
            
    
    
    '''print('1', mega_runtime_1,
    ncflow_runtime_1,
    lpall_runtime_1,
    '10:',
    mega_satisfied_demand_10,
    ncflow_satisfied_demand_10,
    lpall_satisfied_demand_10)'''
        
    # b4
    x = [12, 120]
    name = 'b4.json'
    # satisfied demand
    plt.figure(1)
    plt.plot(x, [mega_satisfied_demand_1[name] * 100, mega_satisfied_demand_10[name] * 100], label='MegaTE') 
    plt.plot(x, [ncflow_satisfied_demand_1[name] * 100, ncflow_satisfied_demand_10[name] * 100], label='Ncflow')
    plt.plot(x, [lpall_satisfied_demand_1[name] * 100, lpall_satisfied_demand_10[name] * 100], label='Lp-all')  
    plt.xlabel('node number')
    plt.ylabel('satisfied demand(%)')
    plt.legend()
    plt.savefig(os.path.join(DEMO_RESULT_DIR, 'b4_SD.png'))
                    
    # runtime
    plt.figure(2)
    plt.plot(x, [mega_runtime_1[name], mega_runtime_10[name]], label='MegaTE')
    plt.plot(x, [ncflow_runtime_1[name], ncflow_runtime_10[name]], label='Ncflow')
    plt.plot(x, [lpall_runtime_1[name], lpall_runtime_10[name]], label='Lp-all')
    plt.xlabel('node number')
    plt.ylabel('runtime(s)')
    plt.legend()
    plt.savefig(os.path.join(DEMO_RESULT_DIR, 'b4_runtime.png'))
    
    
    
    
    # Uninett2010
    x=[74, 740]
    name = 'Uninett2010.graphml'
    # satisfied demand
    plt.figure(3)
    plt.plot(x, [mega_satisfied_demand_1[name] * 100, mega_satisfied_demand_10[name] * 100], label='MegaTE') 
    plt.plot(x, [ncflow_satisfied_demand_1[name] * 100, ncflow_satisfied_demand_10[name] * 100], label='Ncflow')
    plt.plot(x, [lpall_satisfied_demand_1[name] * 100, lpall_satisfied_demand_10[name] * 100], label='Lp-all')   
    plt.xlabel('node number')
    plt.ylabel('satisfied demand(%)')
    plt.legend()
    plt.savefig(os.path.join(DEMO_RESULT_DIR, 'Uninett2010_SD.png'))
                   
    # runtime
    plt.figure(4)
    plt.plot(x, [mega_runtime_1[name], mega_runtime_10[name]], label='MegaTE')
    plt.plot(x, [ncflow_runtime_1[name], ncflow_runtime_10[name]], label='Ncflow')
    plt.plot(x, [lpall_runtime_1[name], lpall_runtime_10[name]], label='Lp-all')
    plt.xlabel('node number')
    plt.ylabel('runtime(s)')
    plt.legend()
    plt.savefig(os.path.join(DEMO_RESULT_DIR, 'Uninett2010_runtime.png'))
    
    # Deltacom
    x=[113, 1130]
    name = 'Deltacom.graphml'
    # satisfied demand
    plt.figure(5)
    plt.plot(x, [mega_satisfied_demand_1[name] * 100, mega_satisfied_demand_10[name] * 100], label='MegaTE') 
    plt.plot(x, [ncflow_satisfied_demand_1[name] * 100, ncflow_satisfied_demand_10[name] * 100], label='Ncflow')
    plt.plot(x, [lpall_satisfied_demand_1[name] * 100, lpall_satisfied_demand_10[name] * 100], label='Lp-all') 
    plt.xlabel('node number')
    plt.ylabel('satisfied demand')
    plt.legend()
    plt.savefig(os.path.join(DEMO_RESULT_DIR, 'Deltacom_SD.png'))
                     
    # runtime
    plt.figure(6)
    plt.plot(x, [mega_runtime_1[name], mega_runtime_1[name]], label='MegaTE')
    plt.plot(x, [ncflow_runtime_1[name], ncflow_runtime_10[name]], label='Ncflow')
    plt.plot(x, [lpall_runtime_1[name], lpall_runtime_10[name]], label='Lp-all')
    plt.xlabel('node number')
    plt.ylabel('runtime(s)')
    plt.legend()
    plt.savefig(os.path.join(DEMO_RESULT_DIR, 'Deltacom_runtime.png'))
    
    
    
    # Uscarrier
    x=[158, 1580]
    name = 'UsCarrier.graphml'
    # satisfied demand
    plt.figure(7)
    plt.plot(x, [mega_satisfied_demand_1[name] * 100, mega_satisfied_demand_10[name] * 100], label='MegaTE') 
    plt.plot(x, [ncflow_satisfied_demand_1[name] * 100, ncflow_satisfied_demand_10[name] * 100], label='Ncflow')
    plt.plot(x, [lpall_satisfied_demand_1[name] * 100, lpall_satisfied_demand_10[name] * 100], label='Lp-all')   
    plt.xlabel('node number')
    plt.ylabel('satisfied demand(%)')  
    plt.legend()    
    plt.savefig(os.path.join(DEMO_RESULT_DIR, 'UsCarrier_SD.png'))
             
    # runtime
    plt.figure(8)
    plt.plot(x, [mega_runtime_1[name], mega_runtime_10[name]], label='MegaTE')
    plt.plot(x, [ncflow_runtime_1[name], ncflow_runtime_10[name]], label='Ncflow')
    plt.plot(x, [lpall_runtime_1[name], lpall_runtime_10[name]], label='Lp-all')
    plt.xlabel('node number')
    plt.ylabel('runtime(s)')
    plt.legend()
    plt.savefig(os.path.join(DEMO_RESULT_DIR, 'UsCarrier_runtime.png'))
    
    
    # Cogentco
    x=[197, 1970]
    name = 'Cogentco.graphml'
    # satisfied demand
    plt.figure(9)
    plt.plot(x, [mega_satisfied_demand_1[name] * 100, mega_satisfied_demand_10[name] * 100], label='MegaTE') 
    plt.plot(x, [ncflow_satisfied_demand_1[name] * 100, ncflow_satisfied_demand_10[name] * 100], label='Ncflow')
    plt.plot(x, [lpall_satisfied_demand_1[name] * 100, lpall_satisfied_demand_10[name] * 100], label='Lp-all')   
    plt.xlabel('node number')
    plt.ylabel('satisfied demand')
    plt.legend()
    plt.savefig(os.path.join(DEMO_RESULT_DIR, 'Cogentco_SD.png'))
    
    # runtime
    plt.figure(10)
    plt.plot(x, [mega_runtime_1[name], mega_runtime_10[name]], label='MegaTE')
    plt.plot(x, [ncflow_runtime_1[name], ncflow_runtime_10[name]], label='Ncflow')
    plt.plot(x, [lpall_runtime_1[name], lpall_runtime_10[name]], label='Lp-all')
    plt.xlabel('node number')
    plt.ylabel('runtime(s)')
    plt.legend()
    plt.savefig(os.path.join(DEMO_RESULT_DIR, 'Cogentco_runtime.png'))
        
        
        
        
        
        
            
            
            
    

    
        
    
    
    