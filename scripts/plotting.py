# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:35:10 2021

@author: think
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['figure.figsize'] = 40, 22
plt.rcParams['font.sans-serif'] = ['Simsun']
ax =plt.subplot()





x=[10,50,100,200,500,1000,2000]

#y1=[30/800*i*(1+i)/2 for i in x] #openstack
#y2 = [1/4000*(1+i)*i/2+0.5*i for i in x]
#y3 = [0.05*i for i in x]

#x1 = ['B4 (120)', 'B4 (600)', 'Uninett2010 (740)', 'Deltacom (1130)', 'B4 (1200)', 'UsCarrier (1580)', 'Cogentco (1970)', 'Uninett2010 (3700)', 'Deltacom (5650)', 'Uninett2010 (7400)', 'UsCarrier (7900)', 'Cogentco (9850)', 'Deltacom (11300)']

#y1 = [0.135, 0.186, 2.748, 6.888, ]

#x1 = ['B4 (120)', 'Uninett2010 (740)', 'Deltacom (1130)', 'UsCarrier (1580)', 'Cogentco (1970)', 'B4 (600)', 'Uninett2010 (3700)', 'Deltacom (5650)', 'UsCarrier (7900)', 'Cogentco (9850)', 'B4 (1200)', 'Uninett2010 (7400)', 'Deltacom (11300)']

x1 = ['B4(10 srv/site)',  'B4(50 srv/site)', 'B4(100 srv/site)', 'B4(200 srv/site)']  #LPtop
x2 = ['B4(10 srv/site)',  'B4(50 srv/site)', 'B4(100 srv/site)', 'B4(200 srv/site)']  #LPall
x3 = ['B4(10 srv/site)',  'B4(50 srv/site)', 'B4(100 srv/site)', 'B4(200 srv/site)']  #ncflow

x4 = ['Uninett2010(10 srv/site)',  'Uninett2010(50 srv/site)', 'Uninett2010(100 srv/site)', 'Uninett2010(200 srv/site)']  #LPtop
x5 = ['Uninett2010(10 srv/site)',  'Uninett2010(50 srv/site)', 'Uninett2010(100 srv/site)', 'Uninett2010(200 srv/site)']  #LPall
x6 = ['Uninett2010(10 srv/site)',  'Uninett2010(50 srv/site)', 'Uninett2010(100 srv/site)', 'Uninett2010(200 srv/site)']  #ncflow

x7 = ['Deltacom(10 srv/site)',  'Deltacom(50 srv/site)', 'Deltacom(100 srv/site)', 'Deltacom(200 srv/site)']  #LPtop
x8 = ['Deltacom(10 srv/site)',  'Deltacom(50 srv/site)', 'Deltacom(100 srv/site)', 'Deltacom(200 srv/site)']  #LPall
x9 = ['Deltacom(10 srv/site)',  'Deltacom(50 srv/site)', 'Deltacom(100 srv/site)', 'Deltacom(200 srv/site)']  #ncflow

x10 = ['UsCarrier(10 srv/site)',  'UsCarrier(50 srv/site)', 'UsCarrier(100 srv/site)', 'UsCarrier(200 srv/site)']  #LPtop
x11 = ['UsCarrier(10 srv/site)',  'UsCarrier(50 srv/site)', 'UsCarrier(100 srv/site)', 'UsCarrier(200 srv/site)']  #LPall
x12 = ['UsCarrier(10 srv/site)',  'UsCarrier(50 srv/site)', 'UsCarrier(100 srv/site)', 'UsCarrier(200 srv/site)']  #ncflow


x13 = ['Cogentco(10 srv/site)',  'Cogentco(50 srv/site)', 'Cogentcor(100 srv/site)', 'Cogentco(200 srv/site)']  #LPtop
x14 = ['Cogentco(10 srv/site)',  'Cogentco(50 srv/site)', 'Cogentcor(100 srv/site)', 'Cogentco(200 srv/site)']  #LPall
x15 = ['Cogentco(10 srv/site)',  'Cogentco(50 srv/site)', 'Cogentcor(100 srv/site)', 'Cogentco(200 srv/site)']  #ncflow




#x2 = ['B4 (120)', 'Uninett2010 (740)', 'Deltacom (1130)', 'UsCarrier (1580)', 'Cogentco (1970)', 'B4 (600)', 'Uninett2010 (3700)', 'Deltacom (5650)', 'B4 (1200)']
#x3 = ['B4 (120)', 'Uninett2010 (740)', 'Deltacom (1130)']

#y1 = [0.135, 2.748, 6.888, 11.33, 22.435, 0.156, 2.988, 8.686, 14.132, 26.316, 0.186, 4.074, 11.959]
#y2 = [0.21, 2.899, 4.46, 8.213, 10.194, 6.588, 31.446, 75.225, 11.431]
#y3 = [0.018,0.8,2.6]

y1 = [0,0,0,0]
y2 = [1.947,,0,0]
y3 = [0.210 ,6.558,11.431,57.081]


y4 = [0,0,0,0]
y5 = [0,0,0,0]
y6 = [2.899 ,31.446,700.948,4740.34]


y7 = [0,0,0,0]
y8 = [0,0,0,0]
y9 = [4.46 ,75.225,1642.206,11105.839]

 

y10 = [0,0,0,0]
y11 = [0,0,0,0]
y12 = [8.213,466.723 ,3218.755,21767.655]


y13 = [0,0,0,0]
y14 = [0,0,0,0]
y15 = [10.194,726.482,5010.181,33882.637]

#y1 = [87.7, 98.8, 99.0, 99.0, 99.0, 87.7, 98.0, 99.0, 99.0, 99.0, 87.7, 98.1, 99.0]
#y2 = [88.2, 90.8, 92.4, 95.5, 96.2, 88.1, 91.0, 92.8, 87.9]
#y3 = [83.8,94.5,94.05]

'''x1 = ['B4 (600)', 'Uninett2010 (3700)', 'Deltacom (5650)', 'UsCarrier (7900)', 'Cogentco (9850)']
x2 = ['B4 (600)', 'Uninett2010 (3700)', 'Deltacom (5650)']
x3 = ['B4 (1200)', 'Uninett2010 (7400)', 'Deltacom (11300)']
y1 = [0.156, 2.988, 8.686, 14.132, 26.316]
y2 = [6.588, 31.446, 75.225]
y1 = [87.7, 98.0, 99.0, 99.0, 99.0]
y2 = [88.1, 91.0, 92.8]
y3 = [90, 90, 90]'''

if __name__ == "__main__":
    
    # a, = plt.plot(x1, y1, linestyle='-', marker='o', markersize=12, linewidth=4, label="Tradition", color='black')
    # b, = plt.plot(x1, y2, linestyle='-', marker='s', markersize=12, linewidth=4, label="MirrorNet", color='red')

    # plt.bar(x,y1, width=width,color='firebrick',hatch="/",label='Snapshot load time')
    # plt.bar(x,y2, bottom=y1, width=width,color='red',hatch="\\",label='Event overlying time')

    # plt.bar(x,y1, width=width,color='goldenrod',hatch="/",label='Load time')
    # plt.bar(x+width,y2, width=width,color='darkorange',hatch="\\",label='Overlying time')
    a, = plt.plot(x1, y1, linestyle='-', marker='s', markersize=12, linewidth=4, label="LP+SSP", color='black')
    
    b, = plt.plot(x2, y2, linestyle='-', marker='v', markersize=10, linewidth=4, label="NCFlow", color='green')
    a, = plt.plot(x3, y3, linestyle='-', marker='o', markersize=12, linewidth=4, label="teal", color='firebrick')
    
    plt.legend(loc='upper left',ncol=1, shadow=True, fontsize=25)
    plt.grid()  # == plt.grid(True)
    plt.grid(color='black', linewidth='0.1', linestyle='--')
    ax.tick_params(direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=25)
    plt.xticks(rotation=45)

    #plt.xscale('log')
    #plt.yscale('log')
    # plt.xticks([1,7,13,19], ['Vendor1\nInd=1', 'Vendor1\nInd=10', 'Vendor2\nInd=1', 'Vendor2\nInd=10'], fontsize=44)
    # plt.xticks([0,2,4,6,8,10,12], ['0','2', '4', '6', '8', '10', '12'], fontsize=25)
    # plt.xticks([0,2,4,6,8,10,12], ['0','Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec'], fontsize=25)
    # plt.xticks(x+width/2, ['1s', '1min', '1h', '1day', '1month', '1year'], fontsize=20) 
    # plt.yticks([0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000 ], ['0', '2M', '4M', '6M', '8M', '10M', '12M' ,'14M', '16M'], fontsize=25)
    #plt.xticks([10, 100, 1000, 2000], ['10', '100', '1000', '2000'], fontsize=25)
    # plt.yticks([0.1, 1, 10, 100, 1000], ['0.1', '1', '10', '$10^2$', '$10^3$'], fontsize=25)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['right'].set_linewidth(2.5)
    ax.spines['top'].set_linewidth(2.5)
    ax.xaxis.grid(True, which='major')
    # plt.legend(loc='lower right', fontsize=25)
    #plt.axis([10, 1000, 0, 200])
    plt.xlabel('networks', fontsize=25)
    plt.ylabel('Time (s)', fontsize=25)
    #plt.ylabel('Satisfied demand (%)', fontsize=25)
    #plt.savefig("D:\\Projects\\networkopti\\strayfiles\\visualize\\images\\time_10srv.png")
    #plt.savefig("D:\\Projects\\networkopti\\strayfiles\\visualize\\images\\obj_50srv.png")
    plt.show()