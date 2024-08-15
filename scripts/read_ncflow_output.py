import pickle


path = "/home/ubuntu/ncflow/benchmarks/ncflow-logs/triangle.json/480528682-uniform/ncflow/fm_partitioning/3-partitions/4-paths/edge_disjoint-True/dist_metric-inv-cap/triangle.json-ncflow-partitioner_fm_partitioning-3_partitions-4_paths-edge_disjoint_True-dist_metric_inv-cap_iter_0-r3-sol-dict.pkl"
with open(path,'rb') as f:
    sol = pickle.load(f)
    
    
print(sol)

ssp_path = "/home/ubuntu/ncflow/benchmarks/ncflow-logs/triangle.json/480528682-uniform/ncflow/fm_partitioning/3-partitions/4-paths/edge_disjoint-True/dist_metric-inv-cap/triangle.json-ncflow-partitioner_fm_partitioning-3_partitions-4_paths-edge_disjoint_True-dist_metric_inv-cap_iter_1-r3-sol-dict.pkl"
with open(ssp_path,'rb') as f:
    sol_ssp = pickle.load(f)
    
print(sol_ssp)

path_path = "/home/ubuntu/ncflow/benchmarks/ncflow-logs/triangle.json/480528682-uniform/ncflow/fm_partitioning/3-partitions/4-paths/edge_disjoint-True/dist_metric-inv-cap/triangle.json-ncflow-partitioner_fm_partitioning-3_partitions-4_paths-edge_disjoint_True-dist_metric_inv-cap_iter_2-r3-sol-dict.pkl"
with open(path_path,'rb') as f:
    path_test = pickle.load(f)

print(path_test)


#print(sol,sol_ssp)