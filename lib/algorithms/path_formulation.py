from ..lp_solver_gurobi import LpSolver_gurobi
from ..graph_utils import path_to_edge_list
from ..path_utils import find_paths, graph_copy_with_edge_weights, remove_cycles
from ..config import TOPOLOGIES_DIR
from .abstract_formulation import AbstractFormulation, Objective, Solver
from gurobipy import GRB, Model, quicksum
from collections import defaultdict
import numpy as np
import re
import os
import time
import pickle
#from pyomo.environ import *


PATHS_DIR = os.path.join(TOPOLOGIES_DIR, 'paths', 'path-form')
PATHS_DIR_site = os.path.join(TOPOLOGIES_DIR, 'paths', 'site-paths')
PATHS_DIR_site_top = os.path.join(TOPOLOGIES_DIR, 'paths', 'site-paths','Top_version')
PATHS_DIR_srv = os.path.join(TOPOLOGIES_DIR, 'paths')


class PathFormulation(AbstractFormulation):
    @classmethod
    def new_max_flow(cls, num_paths, edge_disjoint=True, dist_metric='inv-cap', out=None):
        print('Objective', Objective)
        return cls(objective=0,
                   # Objective.MAX_FLOW   
                   solver_type = Solver.Gurobi,
                   # Solver.Gurobi
                   num_paths=num_paths,
                   edge_disjoint=edge_disjoint,
                   dist_metric=dist_metric,
                   DEBUG=True,
                   VERBOSE=False,
                   out=out)

    @classmethod
    def new_min_max_link_util(cls, num_paths, edge_disjoint=True, dist_metric='inv-cap', out=None):
        return cls(objective=Objective.MIN_MAX_LINK_UTIL,
                   solver = Solver.Gorubi,
                   num_paths=num_paths,
                   edge_disjoint=edge_disjoint,
                   dist_metric=dist_metric,
                   DEBUG=True,
                   VERBOSE=False,
                   out=out)

    @classmethod
    def compute_demand_scale_factor(cls, num_paths, edge_disjoint=True, dist_metric='inv-cap', out=None):
        return cls(objective=Objective.COMPUTE_DEMAND_SCALE_FACTOR,
                   solver = Solver.Clp,
                   num_paths=num_paths,
                   edge_disjoint=edge_disjoint,
                   dist_metric=dist_metric,
                   DEBUG=True,
                   VERBOSE=False,
                   out=out)

    def __init__(self, *, objective, solver_type, num_paths, edge_disjoint, dist_metric, DEBUG, VERBOSE, out=None):
        super().__init__(objective, solver_type, DEBUG, VERBOSE, out)
        if dist_metric != 'inv-cap' and dist_metric != 'min-hop':
            raise Exception(
                'invalid distance metric: {}; only "inv-cap" and "min-hop" are valid choices'.format(dist_metric))
        self._num_paths = num_paths
        self.edge_disjoint = edge_disjoint
        self.dist_metric = dist_metric

    # flow caps = [((k1, ..., kn), f1), ...]
    def _construct_path_lp(self, G, edge_to_paths,
                           num_total_paths, sat_flows):
                           
        if self._solver_type == Solver.Gurobi:
            m = Model('max-flow: path formulation')
            
            print('path number,', num_total_paths)
            
            # Create variables: one for each path
            path_vars = m.addVars(num_total_paths,
                           vtype=GRB.CONTINUOUS,
                           lb=0.0,
                           name='f')
                           
            # Set objective
            if self._objective == Objective.MIN_MAX_LINK_UTIL or self._objective == Objective.COMPUTE_DEMAND_SCALE_FACTOR:
                 self._print('{} objective'.format(self._objective))
            
                 if self._objective == Objective.MIN_MAX_LINK_UTIL:
                     max_link_util_var = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='z')
                 else:
                     # max link util can be large
                     max_link_util_var = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='z')
            
                 m.setObjective(max_link_util_var, GRB.MINIMIZE)
                 # Add edge util constraints
                 for u, v, c_e in G.edges.data('capacity'):
                     if (u, v) in edge_to_paths:
                         paths = edge_to_paths[(u, v)]
                         constr_vars = [path_vars[p] for p in paths]
                         if c_e == 0.:
                             m.addConstr(quicksum(constr_vars) <= 0.0)
                         else:
                             m.addConstr(quicksum(constr_vars) / c_e <= max_link_util_var)
            
                 # Add demand equality constraints
                 commod_id_to_path_inds = {}
                 self._demand_constrs = []
                 for k, d_k, path_ids in self.commodities:
                     commod_id_to_path_inds[k] = path_ids
                     self._demand_constrs.append(m.addConstr(quicksum(path_vars[p] for p in path_ids) == d_k))
            
            else:
                if self._objective == 0:
                    self._print('MAX FLOW objective')
                    obj = quicksum(path_vars)
                
                elif self._objective == Objective.MAX_CONCURRENT_FLOW:
                    self._print("MAX CONCURRENT FLOW objective")
                    self.alpha = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="a")
                    m.update()
                    for k, d_k, path_ids in self.commodities:
                        m.addConstr(
                            quicksum(path_vars[p] for p in path_ids) / d_k >= self.alpha
                        )
                    obj = self.alpha
                m.setObjective(obj, GRB.MAXIMIZE)
                
                
                # Add edge capacity constraints
                for u, v, c_e in G.edges.data('capacity'):
                  if (u, v) in edge_to_paths:
                      paths = edge_to_paths[(u, v)]
                      constr_vars = [path_vars[p] for p in paths]
                      m.addConstr(quicksum(constr_vars) <= c_e)
                
                # Add demand constraints
                commod_id_to_path_inds = {}
                self._demand_constrs = []
                for k, d_k, path_ids in self.commodities:
                    commod_id_to_path_inds[k] = path_ids
                    self._demand_constrs.append(
                        m.addConstr(quicksum(path_vars[p] for p in path_ids) <= d_k)
                    )
                    
                
            
        elif self._solver_type == Solver.Clp:
            m = ConcreteModel('max-flow: LP')

            # Create variables: one for each path
            print(num_total_paths)
            names_set = ['f_{}'.format(i) for i in range(1, num_total_paths + 1)]
            #id_set = list(range(1, num_total_paths+1))
            m.PATHS = Set(initialize = names_set)
            m.path_vars = Var(m.PATHS, domain=NonNegativeReals, initialize=0.0)
            
            if self._objective == 0:
                self._print('MAX FLOW objective')
                def obj_rule(m):
                    result = sum( m.path_vars[i] for i in m.PATHS)
                    return result
                m.obj = Objective(rule=obj_rule, sense=maximize)
            elif self._objective == Objective.MAX_CONCURRENT_FLOW:
                self._print('MAX CONCURRENT FLOW objective')
                self.alpha = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='a')
                m.update()
                for k, d_k, path_ids in self.commodities:
                    m.addConstr(quicksum(path_vars[p] for p in path_ids) / d_k >= self.alpha)
                obj = self.alpha
            m.setObjective(obj, GRB.MAXIMIZE)
            

            # Add edge capacity constraints
            m.edge_constrs = ConstraintList()
            #print(edge_to_paths)   start from 0
            for u, v, c_e in G.edges.data('capacity'):
                if (u, v) in edge_to_paths:
                    paths = edge_to_paths[(u, v)]
                    def create_edge_const(m):
                        return sum( m.path_vars[m.PATHS[i+1]] for i in paths) <= c_e
                    m.edge_constrs.add( create_edge_const(m) )
                    
            # Add demand constraints
            commod_id_to_path_inds = {}
            self._demand_constrs = []
            m.demand_constrs = ConstraintList()
            for k, d_k, path_ids in self.commodities:
                commod_id_to_path_inds[k] = path_ids
                def create_demand_const(m):
                        return sum( m.path_vars[m.PATHS[i+1]] for i in path_ids) <= d_k
                self._demand_constrs.append((m.demand_constrs.add(create_demand_const(m))))


        if self.DEBUG:
            m.write('lp_debug.lp')
        return LpSolver_gurobi(m, None, self.DEBUG, self.VERBOSE, self.out)

    @staticmethod
    def paths_full_fname(problem, num_paths, edge_disjoint, dist_metric): 
        return os.path.join(PATHS_DIR_site, '{}-{}-paths_edge-disjoint-{}_dist-metric-{}-dict.pkl'.format(problem.name, num_paths, edge_disjoint, dist_metric))

    def compute_paths(self, problem):
        paths_dict = {}
        G = graph_copy_with_edge_weights(problem.G, self.dist_metric)
        for s_k in G.nodes:
            for t_k in G.nodes:
                if s_k == t_k:
                    continue
                paths = find_paths(G, s_k, t_k, self._num_paths, self.edge_disjoint)
                paths_no_cycles = [remove_cycles(path) for path in paths]
                paths_dict[(s_k, t_k)] = paths_no_cycles
        return paths_dict

    def get_paths(self, problem):
        paths_fname = PathFormulation.paths_full_fname(problem, self._num_paths, self.edge_disjoint, self.dist_metric)
        self._print('Loading paths from pickle file', paths_fname)
        try:
            with open(paths_fname, 'rb') as f:
                paths_dict = pickle.load(f)
                for key, paths in paths_dict.items():
                    paths_no_cycles = [remove_cycles(path) for path in paths]
                    paths_dict[key] = paths_no_cycles
                self._print('paths_dict size:', len(paths_dict))
                return paths_dict
        except FileNotFoundError:
            self._print('Unable to find {}'.format(paths_fname))
            paths_dict = self.compute_paths(problem)
            if self.VERBOSE:
                self._print('Saving paths to pickle file')
            with open(paths_fname, 'wb') as w:
                pickle.dump(paths_dict, w)
            return paths_dict

    ###############################
    # Override superclass methods #
    ###############################

    def pre_solve(self, problem=None):
        if problem is None:
            problem = self.problem

        self.commodity_list = problem.sparse_commodity_list if self._warm_start_mode else problem.commodity_list
        self.commodities = []
        edge_to_paths = defaultdict(list)
        self._path_to_commod = {}
        self._all_paths = []

        paths_dict = self.get_paths(problem)
        path_i = 0
        for k, (s_k, t_k, d_k) in self.commodity_list:
            print('dict:', paths_dict)
            paths = paths_dict[(s_k, t_k)]
            
            path_ids = []
            for path in paths:
                self._all_paths.append(path)

                for edge in path_to_edge_list(path):
                    edge_to_paths[edge].append(path_i)
                path_ids.append(path_i)

                self._path_to_commod[path_i] = k
                path_i += 1

            self.commodities.append((k, d_k, path_ids))
        if self.DEBUG:
            assert len(self._all_paths) == path_i

        self._print('pre_solve done')
        return dict(edge_to_paths), path_i

    def _construct_lp(self, sat_flows=[]):
        edge_to_paths, num_paths = self.pre_solve(top)
        #print(edge_to_paths)
        self._print('Constructing Path LP')
        return self._construct_path_lp(self._problem.G, edge_to_paths,
                                       num_paths, sat_flows)

    def extract_sol_as_dict(self):
        sol_dict_def = defaultdict(list)
        if self._solver_type == Solver.Gurobi:
            for var in self.model.getVars():
                if var.varName.startswith('f[') and var.x != 0.0:
                    match = re.match(r'f\[(\d+)\]', var.varName)
                    p = int(match.group(1))
                    sol_dict_def[self.commodity_list[
                            self._path_to_commod[p]]] += [
                            (edge, var.x)
                            for edge in path_to_edge_list(self._all_paths[p])
                    ]
                    
        elif self._solver_type == Solver.Clp:
            for i in self.model.PATHS:
                # print(i)
                if i.startswith('f_') and value(self.model.path_vars[i]) != 0.0:
                    match = re.match(r'f\_(\d+)', i)
                    p = int(match.group(1))
                    sol_dict_def[self.commodity_list[
                        self._path_to_commod[p-1]]] += [
                            (edge, value(self.model.path_vars[i]))
                            for edge in path_to_edge_list(self._all_paths[p-1])
                    ]

        # Set zero-flow commodities to be empty lists
        sol_dict = {}
        sol_dict_def = dict(sol_dict_def)
        for commod_key in self.problem.commodity_list:
            if commod_key in sol_dict_def:
                sol_dict[commod_key] = sol_dict_def[commod_key]
            else:
                sol_dict[commod_key] = []

        return sol_dict
    
    def ssp_output(self):
        sol_dict = defaultdict(list)
        if self._solver_type == Solver.Gurobi:
            for var in self.model.getVars():
                if var.varName.startswith('f[') and var.x != 0.0:
                    match = re.match(r'f\[(\d+)\]', var.varName)
                    p = int(match.group(1))
                    path = self._all_paths[p]
                    # print('path', path, 'value', var.x)
                    key = (path[0],path[-1])
                    sol_dict.setdefault(key, []).append((var.x, path))       
        elif self._solver_type == Solver.Clp:
            for i in self.model.PATHS:
                if i.startswith('f_') and value(self.model.path_vars[i]) != 0.0:
                    match = re.match(r'f\_(\d+)', i)
                    p = int(match.group(1))
                    path = self._all_paths[p-1]
                    key = (path[0],path[-1])
                    sol_dict.setdefault(key, []).append((value(self.model.path_vars[i]), path))        
        sol_dict = dict(sol_dict)
        
        return sol_dict
        

    def extract_sol_as_mat(self):
        edge_idx = self.problem.edge_idx
        sol_mat = np.zeros((len(edge_idx), len(self._path_to_commod)),
                           dtype=np.float32)
        if self._solver_type == Solver.Gurobi:
            for var in self.model.getVars():
                if var.varName.startswith('f[') and var.x != 0.0:
                    match = re.match(r'f\[(\d+)\]', var.varName)
                    p = int(match.group(1))
                    k = self._path_to_commod[p]
                    for edge in path_to_edge_list(self._all_paths[p]):
                        sol_mat[edge_idx[edge], k] += var.x
                
        elif self._solver_type == Solver.Clp:
            for i in self.model.PATHS:
                if i.startswith('f_') and value(self.model.path_vars[i]) != 0.0:
                    match = re.match(r'f\[(\d+)\]', i)
                    p = int(match.group(1))
                    k = self._path_to_commod[p]
                    for edge in path_to_edge_list(self._all_paths[p-1]):
                        sol_mat[edge_idx[edge], k] += value(self.model.path_vars[i])

        return sol_mat

    @classmethod
    # Return total number of fib entries and max for any node in topology
    # NOTE: problem has to have a full TM matrix
    def fib_entries(cls, problem, num_paths, edge_disjoint, dist_metric):
        assert problem.is_traffic_matrix_full
        pf = cls.new_max_flow(num_paths=num_paths,
                              edge_disjoint=edge_disjoint,
                              dist_metric=dist_metric)
        pf.pre_solve(problem,top)
        return pf.num_fib_entries_for_path_set()

    def num_fib_entries_for_path_set(self):
        self.fib_dict = defaultdict(dict)
        for k, _, path_ids in self.commodities:
            commod_id_str = 'k-{}'.format(k)
            src = list(path_to_edge_list(self._all_paths[path_ids[0]]))[0][0]
            # For a given TM, we would store weights for each path id. For demo
            # purposes, we just store the path ids
            self.fib_dict[src][commod_id_str] = path_ids

            for path_id in path_ids:
                for u, v in path_to_edge_list(self._all_paths[path_id]):
                    assert path_id not in self.fib_dict[u]
                    self.fib_dict[u][path_id] = v

        self.fib_dict = dict(self.fib_dict)
        fib_dict_counts = [len(self.fib_dict[k]) for k in self.fib_dict.keys()]
        return sum(fib_dict_counts), max(fib_dict_counts)

    @property
    def runtime(self):
        return self._solver.model.Runtime

    @property
    def obj_val(self):
        return self._solver.model.objVal
        #return self._solver.model.obj
