# from gurobipy import GurobiError
from enum import Enum, unique
import sys
from pyomo.environ import *
import time

@unique
class Method(Enum):
    PRIMAL_SIMPLEX = 0
    DUAL_SIMPLEX = 1
    BARRIER = 2
    CONCURRENT = 3
    PRIMAL_AND_DUAL = 5


class LpSolver_pyomo(object):
    def __init__(self,
                 model,
                 debug_fn=None,
                 DEBUG=False,
                 VERBOSE=False,
                 out=None,
                 glpk_out=''):
        if out is None:
            out = sys.stdout
        self._model = model
        self._debug_fn = debug_fn
        self.DEBUG = DEBUG
        self.VERBOSE = VERBOSE
        self.out = out
        self._glpk_out = glpk_out

    def _print(self, *args):
        print(*args, file=self.out)

    @property
    def glpk_out(self):
        return self._glpk_out

    @glpk_out.setter
    def glpk_out(self, glpk_out):
        if glpk_out == 'stdout' or glpk_out == '<stdout>':
            self._glpk_out = 'glpk.log'
        else:
            self._glpk_out = glpk_out


    # Note: this is not idempotent: the `model` parameter will be changed after invoking
    # this function
    def solve_lp(self, method=Method.CONCURRENT, bar_tol=None, err_tol=None, numeric_focus=False):
        model = self._model
        if numeric_focus:
            #model.setParam('NumericFocus', 1)
            model.NumericFocus = Param(initialize=1)
        model.Method = Param(initialize=method.value)
        model.LogFile = Param(initialize=self.glpk_out)
        # model.setParam('Method', method.value)
        # model.setParam('LogFile', self.gurobi_out)
        try:
            solver = SolverFactory('cbc')
            solver.options['threads'] = 6
            
            if bar_tol:
                # model.Params.BarConvTol = bar_tol
                solver.options['tol'] = bar_tol
            if err_tol:
                # model.Params.OptimalityTol = err_tol
                # model.Params.FeasibilityTol = err_tol
                solver.options['tol'] = err_tol

            #if self.VERBOSE:
            self._print('\nSolving LP')
            _time = time.time()
            solver.solve(model)
            runtime = time.time() - _time
            model.Runtime = runtime
            #print(value(model.obj))
            # model.optimize()

            if self.DEBUG or self.VERBOSE:
                #for var in model.getVars():
                for i in model.PATHS:
                    #if var.x != 0:
                    if value(model.path_vars[i]) != 0:
                        if self.DEBUG and self._debug_fn:
                            #if not var.varName.startswith('f['):
                            if not i.startswith('f_'):
                                continue
                            u, v, k, s_k, t_k, d_k = self._debug_fn(var)
                            if self.VERBOSE:
                                #self._print(
                                #    'edge ({}, {}), demand ({}, ({}, {}, {})), flow: {}'
                                #    .format(u, v, k, s_k, t_k, d_k, var.x))
                                self._print(
                                    'edge ({}, {}), demand ({}, ({}, {}, {})), flow: {}'
                                    .format(u, v, k, s_k, t_k, d_k, value(model.path_vars[i])))
                        elif self.VERBOSE:
                            #self._print('{} {}'.format(var.varName, var.x))
                            self._print('{} {}'.format(i, value(model.path_vars[i])))
                #self._print('Obj: %g' % model.obj())
                self._print('Obj: %g' % value(model.obj))
            #return model.obj()
            return model.obj
        #except GurobiError as e:
        #    self._print('Error code ' + str(e.errno) + ': ' + str(e))
        except AttributeError as e:
            self._print(str(e))
            self._print('Encountered an attribute error')
        
    @property
    def model(self):
        return self._model

    @property
    def obj_val(self):
        return self._model.objVal