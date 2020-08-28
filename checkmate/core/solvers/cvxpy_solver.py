import logging
import os

import cvxpy as cp
import numpy as np

from checkmate.core.dfgraph import DFGraph
from checkmate.core.enum_strategy import SolveStrategy
from checkmate.core.schedule import ScheduledResult, ILPAuxData
from checkmate.core.utils.scheduler import schedule_from_rs
from checkmate.core.utils.solver_common import solve_r_opt
from checkmate.core.utils.timer import Timer


class ILPSolverCVXPY:
    def __init__(self, g: DFGraph, budget: int, integral=False):
        self.budget = budget
        self.g = g
        self.T = int(np.ceil(np.log2(self.g.size // 2)))

        self.R = cp.Variable((self.T, self.g.size), name="R", boolean=integral)
        self.S = cp.Variable((self.T, self.g.size), name="S", boolean=integral)
        self.Free_E = cp.Variable((self.T, len(self.g.edge_list)), name="FREE_E", boolean=integral)
        self.U = cp.Variable((self.T, self.g.size), name="U")

        cpu_cost_vec = np.asarray([self.g.cost_cpu[i] for i in range(self.g.size)])[np.newaxis, :].T
        assert cpu_cost_vec.shape == (self.g.size, 1)
        objective = cp.Minimize(cp.sum(self.R @ cpu_cost_vec))
        constraints = self.make_constraints(budget)
        self.problem = cp.Problem(objective, constraints)
        self.num_vars = self.problem.size_metrics.num_scalar_variables
        self.num_constraints = self.problem.size_metrics.num_scalar_eq_constr + self.problem.size_metrics.num_scalar_leq_constr

    def make_constraints(self, budget):
        constraints = []
        ram_costs = self.g.cost_ram
        ram_cost_vec = np.asarray([ram_costs[i] for i in range(self.g.size)])

        with Timer("Var bounds"):
            constraints.extend([self.R >= 0, self.R <= 1])
            constraints.extend([self.S >= 0, self.S <= 1])
            constraints.extend([self.Free_E >= 0, self.Free_E <= 1])
            constraints.extend([self.U >= 0, self.U <= budget])

            constraints.append(self.R[0, self.g.vfwd] == 1)
            constraints.append(self.S[0] == 0)
            constraints.append(cp.sum(self.R[:, self.g.vfwd], axis=0) <= 2)
            constraints.append(cp.sum(self.R[:, self.g.vbwd], axis=0) == 1)

        with Timer("Correctness constraints"):
            # ensure all checkpoints are in memory
            constraints.append(self.S[1:, :] <= self.S[:-1, :] + self.R[:-1, :])

            # ensure all computations are possible
            for (u, v) in self.g.edge_list:
                constraints.append(self.R[:, v] <= self.R[:, u] + self.S[:, u])

        with Timer("Free_E constraints"):
            # Constraint: sum_k Free_{t,i,k} <= 1
            for i in range(self.g.size):
                frees = [self.Free_E[:, eidx] for eidx, (j, _) in enumerate(self.g.edge_list) if i == j]
                if frees:
                    constraints.append(cp.sum(frees, axis=0) <= 1)

            # Constraint: Free_{t,i,k} <= 1 - S_{t+1, i}
            for eidx, (i, k) in enumerate(self.g.edge_list):
                constraints.append(self.Free_E[:-1, eidx] + self.S[1:, i] <= 1)

            # Constraint: Free_{t,i,k} <= 1 - R_{t, j}
            for eidx, (i, k) in enumerate(self.g.edge_list):
                for j in self.g.successors(i):
                    if j > k:
                        constraints.append(self.Free_E[:, eidx] + self.R[:, j] <= 1)

        with Timer("U constraints"):
            constraints.append(self.U[:, 0] == self.R[:, 0] * ram_costs[0] + ram_cost_vec @ self.S.T)
            for k in range(self.g.size - 1):
                mem_freed = cp.sum([ram_costs[i] * self.Free_E[:, eidx] for (eidx, i) in self.g.predecessors_indexed(k)])
                constraints.append(self.U[:, k + 1] == self.U[:, k] + self.R[:, k + 1] * ram_costs[k + 1] - mem_freed)
        return constraints

    def solve(self, solver_override=None, verbose=False, num_threads=os.cpu_count()):
        installed_solvers = cp.installed_solvers()
        with Timer("Solve", print_results=verbose) as solve_timer:
            if solver_override is not None:
                self.problem.solve(verbose=verbose, solver=solver_override)
            elif "MOSEK" in installed_solvers:
                self.problem.solve(verbose=verbose, solver=cp.MOSEK)
            elif "GUROBI" in installed_solvers:
                self.problem.solve(verbose=verbose, solver=cp.GUROBI)
            elif "CBC" in installed_solvers:
                self.problem.solve(verbose=verbose, solver=cp.CBC, numberThreads=num_threads)
            else:
                self.problem.solve(verbose=verbose)
        self.solve_time = solve_timer.elapsed
        if self.problem.status in ["infeasible", "unbounded"]:
            raise ValueError("Model infeasible")
        return self.R.value, self.S.value, self.U.value, self.Free_E.value


def solve_checkmate_cvxpy(g, budget, rounding_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), solver_override=None, verbose=False, integral=True):
    lpsolver = ILPSolverCVXPY(g, budget, integral=integral)  # rounding threshold
    r, s, u, free_e = lpsolver.solve(solver_override=solver_override, verbose=verbose)
    return r, s, u, free_e

    # try:
    #     r, s, u, free_e = lpsolver.solve(solver_override=solver_override)
    #     lp_feasible = True
    # except ValueError as e:
    #     logging.exception(e)
    #     r, s, u, free_e = (None, None, None, None)
    #     lp_feasible = False
    # schedule, aux_data, min_threshold = None, None, None
    # if lp_feasible:  # round the solution
    #     for threshold in rounding_thresholds:
    #         s_ = (s >= threshold).astype(np.int)
    #         r_ = solve_r_opt(g, s_)
    #         schedule_, aux_data_ = schedule_from_rs(g, r_, s_)
    #         if aux_data_.activation_ram <= budget and (aux_data is None or aux_data_.cpu <= aux_data.cpu):
    #             aux_data = aux_data_
    #             schedule = schedule_
    #             min_threshold = threshold
    # solve_strategy = (
    #     SolveStrategy.APPROX_DET_ROUND_LP_05_THRESH
    #     if len(rounding_thresholds) == 1
    #     else SolveStrategy.APPROX_DET_ROUND_LP_SWEEP
    # )
    # return ScheduledResult(
    #     solve_strategy=solve_strategy,
    #     solver_budget=budget,
    #     feasible=lp_feasible and aux_data is not None,
    #     schedule=schedule,
    #     schedule_aux_data=aux_data,
    #     solve_time_s=lpsolver.solve_time,
    #     ilp_aux_data=ILPAuxData(
    #         U=u,
    #         Free_E=free_e,
    #         ilp_approx=False,
    #         ilp_time_limit=None,
    #         ilp_eps_noise=0.0,
    #         ilp_num_constraints=lpsolver.num_vars,
    #         ilp_num_variables=lpsolver.num_constraints,
    #         approx_deterministic_round_threshold=min_threshold,
    #     ),
    # )


# Extra, unnecessary constraints
# for eidx, (i, k) in enumerate(self.g.edge_list):
#     num_hazards = 1 - self.R[:-1, k] + self.S[1:, i]
#       + cp.sum([self.R[:-1, j] for j in self.g.successors(i) if j > k], axis=0)
#     num_uses_after_k = sum(1 for j in self.g.successors(i) if j > k)
#     max_num_hazards = 2 + num_uses_after_k
#     constraints.append(max_num_hazards * (1 - self.Free_E[:-1, eidx]) >= num_hazards)
#     constraints.append(1 - self.Free_E[:-1, eidx] <= num_hazards)
#     num_hazards = 1 - self.R[-1, k] + cp.sum([self.R[-1, j] for j in self.g.successors(i) if j > k])
#     constraints.append(max_num_hazards * (1 - self.Free_E[-1, eidx]) >= num_hazards)
#     constraints.append(1 - self.Free_E[-1, eidx] <= num_hazards)
