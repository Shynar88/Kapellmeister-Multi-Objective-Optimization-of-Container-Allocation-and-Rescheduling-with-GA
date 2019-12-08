import numpy as np
import main as dk

from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.factory import get_reference_directions


class DockProblem(Problem):

    def __init__(self, docker_problem):
        super().__init__(n_var=1, n_obj=5, n_constr=0, elementwise_evaluation=True)
        self.docker_problem = docker_problem

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array(x.get_fitness(), dtype=np.float)

class DockSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = problem.docker_problem.create_initial_population()
        return X


class DockCrossover(Crossover):

    def __init__(self):
        super().__init__(2,1)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full((1, n_matings, n_var), None, dtype=np.object)
        for k in range(n_matings):
            a, b = X[0, k, 0], X[1, k, 0]
            Y[0, k, 0] = problem.docker_problem.crossover(a,b)
        return Y


class DockMutation(Mutation):

    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i, 0] = problem.dock_problem.mutate(X[i, 0])
        return X



if __name__ == "__main__":
    pop_size, mat_pool_size, tournament_size, elite_size, max_generations, mutation_rate, nodes_num, containers_num, rescheduling = dk.parse_arguments()
    docker_problem = dk.GeneticAlgorithm(pop_size, mat_pool_size, tournament_size, elite_size, max_generations, mutation_rate, nodes_num, containers_num, rescheduling, None)
    p = DockProblem(docker_problem)
    """
    s = DockSampling()
    X = s._do(p, 100000)
    out = {}
    p._evaluate(X[0], out)
    print(out)

    """
    ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=6)
    algorithm = NSGA3(pop_size=pop_size,
                      ref_dirs=ref_dirs)

    res = minimize(p,
                   algorithm,
                   seed=1,
                   termination=('n_gen', 100))

    print(res.F)
