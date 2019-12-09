import numpy as np
from main import parse_arguments, GeneticAlgorithm

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
        out["F"] = np.array(x[0].get_fitness(), dtype=np.float)

class DockSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = problem.docker_problem.create_initial_population()
        X = np.reshape(X, (problem.docker_problem.population_size,1))
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
            X[i, 0] = problem.docker_problem.mutate(X[i, 0])
        return X


def func_is_duplicate(pop, *other, **kwargs):
    if len(other) == 0:
        return np.full(len(pop), False)

    # value to finally return
    is_duplicate = np.full(len(pop), False)

    return is_duplicate



def nsga3():
    pop_size, mat_pool_size, tournament_size, elite_size, max_generations, mutation_rate, nodes_num, containers_num, rescheduling = parse_arguments()
    docker_problem = GeneticAlgorithm(pop_size, mat_pool_size, tournament_size, elite_size, max_generations, mutation_rate, nodes_num, containers_num, rescheduling, None)

    ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=6)
    algorithm = NSGA3(pop_size=pop_size,
                      sampling=DockSampling(),
                      crossover=DockCrossover(),
                      mutation=DockMutation(),
                      ref_dirs=ref_dirs,
                      eliminate_duplicates=func_is_duplicate)

    res = minimize(DockProblem(docker_problem),
                   algorithm,
                   seed=1,
                   verbose=True,
                   termination=('n_gen', 1000))

    return res.F.flatten()
