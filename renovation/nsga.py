from chromosome import *

class DockProblem(Problem):

    def __init__(self,
                 n_obj,
                 population_size,
                 tournament_size,
                 generations,
                 mutation_rate,
                 containers,
                 nodes,
                 **kwargs,
                 ):
        self.n_obj = n_obj
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.containers = containers
        self.nodes = nodes
        super().__init__(n_var=1, n_obj=self.n_obj, n_constr=0, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[0].get_fitness()

class DockSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = generate_initial_population(
                                       problem.population_size,
                                       problem.containers,
                                       problem.nodes,
                                       )
        X = np.reshape(X, (problem.population_size,1))
        return X

class DockCrossover(Crossover):

    def __init__(self):
        super().__init__(2,1)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full((1, n_matings, n_var), None, dtype=np.object)
        for k in range(n_matings):
            a, b = X[0, k, 0], X[1, k, 0]
            Y[0, k, 0] = a.crossover(b)
        return Y

class DockMutation(Mutation):

    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            if np.random.rand() < problem.mutation_rate:
                X[i, 0].mutate()
        return X

def feasibility_tournament(pop, P, algorithm, **kwargs):

    # P is a matrix with chosen indices from pop
    n_tournaments, n_competitors = P.shape

    S = np.zeros(n_tournaments, dtype=np.int)

    for i in range(n_tournaments):

        tournament = P[i]

        scores = np.zeros(n_competitors, dtype=np.int)
        for j in range(n_competitors):
            infeasibility_score = pop[tournament[j]].X[0].get_infeasibility()
            scores[j] = infeasibility_score

        winner_index = scores.argsort()[0]

        S[i] = tournament[winner_index]

    return S


def func_is_duplicate(pop, *other, **kwargs):
    if len(other) == 0:
        return np.full(len(pop), False)

    # value to finally return
    is_duplicate = np.full(len(pop), False)

    return is_duplicate


def solve(kwargs):

    n_obj = 5

    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=kwargs['n_partitions'])

    selection = get_selection('tournament',
                              func_comp=feasibility_tournament,
                              pressure=kwargs['tournament_size'])

    algorithm = NSGA3(pop_size=kwargs['population_size'],
                      sampling=DockSampling(),
                      selection=selection,
                      crossover=DockCrossover(),
                      mutation=DockMutation(),
                      ref_dirs=ref_dirs,
                      eliminate_duplicates=func_is_duplicate)

    res = minimize(DockProblem(**kwargs),
                   algorithm,
                   seed=1,
                   verbose=True,
                   save_history=True,
                   termination=('n_gen', kwargs['generations']))

    return res.X.flatten(), res.history
