import numpy as np
import copy
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.factory import get_reference_directions
from pymoo.factory import get_selection

class Node():

    """
    This node is not supposed to have a list of assigned containers,
    it is containers that have to be assigned with nodes
    """

    def __init__(self, max_cpu, max_memory):
        self.max_cpu = max_cpu
        self.max_memory = max_memory
        self.free_cpu = max_cpu
        self.free_memory = max_memory
        self.max_power = self._calc_max_power()
        self.idle_power = self._calc_idle_power()
        self.feasible = True
        self.containers = []

    def __repr__(self):
        return f"Node(free_cpu={self.free_cpu},free_mem={self.free_memory},p_max={self.max_power},p_idle={self.idle_power},feas={self.feasible})"

    def _assign_container(self, container):
        self.free_cpu -= container.required_cpu
        self.free_memory -= container.required_memory
        if self.free_cpu < 0 or self.free_memory < 0:
            self.feasible = False
        self.containers.append(container)

    def _unassign_container(self, container):
        self.free_cpu += container.required_cpu
        self.free_memory += container.required_memory
        assert self.free_cpu <= self.max_cpu, "error in node: free cpu is larger than max"
        assert self.free_memory <= self.max_memory, "error in node: free memory is larger than max"
        if self.free_cpu > 0 and self.free_memory > 0:
            self.feasible = True
        self.containers.remove(container)

    def _calc_idle_power(self):

        p_idle_total_ref = 171
        p_cpu_part_ref = 0.3
        p_mem_part_ref = 0.11
        p_cpu_ref = 4
        p_mem_ref = 4000

        p_cpu_idle_ref = p_idle_total_ref * p_cpu_part_ref
        del_p_cpu_idle = p_cpu_idle_ref * (self.max_cpu / p_cpu_ref - 1)

        p_mem_idle_ref = p_idle_total_ref * p_mem_part_ref
        del_p_mem_idle = p_mem_idle_ref * (self.max_memory / p_mem_ref - 1)

        return p_idle_total_ref + del_p_mem_idle + del_p_cpu_idle

    def _calc_max_power(self):

        p_max_total_ref = 218
        p_cpu_part_ref = 0.3
        p_mem_part_ref = 0.11
        p_cpu_ref = 4
        p_mem_ref = 4000

        p_cpu_max_ref = p_max_total_ref * p_cpu_part_ref
        del_p_cpu_max = p_cpu_max_ref * (self.max_cpu / p_cpu_ref - 1)

        p_mem_max_ref = p_max_total_ref * p_mem_part_ref
        del_p_mem_max = p_mem_max_ref * (self.max_memory / p_mem_ref - 1)

        return p_max_total_ref + del_p_mem_max + del_p_cpu_max

class Container():

    def __init__(self, required_cpu, required_memory, task_type):
        self.required_cpu    = required_cpu
        self.required_memory = required_memory
        self.task_type       = task_type
        self.node            = None

    def __repr__(self):
        return f"Container(cpu={self.required_cpu},mem={self.required_memory},type={self.task_type})"

    def assign_node(self, node):
        if node is not None:
            node._assign_container(self)
            self.node = node

    def unassign_node(self):
        if self.node is not None:
            self.node._unassign_container(self)
            self.node = None


class Chromosome():

    def __init__(self, containers, nodes):
        self.containers = containers
        self.nodes      = nodes
        self._allocate_nodes()

    def __str__(self):
        s = ""
        for c in self.containers:
            s += str(c) + " | " + str(c.node) + "\n"

        return s

    def _allocate_nodes(self):
        for c in self.containers:
            c.assign_node(np.random.choice(self.nodes))

    def _obj_1(self):
        s = 0
        for n in self.nodes:
            t = 0
            for i in range(1, len(n.containers)+1):
                t += i
            s += t
        return s

    def _obj_2(self):
        s = 0
        for n in self.nodes:
            t = 0
            types = []
            types_count = []
            for c in n.containers:
                if c.task_type not in types:
                    types.append(c.task_type)
                    types_count.append(1)
                    t += 1
                else:
                    index = types.index(c.task_type)
                    types_count[index] += 1
                    t += types_count[index]
            s += t
        return s

    def _obj_3(self):
        s = 0
        for n in self.nodes:
            cpu_share = (n.max_cpu - n.free_cpu) / n.max_cpu
            memory_share = (n.max_memory - n.free_memory) / n.max_memory
            s += n.idle_power + (cpu_share + memory_share)/2 * (n.max_power - n.idle_power)
        return s

    def _obj_4(self):
        s = 0
        for n in self.nodes:
            rem_cpu_share = n.free_cpu / n.max_cpu
            rem_memory_share = n.free_memory / n.max_memory
            s += abs(rem_memory_share - rem_cpu_share)
        return s*100

    def _obj_5(self):
        s = 0
        for c in self.containers:
            if c.node is None:
                s += 1
        return s

    def get_fitness(self):
        return np.array((self._obj_1(), self._obj_2(), self._obj_3(), self._obj_4(), self._obj_5()),
                         dtype=np.float)

    def get_infeasibility(self):
        s = 0
        for n in self.nodes:
            if not n.feasible:
                s += 1
        return s

    def crossover(self, other):
        cross_point = np.random.randint(1, len(self.containers) + 1)
        child = copy.deepcopy(self)
        for i in range(cross_point, len(self.containers)):
            child.containers[i].unassign_node()
            child.containers[i].assign_node(other.containers[i].node)
        return child

    def mutate(self):
        r = np.random.randint(0,4)
        if r == 0:
            self._swap_mutation()
        elif r == 1:
            self._change_mutation()
        elif r == 2:
            self._unassign_mutation()
        else:
            self._assign_mutation()

    def _swap_mutation(self):
        viable_containers = []
        for c in self.containers:
            if c.node is not None:
                viable_containers.append(c)
        if len(viable_containers) < 2:
            return

        s1, s2 = np.random.choice(viable_containers, 2, replace=False)
        node = s1.node
        s1.unassign_node()
        s1.assign_node(s2.node)
        s2.unassign_node()
        s2.assign_node(node)

    def _change_mutation(self):
        viable_containers = []
        for c in self.containers:
            if c.node is not None:
                viable_containers.append(c)
        if len(viable_containers) < 1:
            return

        c = np.random.choice(viable_containers)
        c.unassign_node()
        node = np.random.choice(self.nodes)
        c.assign_node(node)

    def _assign_mutation(self):
        viable_containers = []
        for c in self.containers:
            if c.node is None:
                viable_containers.append(c)
        if len(viable_containers) < 1:
            return

        c = np.random.choice(viable_containers)
        node = np.random.choice(self.nodes)
        c.assign_node(node)

    def _unassign_mutation(self):
        viable_containers = []
        for c in self.containers:
            if c.node is not None:
                viable_containers.append(c)
        if len(viable_containers) < 1:
            return

        c = np.random.choice(viable_containers)
        c.unassign_node()


def generate_initial_population(
                                population_size,
                                n_containers,
                                n_services,
                                containers_cpu_bounds,
                                containers_memory_bounds,
                                nodes_cpu_types,
                                nodes_memory_types,
                                nodes_per_type,
                                ):
    assert len(containers_cpu_bounds) == len(containers_memory_bounds)
    assert len(nodes_cpu_types) == len(nodes_memory_types)

    """Generate_containers"""

    spread = 2
    avg_cont_per_serv = int(n_containers/n_services)
    cont_per_service = np.random.randint(avg_cont_per_serv-spread,
                                         avg_cont_per_serv+spread+1,
                                         size=n_services)

    containers = []
    min_cpu, max_cpu = containers_cpu_bounds
    min_memory, max_memory = containers_memory_bounds

    for i in range(len(cont_per_service)):
        for j in range(cont_per_service[i]):
            cpu = np.random.randint(min_cpu, max_cpu+1)
            memory = np.random.randint(min_memory, max_memory+1)
            containers.append(Container(cpu, memory, i))

    """ Generate nodes """

    nodes = []
    for cpu, mem in zip(nodes_cpu_types, nodes_memory_types):
        for i in range(nodes_per_type):
            nodes.append(Node(cpu, mem))

    pop = np.full(population_size, None)
    for i in range(population_size):
        pop[i] = Chromosome(np.array(copy.deepcopy(containers)), np.array(copy.deepcopy(nodes)))

    return pop


class Kapellmeister():

    def __init__(self,
                 pop_size,
                 tournamet_size,
                 generations,
                 mutation_rate,
                 nodes,
                 containers):
        self.pop_size = pop_size
        self.tournamet_size = tournamet_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.nodes = nodes
        self.containers = containers

    def create_initial_population(self):
        pass

class DockProblem(Problem):

    def __init__(self,
                 n_obj,
                 population_size,
                 tournament_size,
                 generations,
                 mutation_rate,
                 n_containers,
                 n_services,
                 containers_cpu_bounds,
                 containers_memory_bounds,
                 nodes_cpu_types,
                 nodes_memory_types,
                 nodes_per_type,
                 **kwargs,
                 ):
        self.n_obj = n_obj
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.n_containers = n_containers
        self.n_services = n_services
        self.containers_cpu_bounds = containers_cpu_bounds
        self.containers_memory_bounds = containers_memory_bounds
        self.nodes_cpu_types = nodes_cpu_types
        self.nodes_memory_types = nodes_memory_types
        self.nodes_per_type = nodes_per_type
        super().__init__(n_var=1, n_obj=self.n_obj, n_constr=0, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[0].get_fitness()

class DockSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = generate_initial_population(
                                        problem.population_size,
                                        problem.n_containers,
                                        problem.n_services,
                                        problem.containers_cpu_bounds,
                                        problem.containers_memory_bounds,
                                        problem.nodes_cpu_types,
                                        problem.nodes_memory_types,
                                        problem.nodes_per_type,
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

    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=6)

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

kwargs = {  'n_obj':5,
       'population_size':200,
       'tournament_size':7,
       'generations':10,
       'mutation_rate':0.3,
       'n_containers':300,
       'n_services':100,
       'containers_cpu_bounds':(0,1),
       'containers_memory_bounds':(25,1000),
       'nodes_cpu_types':[1,2,4,8,16],
       'nodes_memory_types':[2048,4096,8192,16384,32768],
       'nodes_per_type':20,
 }

r, h = solve(kwargs)

