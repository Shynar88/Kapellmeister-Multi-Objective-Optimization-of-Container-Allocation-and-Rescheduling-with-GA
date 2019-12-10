import argparse
import math
import random
import operator
import sys
import copy
import numpy as np
import logging

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


class Node():
    # {}_specified means initial capacity of the resource
    def __init__(self, cpu_specified, memory_specified, id, max_power, idle_power):
        self.id = id
        self.remaining_cpu = cpu_specified
        self.remaining_memory = memory_specified
        self.cpu_specified = cpu_specified
        self.memory_specified = memory_specified
        self.containers_list = []
        self.max_power = max_power
        self.idle_power = idle_power

    def assign_container(self, container):
        self.remaining_cpu -= container.required_cpu
        self.remaining_memory -= container.required_memory
        self.containers_list.append(container)

    def unassign_container(self, container):
        self.remaining_cpu += container.required_cpu
        self.remaining_memory += container.required_memory
        self.containers_list.remove(container) # might crash because __eq__ for Conatiner class is not defined

class Container():
    def __init__(self, required_cpu, required_memory, task_type):
        self.required_cpu = required_cpu
        self.required_memory = required_memory #size in MB
        self.task_type = task_type

class Chromosome():
    def __init__(self, node_ids, containers, nodes_info, rescheduling, initial_placement):
        self.node_ids = node_ids #node ids
        self.containers = containers
        self.nodes_info = nodes_info #for tracking the resource usage per chromosome 
        self.rescheduling = rescheduling 
        self.initial_placement = initial_placement
        self.fitness = self.get_fitness()

    def get_fitness(self):
        #TODO get fitness, implement 5 objective fitness evalutaion:
        #TODO Equal task distribution
        #TODO Availability
        #TODO Power efficient
        #TODO Resources utilization balancing
        #TODO Unassigned tasks reduction
        if self.rescheduling and self.initial_placement == None: #the corner case when initial placement chromosome is created
            return None
        if self.rescheduling:
            return (self.off_1(), self.off_2(), self.off_3(), self.off_4(), self.off_5(), self.off_6(self.initial_placement))
        else:
            return (self.off_1(), self.off_2(), self.off_3(), self.off_4(), self.off_5())
    # the higher the score, the more infeasable solution is
    # 0 means solution is feasable
    # The number of constraint violations is computed by counting the number of nodes that host containers more than its computational capabilities (CPU or memory).
    def get_infeasability(self):
        infeasability_score = 0
        for node in self.nodes_info:
            if node.remaining_cpu < 0 or node.remaining_memory < 0:
                infeasability_score += 1
        return infeasability_score 

    def off_1(self):
        v = 0
        nodes = self.nodes_info
        for node in nodes:
            t = 0
            i = 0
            for each in node.containers_list:
                i+=1
                t+=i
            v += t
        return v

    def off_2(self):
        v =0
        nodes = self.nodes_info
        for node in nodes:
            dic = {}
            i =1
            for container in node.containers_list:
                if (dic.get(container.task_type) == None):
                    dic[container.task_type] = 1
                else:
                    dic[container.task_type] += 1
            for key in dic:
                n = dic[key]
                v += (n+1)*n/2
        return v

    def off_3(self):
        v = 0
        nodes = self.nodes_info
        for node in nodes:
            c = (node.cpu_specified - node.remaining_cpu)/node.cpu_specified
            m = (node.memory_specified - node.remaining_memory)/node.memory_specified
            p = (node.max_power - node.idle_power)* (c+m)/2 + node.idle_power
        v += p
        return v

    def off_4(self):
        v = 0
        i = 0
        node_ids = self.node_ids
        nodes = self.nodes_info
        for node_id in node_ids:
            if (node_id == None):
                continue
            node = nodes[node_id]
            i += 1
            v += abs(node.remaining_cpu/node.cpu_specified - node.remaining_memory/node.memory_specified)
        if (i == 0):
            return 0
        return 100*v/i
        
    def off_5(self):
        node_ids = self.node_ids
        v = 0
        for node_id in node_ids:
            if (node_id == None):
                v += 1
                continue
        return v
    
    def off_6(self, init_chromosome):
        node_ids = self.node_ids
        init_node_ids = init_chromosome.node_ids
        i = 0
        v = 0
        for node_id in node_ids:
            if (node_id != init_chromosome[i]):
                i +=1
                continue
            v += 1
            i += 1
        return v 

class GeneticAlgorithm():
    def __init__(self, population_size, tournament_size, max_generations, mutation_rate, nodes, containers, rescheduling, initial_placement):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate 
        self.nodes = nodes
        self.containers = containers
        self.rescheduling = rescheduling
        self.initial_placement = initial_placement

    def generate_chromosome(self):
        #size of the chromosome is equal to the number of containers
        #initial solution, is generated by assigning each container to a random node.
        #each chromosome should have deepcopy of resources 
        containers = copy.deepcopy(self.containers) 
        nodes_info = copy.deepcopy(self.nodes)
        node_ids = []
        for i in range(len(self.containers)):
            if random.random() < 0.9: #with 90% probability assign conainer to node, not specified in paper
                node_selected = random.choice(nodes_info) # the node should be passed by reference for keeping track of resources
                node_ids.append(node_selected.id)
                node_selected.assign_container(containers[i])
            else:
                node_ids.append(None)
        chromosome = Chromosome(node_ids, containers, nodes_info, self.rescheduling, self.initial_placement)
        return chromosome

    def create_initial_population(self):
        #TODO initial_population
        initial_population = []
        for _ in range(self.population_size):
            initial_population.append(self.generate_chromosome())
        return initial_population

    def crossover(self, p1, p2):  #p1 and p2 are of type Chromosome
        #TODO crossover
        crossover_point = random.randint(1, len(p1.containers) - 1)
        rand_int = random.randint(0, 1)
        nodes_info = copy.deepcopy(self.nodes) # create nodes_info for future child
        if rand_int == 1:
            child_node_ids = (p1.node_ids[:crossover_point] +  #here could be problem with synchronization, as node should be passed by reference and not by instance
                     p2.node_ids[crossover_point:]) 
        else:
            child_node_ids = (p2.node_ids[crossover_point:] +
                     p1.node_ids[:crossover_point])
        node_ids = copy.deepcopy(child_node_ids)
        containers = copy.deepcopy(p1.containers) #both parents have same containers
        #recalculating resources of nodes
        for (node_id, container) in zip(node_ids, containers):
            if node_id != None:
                nodes_info[node_id].assign_container(container)
        chromosome = Chromosome(node_ids, containers, nodes_info, self.rescheduling, self.initial_placement)
        return chromosome

    def mutate(self, chromosome):  
        #TODO mutation
        mutation_type = random.randint(0, 3)
        if mutation_type == 0:
            return self.swap_mutation(chromosome)
        elif mutation_type == 1:
            return self.change_mutation(chromosome)
        elif mutation_type == 2:
            return self.assign_unassigned_mutation(chromosome)
        elif mutation_type == 3:
            return self.unassign_assigned_mutation(chromosome)

    def swap_mutation(self, chromosome):
        #TODO mutation type 0
        if random.random() < self.mutation_rate:
            none_count = chromosome.node_ids.count(None)
            if none_count > len(chromosome.node_ids) - 2:
                return chromosome
            i1, i2 = random.sample(range(len(chromosome.node_ids)), 2)
            while chromosome.node_ids[i1] == None or chromosome.node_ids[i2] == None:
                i1, i2 = random.sample(range(len(chromosome.node_ids)), 2)
            # recalculating resources
            # TODO: swap only nodes
            chromosome.nodes_info[chromosome.node_ids[i1]].unassign_container(chromosome.containers[i1])
            chromosome.nodes_info[chromosome.node_ids[i1]].assign_container(chromosome.containers[i2])
            chromosome.nodes_info[chromosome.node_ids[i2]].unassign_container(chromosome.containers[i2])
            chromosome.nodes_info[chromosome.node_ids[i2]].assign_container(chromosome.containers[i1])
            chromosome.fitness = chromosome.get_fitness()
            # swapping ids
            chromosome.node_ids[i1], chromosome.node_ids[i2] = chromosome.node_ids[i2], chromosome.node_ids[i1]
            chromosome.fitness = chromosome.get_fitness() # fitness recalculation 
            return chromosome
        return chromosome
    
    def change_mutation(self, chromosome):
        #TODO mutation type 1
        none_count = chromosome.node_ids.count(None)
        if none_count > len(chromosome.node_ids) - 1:
            return chromosome
        change_index = random.randint(0, len(chromosome.node_ids) - 1)
        while chromosome.node_ids[change_index] == None:
            change_index = random.randint(0, len(chromosome.node_ids) - 1)
        while True:
            new_node = random.choice(self.nodes)
            if chromosome.node_ids[change_index] == new_node.id:
                continue
            else: 
                chromosome.nodes_info[chromosome.node_ids[change_index]].unassign_container(chromosome.containers[change_index])
                chromosome.node_ids[change_index] = new_node.id
                chromosome.nodes_info[new_node.id].assign_container(chromosome.containers[change_index])
                chromosome.fitness = chromosome.get_fitness()
                break
        return chromosome

    def assign_unassigned_mutation(self, chromosome):
        #TODO mutation type 2
        #selects one of the unassigned containers and assigns it to a random node regardless of its remaining resources
        indexes = [i for i, x in enumerate(chromosome.node_ids) if x == None]
        if not indexes:
            return chromosome
        assign_index = random.choice(indexes)
        new_node = random.choice(self.nodes)
        chromosome.node_ids[assign_index] = new_node.id
        chromosome.nodes_info[new_node.id].assign_container(chromosome.containers[assign_index])
        chromosome.fitness = chromosome.get_fitness()
        return chromosome

    def unassign_assigned_mutation(self, chromosome):
        #TODO mutation type 3
        while True:
            none_count = chromosome.node_ids.count(None)
            if none_count == len(chromosome.node_ids):
                return chromosome
            change_index = random.randint(0, len(chromosome.node_ids) - 1)
            if chromosome.node_ids[change_index] == None:
                continue
            else: 
                chromosome.nodes_info[chromosome.node_ids[change_index]].unassign_container(chromosome.containers[change_index])
                chromosome.node_ids[change_index] = None
                chromosome.fitness = chromosome.get_fitness()
                break
        return chromosome

    def get_fittest(self, candidates):
        score_n_candidate_list = []
        for chromosome in candidates:
            infeasability_score = chromosome.get_infeasability()
            score_n_candidate_list.append((infeasability_score, chromosome))
        score_n_candidate_list.sort(key=lambda tup: tup[0])
        return score_n_candidate_list[0][1] 

    def selection(self, population): #-> mating pool of size population/2
        #TODO selection
        #Tournament selection   P/2 size might be better
        mating_pool = []
        while len(mating_pool) < len(population)//2: #or self.mat_pool_size
            participants = random.sample(population, self.tournament_size) #self.tournament_size
            fittest = self.get_fittest(participants)
            mating_pool.append(fittest)
        return mating_pool

    def generate_new_population(self, old_population):
        mating_pool = np.array(self.selection(old_population))
        new_population = []
        for i in range(len(old_population)):
            p1, p2 = np.random.choice(mating_pool, 2)
            new_solution = self.crossover(p1, p2)
            new_solution = self.mutate(new_solution)
            new_population.append(new_solution)
        return new_population


    def generate_solution(self): 
        ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=6)

        algorithm = NSGA3(pop_size=self.population_size,
                          sampling=DockSampling(),
                          crossover=DockCrossover(),
                          mutation=DockMutation(),
                          ref_dirs=ref_dirs,
                          eliminate_duplicates=func_is_duplicate)

        res = minimize(DockProblem(self),
                       algorithm,
                       seed=1,
                       verbose=True,
                       termination=('n_gen', self.max_generations))

        return res.X.flatten()

def nsga3_dummy(population_coords, divisions):
    pop_length = population_coords.shape[0]
    selected_indices = np.arange(0, int(pop_length/2))
    best_front_indices = np.arange(0, int(pop_length/4))
    return selected_indices, best_front_indices

# logging data
def write_log(population):
    logging.basicConfig(filename = "fitness.log",
                        filemode = 'w+', 
                        level = logging.INFO,
                        format='%(message)s')
    logger = logging.getLogger()
    logging.FileHandler(logger.name + '-fitness.log', mode='w+')
    #make logging every 100 generations
    fintesses_list = []
    for chromosome in population:
        fintesses_list.append(chromosome.fitness)
    logger.info("{}".format(fintesses_list))

# parses command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, default=50, help="population size") #212
    parser.add_argument('-ts', type=int, default=15, help="tournament size")
    parser.add_argument('-mg', type=int, default=29, help="max generations") #1000
    parser.add_argument('-mr', type=float, default=0.3, help="mutation rate") #0.3
    parser.add_argument('-rsch', type=bool, default=False, help="rescheduling")
    args = parser.parse_args()
    return args.s, args.ts, args.mg, args.mr, args.rsch

def main():
    #################
    # the main will not work without passing in the nodes and containers which are created in Evaluation2.py. Thus, it is recommended to use Evaluation2.py for testing
    #################
    return
    # population_size, tournament_size, max_generations, mutation_rate, rescheduling = parse_arguments()
    # # need to pass nodes and containers here 
    # gen_algo = GeneticAlgorithm(population_size, tournament_size, max_generations, mutation_rate, nodes, containers, rescheduling, None)
    # solution = gen_algo.generate_solution()

if __name__ == "__main__":
    main()
