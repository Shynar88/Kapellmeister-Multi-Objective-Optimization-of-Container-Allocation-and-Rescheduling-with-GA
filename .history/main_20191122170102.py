import argparse
import math
import random
import operator
import sys
import copy

class Node():
    # {}_specified means initial capacity of the resource
    def __init__(self, remaining_cpu, remaining_memory, cpu_specified, memory_specified):
        self.remaining_cpu = remaining_cpu
        self.remaining_memory = remaining_memory
        self.cpu_specified = cpu_specified
        self.memory_specified = memory_specified
        self.containers_list = []

class Container():
    def __init__(self, required_cpu, required_memory):
        self.required_cpu = required_cpu
        self.required_memory = required_memory #size in MB

# class Gene():
#     def __init__(self, node, container):
#         self.node = node
#         self.container = container

class Chromosome():
    def __init__(self, nodes, containers):
        self.nodes = nodes
        self.containers = containers
        self.fitness = self.get_fitness()

    def get_fitness(self):
        #TODO get fitness, implement 5 objective fitness evalutaion:
        #TODO Equal task distribution
        #TODO Availability
        #TODO Power efficient
        #TODO Resources utilization balancing
        #TODO Unassigned tasks reduction
        return

class GeneticAlgorithm():
    def __init__(self, population_size, mat_pool_size, tournament_size, elite_size, max_generations, mutation_rate, nodes_num, containers_num):
        self.population_size = population_size
        self.mat_pool_size = mat_pool_size
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate 
        self.nodes_num = nodes_num
        self.containers_num = containers_num
        self.nodes = self.create_nodes()
        self.containers = self.create_containers()

    def create_nodes(self):
        nodes_list = []
        cpu_specified = 6  #TODO make configs diverse 
        memory_specified = 1000
        for _ in range(self.nodes_num):
            node = Node(cpu_specified, memory_specified, cpu_specified, memory_specified)
            nodes_list.append(node)
        return nodes_list
    
    def create_containers(self):
        containers_list = []
        cpu_required = 2
        memory_required = 200
        for _ in range(self.containers_num):
            container = Container(cpu_required, memory_required)
            containers_list.append(container)
        return containers_list

    def generate_chromosome(self):
        #size of the chromosome is equal to the number of containers
        #initial solution, is generated by assigning each container to a random node.
        containers = copy.deepcopy(self.containers) 
        nodes = []
        for _ in range(self.containers_num):
            nodes.append(random.choice(self.nodes))
        chromosome = Chromosome(nodes, containers)
        return chromosome

    def create_initial_population(self):
        #TODO initial_population
        initial_population = []
        for _ in range(self.population_size):
            initial_population.append(self.generate_chromosome())
        return initial_population

    def crossover(self, p1, p2):  #p1 and p2 are of type Chromosome
        #TODO crossover
        crossover_point = random.randint(0, len(p1.containers) - 1)
        rand_int = random.randint(0, 1)
        if rand_int == 1:
            child_nodes = (p1.nodes[:crossover_point] +  #here could be problem with synchronization, as node should be passed by reference nad not by instance
                     p2.nodes[crossover_point:])
        else:
            child_nodes = (p2.nodes[crossover_point:] +
                     p1.nodes[:crossover_point])
        return Chromosome(nodes, copy.deepcopy(p1.containers))

    def mutate(self, chromosome, mutation_type):  
        #TODO mutation
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
        return
    
    def change_mutation(self, chromosome):
        #TODO mutation type 1
        return

    def assign_unassigned_mutation(self, chromosome):
        #TODO mutation type 2
        return

    def unassign_assigned_mutation(self, chromosome):
        #TODO mutation type 3
        return


    def selection(self, population):
        #TODO selection
        return

    def generate_solution(self): 
        #TODO NSGA III
        population = self.create_initial_population()
        best_pareto_front = None
        for generation in range(self.max_generations):
            break
        return best_pareto_front

# parses command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, default=300, help="population size")
    parser.add_argument('-ms', type=int, default=150, help="mating pool size")
    parser.add_argument('-ts', type=int, default=7, help="tournament size")
    parser.add_argument('-e', type=int, default=30, help="elite_size")
    parser.add_argument('-mg', type=int, default=50, help="max generations")
    parser.add_argument('-mr', type=float, default=0.3, help="mutation rate")
    parser.add_argument('-nn', type=int, default=5, help="nodes number")
    parser.add_argument('-cn', type=int, default=8, help="containers number")
    args = parser.parse_args()
    return args.s, args.ms, args.ts, args.e, args.mg, args.mr, args.nn, args.cn

def main():
    population_size, mat_pool_size, tournament_size, elite_size, max_generations, mutation_rate, nodes_num, containers_num = parse_arguments()
    gen_algo = GeneticAlgorithm(population_size, mat_pool_size, tournament_size, elite_size, max_generations, mutation_rate, nodes_num, containers_num)
    solution = gen_algo.generate_solution()

if __name__ == "__main__":
    main()