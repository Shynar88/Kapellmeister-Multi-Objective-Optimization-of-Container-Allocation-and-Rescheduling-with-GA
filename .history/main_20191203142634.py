import argparse
import math
import random
import operator
import sys
import copy
import numpy as np
import logging

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
    def __init__(self, node_ids, containers, nodes_info):
        self.node_ids = node_ids #node ids
        self.containers = containers
        self.nodes_info = nodes_info #for tracking the resource usage per chromosome 
        self.fitness = self.get_fitness()

    def get_fitness(self):
        #TODO get fitness, implement 5 objective fitness evalutaion:
        #TODO Equal task distribution
        #TODO Availability
        #TODO Power efficient
        #TODO Resources utilization balancing
        #TODO Unassigned tasks reduction
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
            v += abs(node.remaining_cpu-node.remaining_memory)
        return
    def off_5(self):
        node_ids = self.node_ids
        v = 0
        for node_id in node_ids:
            if (node_id == None):
                v += 1
                continue
        return v

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
        # in Table 6 there are different settings on number of nodes and their specifications
        nodes_list = []
        cpu_specified = 4  #according to paper
        memory_specified = 4000 #according to paper
        max_power = 160 
        idle_power = 80
        for i in range(self.nodes_num):
            node = Node(cpu_specified, memory_specified, i, max_power, idle_power)
            nodes_list.append(node)
        return nodes_list
    
    def create_containers(self):
        # in Table 6 there are different settings on number of containers and their specifications
        containers_list = []
        cpu_required = 2
        memory_required = 200
        task_type = "A"
        for _ in range(self.containers_num):
            container = Container(cpu_required, memory_required, task_type)
            containers_list.append(container)
        return containers_list

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
        chromosome = Chromosome(node_ids, containers, nodes_info)
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
        chromosome = Chromosome(node_ids, containers, nodes_info)
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
            participants = random.sample(population, 15) #self.tournament_size
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
        #TODO NSGA III
        population = self.create_initial_population()
        number_of_objectives = 5
        divisions = 6
        for i in range(self.max_generations):
            new_population = self.generate_new_population(population)
            combined_population = population + new_population
            combined_population_coords = np.array([p.fitness for p in combined_population])
            selected_indices = nsga3_dummy(combined_population_coords, divisions)
            population, best_front_indices = combined_population[selected_indices], selected_indices
            write_log(population)
        best_pareto_front = combined_population[best_front_indices]
        return best_pareto_front

def nsga3_dummy(population_coords, divisions):
    pop_length = population_coords.shape[0]
    selected_indices = np.arange(0, int(pop_length/2))
    return selected_indices

# logging data
def write_log(population):
    # LOG_FORMAT = "%(asctime)s %(message)s"
    logging.basicConfig(filename = "fitness.log",
                        level = logging.DEBUG,
                        format = LOG_FORMAT)
    logger = logging.getLogger()
    #make logging every 100 generations
    logger.info("Numbers in num_list are: {}".format(' '.join(map(str, num_list))))

# parses command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, default=300, help="population size") #212
    parser.add_argument('-ms', type=int, default=150, help="mating pool size") #106
    parser.add_argument('-ts', type=int, default=7, help="tournament size")
    parser.add_argument('-e', type=int, default=30, help="elite_size")
    parser.add_argument('-mg', type=int, default=50, help="max generations") #1000
    parser.add_argument('-mr', type=float, default=0.3, help="mutation rate") #0.3
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