import argparse
import math
import random
import operator
import sys

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
        self.required_memory = required_memory

class Gene():
    def __init__(self, node, container):
        self.node = node
        self.container = container

class Chromosome():
    def __init__(self, gene):
        self.gene = gene
        self.fitness = self.get_fitness()

    def get_fitness(self):
        #TODO get fitness, implement 5(TODO TODO TODO TODO TODO) objective fitness evalutaion 
        return

class GeneticAlgorithm():
    def __init__(self, population_size, mat_pool_size, tournament_size, elite_size, max_generations, crossover_rate, mutation_rate, cities_list):
        self.population_size = population_size
        self.mat_pool_size = mat_pool_size
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate 
        self.cities_list = cities_list

    def create_initial_population(self):
        #TODO initial_population
        return 

    def crossover(self, p1, p2): 
        #TODO crossover
        return

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
    parser.add_argument('-p', type=str, default="a280.tsp", help="path to the input file")
    parser.add_argument('-s', type=int, default=300, help="population size")
    parser.add_argument('-ms', type=int, default=150, help="mating pool size")
    parser.add_argument('-ts', type=int, default=7, help="tournament size")
    parser.add_argument('-e', type=int, default=30, help="elite_size")
    parser.add_argument('-mg', type=int, default=50, help="max generations")
    parser.add_argument('-cr', type=float, default=0.3, help="crossover rate")
    parser.add_argument('-mr', type=float, default=0.3, help="mutation rate")
    args = parser.parse_args()
    return args.p, args.s, args.ms, args.ts, args.e, args.mg, args.cr, args.mr

def main():
    path, population_size, mat_pool_size, tournament_size, elite_size, max_generations, crossover_rate, mutation_rate = parse_arguments()
    coordinates_list = parse(path)
    cities_list = create_cities(coordinates_list)
    gen_algo = GeneticAlgorithm(population_size, mat_pool_size, tournament_size, elite_size, max_generations, crossover_rate, mutation_rate, cities_list)
    distance = gen_algo.generate_path()
    print(distance)

if __name__ == "__main__":
    main()