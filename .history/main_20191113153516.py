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
        #TODO get fitness, implement 5 objective fitness evalutaion
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

    def generate_instance(self):
        route = random.sample(self.cities_list, len(self.cities_list))
        instance = Instance(route)
        return instance

    def create_initial_population(self):
        initial_population = []
        for _ in range(self.population_size):
            initial_population.append(self.generate_instance())
        return initial_population

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
        #experiment on selection way
        #implement the simple one
        #Tournament selection   P/2 size might be better
        mating_pool = []
        while len(mating_pool) < self.mat_pool_size:
            participants = random.sample(population, self.tournament_size)
            fittest = max(participants, key=operator.attrgetter('fitness'))
            mating_pool.append(fittest)
        return mating_pool

    def generate_path(self):
        #for logging
        f = open('logs.log', mode='w+')
        # Step 1. Create an initial population of P chromosomes.
        population = self.create_initial_population()
        shortest_ever = float("inf")
        shortest_ever_route = None
        # Step 2. Evaluate the fitness of each chromosome. done in create population
        for generation in range(self.max_generations):
            # Step 3. Choose P/2 parents from the current population.
            mating_pool = self.selection(population)
            population_sorted = sorted(population, key=lambda instance: instance.fitness, reverse=True)
            old_elite = population_sorted[:self.elite_size]
            new_population = old_elite
            while len(new_population) < self.population_size:
                # Step 4. Randomly select two parents to create offspring using crossover operator.
                parents = random.sample(mating_pool, 2)
                child = self.crossover(parents[0], parents[1])
                # Step 5. Apply mutation operators for minor changes in the results.
                child = self.mutate(child)
                new_population.append(child)
                # Step 6. Repeat Steps  4 and 5 until all parents are selected and mated.
            # Step 7. Replace old population of chromosomes with new one.
            population = new_population
            # Step 8. Evaluate the fitness of each chromosome in the new population. Already done in crossover when creating the child
            # Step 9. Terminate if the number of generations meets some upper bound; otherwise go to Step  3.
            if population_sorted[0].route_distance < shortest_ever:
                shortest_ever = population_sorted[0].route_distance
                shortest_ever_route = population_sorted[0].route
            f.write(f'{generation} {population_sorted[0].fitness}') 
            f.write('\n')
        write_csv(shortest_ever_route)
        return shortest_ever   

def write_csv(route):
    with open('solution.csv', mode='w+') as f:
        for city in route:
            f.write(city.index) 
            f.write('\n')   

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

# parses file, returns list of city coordinates(ex: [(x1, y1), ...])
def parse(file_path): #coordinates start from the 7th line, end with EOF
    cities = []
    f = open(file_path, "r")
    for _ in range(6):
        f.readline()
    for line in f:
        line_contents = line.split()
        if len(line_contents) == 3:
            cities.append((line_contents[0], line_contents[1], line_contents[2]))
    f.close()
    return cities

def create_cities(coordinates_list):
    cities_list = []
    for coordinates in coordinates_list:
        cities_list.append(City(coordinates[0], coordinates[1], coordinates[2]))
    return cities_list

def main():
    path, population_size, mat_pool_size, tournament_size, elite_size, max_generations, crossover_rate, mutation_rate = parse_arguments()
    coordinates_list = parse(path)
    cities_list = create_cities(coordinates_list)
    gen_algo = GeneticAlgorithm(population_size, mat_pool_size, tournament_size, elite_size, max_generations, crossover_rate, mutation_rate, cities_list)
    distance = gen_algo.generate_path()
    print(distance)

if __name__ == "__main__":
    main()