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

    def get_distance_to(self, other):
        dx = (other.x_coord - self.x_coord) ** 2
        dy = (other.y_coord - self.y_coord) ** 2
        return math.sqrt(dx + dy)

class Chromosome():
    def __init__(self, route):
        self.route = route
        self.route_distance = self.get_route_distance()
        self.fitness = self.get_fitness()

    def get_route_distance(self):
        distance = 0
        for i in range(len(self.route)):
            src = self.route[i]
            dest = self.route[i + 1] if i + 1 < len(self.route) else self.route[0]
            distance += src.get_distance_to(dest)
        return distance

    def get_fitness(self):
        return 1 / self.route_distance

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

    def crossover(self, p1, p2): # proposing good crossover method https://www.hindawi.com/journals/cin/2017/7430125/
        #implement simple crossover then try to enhance it 
        #ordered crossover 
        li = 0
        hi = 0
        while hi <= li:
            li = int(random.random() * len(p1.route)) 
            hi = int(random.random() * len(p1.route))
        chunk = p1.route[li:hi] 
        child_route = []
        not_used_el_in_p2 = [el for el in p2.route if el not in chunk]
        pointer = 0
        for _ in range(li):
            child_route.append(not_used_el_in_p2[pointer])
            pointer += 1
        child_route += chunk
        for _ in range(hi, len(p1.route)):
            child_route.append(not_used_el_in_p2[pointer])
            pointer += 1
        child = Instance(child_route)
        return child

    def mutate(self, instance):  #mutation operator is weak. increase mutation rate 
        # RSM mutation
        # if random.random() < self.mutation_rate:
        #     route = instance.route.copy()
        #     li = 0
        #     hi = 0
        #     while hi <= li:
        #         li = int(random.random() * len(route)) 
        #         hi = int(random.random() * len(route))
        #     while li < hi:
        #         route[li], route[hi] = route[hi], route[li]
        #         li += 1
        #         hi -= 1
        #     return Instance(route)
        # return instance

        # mutation by swapping
        if random.random() < self.mutation_rate:
            route = instance.route.copy()
            i1, i2 = random.sample(range(len(self.cities_list)), 2)
            route[i1], route[i2] = route[i2], route[i1]
            return Instance(route)
        return instance

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