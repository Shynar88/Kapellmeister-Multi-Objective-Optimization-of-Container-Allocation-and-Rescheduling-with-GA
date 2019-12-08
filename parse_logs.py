from ast import literal_eval

def parse_log_data():
    list_of_population_fitnesses = []
    for line in open("fitness.log", "r"):
        population_fitnesses = literal_eval(line)
        print(len(population_fitnesses))
        list_of_population_fitnesses.append(population_fitnesses)
    print(len(list_of_population_fitnesses))
    return list_of_population_fitnesses

parse_log_data()