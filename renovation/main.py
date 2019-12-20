from nsga import *
from chromosome import *
from visualize import *
from copy import deepcopy

n_containers = 300
n_services = 100
containers_cpu_bounds = (0,1)
containers_memory_bounds = (25,1000)
nodes_cpu_types = [1,2,4,8,16]
nodes_memory_types = [2048,4096,8192,16384,32768]
nodes_per_type = 20

containers = generate_containers(
                                n_containers,
                                n_services,
                                containers_cpu_bounds,
                                containers_memory_bounds,
                                )

nodes = generate_nodes(
                      nodes_cpu_types,
                      nodes_memory_types,
                      nodes_per_type,
                      )

#kub_chromosome = generate_kubernetes_solution(deepcopy(containers), deepcopy(nodes))


kwargs = \
{
       'n_partitions':20,
       'n_obj':5,
       'population_size':200,
       'tournament_size':7,
       'generations':3,
       'mutation_rate':0.3,
       'containers':containers,
       'nodes':nodes,
 }

r, h = solve(kwargs)
visualize_history(h, "plot.png")
