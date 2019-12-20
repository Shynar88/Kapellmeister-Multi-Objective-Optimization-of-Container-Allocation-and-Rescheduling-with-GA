from nsga import *
from chromosome import *
from visualize import *
from kubernetes import *
from copy import deepcopy

"""
# A1
n_containers = 300
n_services = 100
containers_cpu_bounds = (0,1)
containers_memory_bounds = (25,1000)
nodes_cpu_types = [1,2,4,8,16]
nodes_memory_types = [2048,4096,8192,16384,32768]
nodes_per_type = 20
"""

# M4
n_containers = 600
n_services = 200
containers_cpu_bounds = (1,2)
containers_memory_bounds = (50,2000)
nodes_cpu_types = [2,4,8,16,40]
nodes_memory_types = [8*1024,16*1024,32*1024,64*1024,160*1024]
nodes_per_type = 40

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

kub_chromosome = generate_kubernetes_solution(deepcopy(containers), deepcopy(nodes))

kwargs = \
{
       'n_partitions':6,
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
nsga_chromosome = get_nsga_best_of_front(r)
visualize_kub_nsga(kub_chromosome.get_fitness(), nsga_chromosome.get_fitness())
