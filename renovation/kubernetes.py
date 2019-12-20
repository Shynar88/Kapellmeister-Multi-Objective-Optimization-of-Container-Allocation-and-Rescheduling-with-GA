import numpy as np
from chromosome import Chromosome

def generate_kubernetes_solution(containers, nodes):

    for c in containers:
        """ Find nodes that fit resources """
        fit_nodes = []
        for n in nodes:
            if n.free_cpu >= c.required_cpu and n.free_memory >= c.required_memory:
                fit_nodes.append(n)

        """ Choose best node out of them """
        best_score = 0
        best_candidate = None
        for n in fit_nodes:
            free_cpu_frac = max(n.free_cpu, 0)/n.max_cpu
            free_mem_frac = max(n.free_memory, 0)/n.max_memory
            score = (free_cpu_frac + free_mem_frac)/2
            if score > best_score:
                best_score = score
                best_candidate = n
        c.assign_node(best_candidate)
    kub_chromosome = Chromosome([],[])
    kub_chromosome.containers = containers
    kub_chromosome.nodes = nodes
    return kub_chromosome

def get_nsga_score():
    pass
