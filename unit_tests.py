import argparse
import math
import random
import operator
import sys
import main as ga
def off_1_test():
    n1 = ga.Node (4, 16000, 4, 16000, 140, 80)
    n2 = ga.Node (4, 8000, 4, 8000, 160, 90)
    n3 = ga.Node (2, 1024, 2, 1024, 180, 100)
    n4 = ga.Node (6, 6512, 6, 6512, 200, 110)
    nodes_info = [n1,n2,n3,n4]
    node_ids = [0, 0, 0, 0, 2, 2, 3, None]
    c1 = ga.Container(1, 128, "A")
    c2 = ga.Container(1, 128, "B")
    c3 = ga.Container(1, 128, "C")
    c4 = ga.Container(1, 128, "D")
    c5 = ga.Container(1, 128, "E")
    c6 = ga.Container(1, 128, "F")
    c7 = ga.Container(1, 128, "G")
    c8 = ga.Container(1, 128, "H")
    containers = [c1, c2, c3, c4, c5, c6, c7, c8]
    ex1 = ga.Chromosome(node_ids, containers, nodes_info)
    n1.containers_list = [c1, c2, c3, c4]
    n3.containers_list = [c5, c6]
    n4.containers_list = [c7]
    return ex1.get_fitness()
def off_2_test():
    n1 = ga.Node (4, 16000, 4, 16000, 140, 80)
    n2 = ga.Node (4, 8000, 4, 8000, 160, 90)
    n3 = ga.Node (6, 10024, 2, 1024, 180, 100)
    n4 = ga.Node (6, 6512, 6, 6512, 200, 110)
    nodes_info = [n1,n2,n3,n4]
    node_ids = [0, 0, 0, 3, 2, 2, 2, 2, 1, 3, 1, 1, 3]
    c1 = ga.Container(1, 128, "A")
    c2 = ga.Container(1, 128, "A")
    c3 = ga.Container(1, 128, "A")
    c4 = ga.Container(1, 128, "A")
    c5 = ga.Container(1, 128, "D")
    c6 = ga.Container(1, 128, "D")
    c7 = ga.Container(1, 128, "D")
    c8 = ga.Container(1, 128, "D")
    c9 = ga.Container(1, 128, "C")
    c10 = ga.Container(1, 128, "C")
    c11 = ga.Container(1, 128, "B")
    c12 = ga.Container(1, 128, "B")
    c13 = ga.Container(1, 128, "B")
    containers = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13]
    ex1 = ga.Chromosome(node_ids, containers, nodes_info)
    n1.containers_list = [c1, c2, c3]
    n2.containers_list = [c9, c11, c12]
    n3.containers_list = [c5, c6, c7, c8]
    n4.containers_list = [c4, c10, c13]
    return ex1.get_fitness()
def off_3_test():
    n1 = ga.Node (4, 16000, 4, 16000, 140, 80)
    n2 = ga.Node (4, 8000, 4, 8000, 160, 90)
    n3 = ga.Node (6, 10024, 2, 1024, 180, 100)
    n4 = ga.Node (6, 6512, 6, 6512, 200, 110)
    nodes_info = [n1,n2,n3,n4]
    node_ids = [0, 0, 0, 3, 2, 2, 2, 2, 1, 3, 1, 1, 3]
    c1 = ga.Container(1, 128, "A")
    c2 = ga.Container(1, 128, "A")
    c3 = ga.Container(1, 128, "A")
    c4 = ga.Container(1, 128, "A")
    c5 = ga.Container(1, 128, "D")
    c6 = ga.Container(1, 128, "D")
    c7 = ga.Container(1, 128, "D")
    c8 = ga.Container(1, 128, "D")
    c9 = ga.Container(1, 128, "C")
    c10 = ga.Container(1, 128, "C")
    c11 = ga.Container(1, 128, "B")
    c12 = ga.Container(1, 128, "B")
    c13 = ga.Container(1, 128, "B")
    containers = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13]
    ex1 = ga.Chromosome(node_ids, containers, nodes_info)
    n1.containers_list = [c1, c2, c3]
    n2.containers_list = [c9, c11, c12]
    n3.containers_list = [c5, c6, c7, c8]
    n4.containers_list = [c4, c10, c13]
    return ex1.get_fitness()