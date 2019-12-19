from chromosome import *

""" Node and Container tests """

# Node init
max_cpu, max_memory = 10, 1000
n = Node(max_cpu, max_memory)

assert n.max_power - 298.115 < 0.01
assert n.idle_power - 233.8425 < 0.01

# Containers init
required_cpu_1, required_memory_1, task_type_1 = 2, 100, None
c1 = Container(required_cpu_1, required_memory_1, task_type_1)

required_cpu_2, required_memory_2, task_type_2 = 10, 300, None
c2 = Container(required_cpu_2, required_memory_2, task_type_2)

# Assign
c1.assign_node(n)
assert c1.node == n
assert n.free_cpu == n.max_cpu - required_cpu_1
assert n.free_memory == n.max_memory - required_memory_1
assert n.feasable == True
c2.assign_node(n)
assert n.feasable == False
c1.unassign_node()
c2.unassign_node()
assert n.free_cpu == n.max_cpu
assert n.free_memory == n.max_memory
assert n.feasable == True

""" Objective functions """

nodes = []
for i in range(4):
    nodes.append(Node(10, 1000))

containers = []
for i in range(2):
    containers.append(Container(2,100,1))
for i in range(3):
    containers.append(Container(2,100,2))
for i in range(2):
    containers.append(Container(2,100,3))
for i in range(1):
    containers.append(Container(2,100,4))
for i in range(2):
    containers.append(Container(2,100,5))
#1
containers[0].assign_node(nodes[0])
containers[1].assign_node(nodes[1])
#2
containers[2].assign_node(nodes[0])
containers[3].assign_node(nodes[0])
containers[4].assign_node(nodes[1])
#3
containers[5].assign_node(nodes[0])
containers[6].assign_node(nodes[1])
#4
containers[7].assign_node(nodes[2])
#5
containers[8].assign_node(nodes[3])
containers[9].assign_node(nodes[3])

cr = Chromosome([],[])
cr.containers = containers
cr.nodes = nodes
scores = cr.get_fitness()

assert scores[0] == 20
assert scores[1] == 12
assert scores[2] - 1031.778 < 0.01
assert scores[3] == 100
assert scores[4] == 0

containers[8].unassign_node()
containers[9].unassign_node()

scores = cr.get_fitness()

assert scores[4] == 2
kwargs = {  'n_obj':5,
            'population_size':200,
            'tournament_size':7,
            'generations':100,
            'mutation_rate':0.3,
            'n_containers':300,
            'n_services':100,
            'containers_cpu_bounds':(0,1),
            'containers_memory_bounds':(25,1000),
            'nodes_cpu_types':[1,2,4,8,16],
            'nodes_memory_types':[2048,4096,8192,16384,32768],
            'nodes_per_type':20,
        }
    cpu_A1 = [1,2,4,8,16]
    mem_A1 = [2048,4096,8192,16384,32768]


print("Tests completed")
