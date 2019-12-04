import main
import kubernetes
import nsga_3
import numpy as np

"""
1)make physical configs
2)decide workload
3)get allocation from kub
4)get allocation from nsga
5)measure quality of kub allocation vs nsga allocation
6)show graphs of kub vs nsga

7)look at history of nsga and make graphs of front
"""
class Experiment:
    def __init__(self,node_cpu,node_mem,per_type,service_cpu,service_mem,total_services,service_cpu_low,service_cpu_high,service_mem_low,service_mem_high):
        self.node_cpu=node_cpu
        self.node_mem=node_mem
        self.per_type=per_type
        self.service_cpu=service_cpu
        self.service_mem=service_mem
        self.total_services=total_services
        self.service_cpu_low=service_cpu_low
        self.service_cpu_high=service_cpu_high
        self.service_mem_low=service_mem_low
        self.service_mem_high=service_mem_high
        
    def make_nodes(self): #physical config
        nodes=[]
        id=0 #first node has id 0, not 1
        
        #make nodes, only using A1 at the moment
        for i in range(len(self.node_cpu)): #for each kind of node
            for j in range(self.per_type): #for each node of its kind
                nodes.append(kubernetes.Node(
                    self.node_cpu[j],
                    self.node_mem[j],
                    id,"",[]
                ))
                id=id+1
        
        return nodes

    def make_containers(self): #workload
        #assumption is that a service is equivalent to a set of containers
        #all tasks in a service have common requirements because they are replicas
        
        #service specifications
        tasks_per_service =[3 for i in range(self.total_services)] #add randomness
        for i in range(self.total_services):
            self.service_cpu.append(np.random.randint(self.service_cpu_low,self.service_cpu_high))
            self.service_mem.append(np.random.randint(self.service_mem_low,self.service_mem_high))
        
        #make containers    
        containers=[]
        id=0
        for i in range(len(tasks_per_service)):
            for j in range(tasks_per_service[i]):
                containers.append(kubernetes.Container(
                    self.service_cpu[i],
                    self.service_mem[i],
                    id,""
                ))
                id+=1
        return containers

def get_kub_allocations(nodes,containers):
    #for kubernetes
    for i in range(len(containers)):
        chosen_node = kubernetes.least_request_priority(nodes,containers[i])
        id=chosen_node.id
        for j in range(len(nodes)):
            if nodes[j].id==id:
                nodes[j].allocate(containers[i])

    return nodes

def get_nsga_allocations(nodes,containers):
    #for nsga
    ga=main.GeneticAlgorithm(300,150,7,30,50,0.3,5,8) #used default values
    front=ga.generate_solution()
    #get solution from front that has lowest average objective value and return it
    
    pass

def evaluate_allocation():
    evaluation = []
    
    return evaluation    
def main():
    #AWS A1
    per_type_A1 = 20
    cpu_A1 = [1,2,4,8,16]
    mem_A1 = [2048,4096,8192,16384,32768]
    #AWS M4
    per_type_M4 = 40
    cpu_M4 = [2,4,8,16,40,64]
    mem_M4 = [8192,16384,32768,65536,163840,262144]
    #AWS M5a
    per_type_M5a = 80
    cpu_M5a = [2,4,8,16,32,48,64,96]
    mem_M5a = [8192,16384,32768,65536,131072,196608,262144,393216]
    
    Exp_1=Experiment(cpu_A1,mem_A1,per_type_A1,[],[],100,0,2,25,1000)
    Exp_2=Experiment(cpu_M4,mem_M4,per_type_M4,[],[],200,1,2,50,2000)
    Exp_3=Experiment(cpu_M5a,mem_M5a,per_type_M5a,[],[],400,2,4,200,4000)
    
    #instantiate nodes from Node class using specifications
    nodes=Exp_1.make_nodes() #list of nodes of Node class
    #instantiate containers from Container class using specifications
    containers=Exp_1.make_containers() #list of containers of container class
    population_size, mat_pool_size, tournament_size, elite_size, max_generations, mutation_rate, nodes_num, containers_num = main.parse_arguments()
    ga = main.GeneticAlgorithm(population_size, mat_pool_size, tournament_size, elite_size, max_generations, mutation_rate, nodes_num, containers_num,nodes,containers)
    #get the kubernetes allocation for the nodes and containers made
    kub_alloc=get_kub_allocations(nodes,containers) 
    nsga_alloc=get_nsga_allocations(nodes,containers) #likewise
    
                              