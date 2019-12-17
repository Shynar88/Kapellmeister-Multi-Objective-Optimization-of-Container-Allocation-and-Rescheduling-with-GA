import main as ga
import kubernetes
import visualise
from visualise import visualize_history
import numpy as np
from ast import literal_eval
import timeit

#-----Helper Functions -----------------------------------
def calc_idle_power(cores_new,mem_new):
    p_ref=171

    cores_ref=4
    p_frac_cpu=0.3
    p_cpu=p_ref*p_frac_cpu

    mem_ref=4000
    p_frac_mem=0.11
    p_mem=p_ref*p_frac_mem

    delta_p_cpu=(p_cpu*cores_new/cores_ref)-p_cpu
    delta_p_mem=(p_mem*mem_new/mem_ref)-p_mem

    return p_ref+delta_p_cpu+delta_p_mem

def calc_max_power(cores_new,mem_new):
    p_ref=218

    cores_ref=4
    p_frac_cpu=0.3
    p_cpu=p_ref*p_frac_cpu

    mem_ref=4000
    p_frac_mem=0.11
    p_mem=p_ref*p_frac_mem

    delta_p_cpu=(p_cpu*cores_new/cores_ref)-p_cpu
    delta_p_mem=(p_mem*mem_new/mem_ref)-p_mem

    return p_ref+delta_p_cpu+delta_p_mem

def find_assigned(nodes):
    node_ids=[]
    for i in range(len(nodes)):
        if len(nodes[i].get_containers_list())!=0:
            node_ids.append(nodes[i].id)
    return node_ids

def normalised_fitness_5(max,min,point): 
    (f1,f2,f3,f4,f5)=point.get_fitness()
    f=[f1,f2,f3,f4,f5]
    res=0
    for i in range(5):
        if max[i] != min[i]:
            res+=0.2*(f[i]-min[i])/(max[i]-min[i])
    return res

def normalised_fitness_6(max,min,point): 
    (f1,f2,f3,f4,f5,f6)=point.get_fitness()
    f=[f1,f2,f3,f4,f5,f6]
    res=0
    for i in range(6):
        if max[i] != min[i]:
            res+=0.2*(f[i]-min[i])/(max[i]-min[i])

    return res

def select_from_front_5(front):
    
    (f1,f2,f3,f4,f5)=front[0].get_fitness()
    max=[f1,f2,f3,f4,f5]
    min=[f1,f2,f3,f4,f5]
    
    #find max and min for each objective
    for i in range(len(front)):
        (f1,f2,f3,f4,f5)=front[i].get_fitness()
        f=[f1,f2,f3,f4,f5]
        for obj in range(5):
            if f[obj]>max[obj]:
                max[obj]=f[obj]  
            if f[obj]<min[obj]:
                min[obj]=f[obj] 

    index=0
    best=normalised_fitness_5(max,min,front[0])
    for i in range(1,len(front)):
        f=normalised_fitness_5(max,min,front[i])
        if f<best:
            index=i
            best=f
            
    return front[index]

def select_from_front_6(front):
    
    (f1,f2,f3,f4,f5, f6)=front[0].get_fitness()
    max=[f1,f2,f3,f4,f5,f6]
    min=[f1,f2,f3,f4,f5,f6]
    
    #find max and min for each objective
    for i in range(len(front)):
        (f1,f2,f3,f4,f5,f6)=front[i].get_fitness()
        f=[f1,f2,f3,f4,f5,f6]
        for obj in range(6):
            if f[obj]>max[obj]:
                max[obj]=f[obj]  
            if f[obj]<min[obj]:
                min[obj]=f[obj] 

    index=0
    best=normalised_fitness_6(max,min,front[0])
    for i in range(1,len(front)):
        f=normalised_fitness_6(max,min,front[i])
        if f<best:
            index=i
            best=f

    return front[index]

def parse_log_data():
        list_of_population_fitnesses = []
        for line in open("fitness.log", "r"):
            population_fitnesses = literal_eval(line)
            #print(len(population_fitnesses))
            list_of_population_fitnesses.append(population_fitnesses)
        #print(len(list_of_population_fitnesses))
        return list_of_population_fitnesses

#-----End of Helper functions----------------------------

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
        nodes_kub=[] #Will be instantiated as object of kubernetes.Node class
        nodes_ga=[] #same specifications as nodes_ga but instantiated as an object of ga.Node class
        id=0 #first node has id 0, not 1

        for i in range(len(self.node_cpu)): #for each kind of node
            for j in range(self.per_type): #for each node of its kind
                nodes_kub.append(kubernetes.Node(
                    self.node_cpu[i],
                    self.node_mem[i],
                    id,
                    calc_max_power(self.node_cpu[i],self.node_mem[i]),
                    calc_idle_power(self.node_cpu[i],self.node_mem[i]),
                ))
                nodes_ga.append(ga.Node(
                    self.node_cpu[i],
                    self.node_mem[i],
                    id,
                    calc_max_power(self.node_cpu[i],self.node_mem[i]),
                    calc_idle_power(self.node_cpu[i],self.node_mem[i]),
                ))
                id=id+1

        return (nodes_kub,nodes_ga)
    def make_containers(self): #workload
        #assumption is that a service is equivalent to a set of containers
        #all tasks in a service have common requirements because they are replicas

        #service specifications
        tasks_per_service =[np.random.randint(1,4) for i in range(self.total_services)] #add randomness
        for i in range(self.total_services):
            self.service_cpu.append(np.random.randint(self.service_cpu_low,self.service_cpu_high))
            self.service_mem.append(np.random.randint(self.service_mem_low,self.service_mem_high))

        #make containers    
        containers_kub=[] #Instantiated as objects of kubernetes.Container
        containers_ga=[] #Instantiated as objects of ga.Container
        id=0
        for i in range(len(tasks_per_service)):
            for j in range(tasks_per_service[i]):
                containers_kub.append(kubernetes.Container(
                    self.service_cpu[i],
                    self.service_mem[i],
                    str(id)

                ))
                containers_ga.append(ga.Container(
                    self.service_cpu[i],
                    self.service_mem[i],
                    str(id)

                ))
            id+=1
        return (containers_kub,containers_ga)

def get_kub_allocations(nodes,containers):
    for i in range(len(containers)):
        chosen_node = kubernetes.least_request_priority(nodes,containers[i])
        for j in range(len(nodes)):
            if nodes[j].id==chosen_node.id:
                nodes[j].allocate(containers[i])
    return nodes

def get_nsga_allocations(nodes,containers):
    genalg=ga.GeneticAlgorithm(100,7,200,0.3,nodes,containers,False,None) #CHANGE PARAMETERS FOR GA HERE
    front, history = genalg.generate_solution()
    return select_from_front_5(front), history

def get_nsga_rescheduling(nodes, containers, init_chromosome):
    genalg=ga.GeneticAlgorithm(100,7,200,0.3,nodes,containers,True,init_chromosome) #CHANGE PARAMETERS FOR GA HERE
    front, history = genalg.generate_solution()
    return select_from_front_6(front), history

def main():
    #1)make physical configs
    #2)decide workload
    #3)get allocation from kub
    #4)get allocation from nsga
    #5)measure quality of kub allocation vs nsga allocation
    #6)show graphs of kub vs final nsga 
    #7)look at history of nsga and make graphs of front

    start = timeit.timeit()
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

    #Parameters for instantiating an Experiment are node_cpu,node_mem,per_type,service_cpu,service_mem,total_services,service_cpu_low,service_cpu_high,service_mem_low,service_mem_high
    Exp_1=Experiment(cpu_A1,mem_A1,per_type_A1,[],[],100,0,2,25,1000) 
    Exp_2=Experiment(cpu_M4,mem_M4,per_type_M4,[],[],200,1,2,50,2000)
    Exp_3=Experiment(cpu_M5a,mem_M5a,per_type_M5a,[],[],400,2,4,200,4000)
    Exp_4=Experiment(cpu_M5a,mem_M5a,per_type_M5a,[],[],100,0,1,25,1000)

    Experiments=[Exp_1] #[Exp_1, Exp_2, Exp_3, Exp_4]
    for experiment in Experiments:

        (nodes_kub,nodes_ga)=experiment.make_nodes() #list of nodes of Node class
        (containers_kub,containers_ga)=experiment.make_containers() #list of containers of Container class

        kub_alloc=get_kub_allocations(nodes_kub,containers_kub) 
        chr=ga.Chromosome(find_assigned(kub_alloc),containers_kub,kub_alloc,False,None)
        (fa,fb,fc,fd,fe)=chr.get_fitness() #fitness of kubernetes allocation

        nsga_alloc, history = get_nsga_allocations(nodes_ga,containers_ga) 
        visualize_history(history, "5_obj.png")
        (f1,f2,f3,f4,f5)=nsga_alloc.get_fitness() #fitness of selected nsga solution

        x_names=['Obj1','Obj2','Obj3','Obj4','Obj5']
        kub=[fa,fb,fc,fd,fe]
        nsga=[f1,f2,f3,f4,f5]

        visualise.obj_over_configs(x_names,kub,nsga,"Objective Values","Perfomance of Kubernetes vs NSGA-3 on the 5 fitness objectives")

        end = timeit.timeit()
        print("Execution Time: %s" %(end - start))

        #Rescheduling Experiment
        #Adding dummy container, so that rescheduling is needed
        nodes = nsga_alloc.nodes_info
        min_cpu = 0
        min_mem = 0
        for node in nodes:
            if node.remaining_cpu < experiment.service_cpu_high and node.remaining_memory < experiment.service_mem_high :
                min_cpu = experiment.service_cpu_high
                min_mem = experiment.service_mem_high
                dummy_container = ga.Container(min_cpu, min_mem, len(containers_ga))
                containers_ga.append(dummy_container)
                #nsga_alloc.node_ids.append(node.id)
                #node.containers_list.append(dummy_container) #maybe then need to compute obj6 except last element
                break

        nsga_rescheduling, history = get_nsga_rescheduling(nodes_ga, containers_ga, nsga_alloc)
        visualize_history(history, '6_obj.png')
        (off1,off2, off3, off4, off5, off6) = nsga_rescheduling.get_fitness()
        
        #Rescheduling impemented as Scheduling
        nsga_re_scheduling, _ = get_nsga_allocations(nodes_ga, containers_ga)
        off_6 = nsga_re_scheduling.off_6(nsga_alloc)
        (off_1,off_2, off_3, off_4, off_5)= nsga_re_scheduling.get_fitness()

        x_names=['Obj1','Obj2','Obj3','Obj4','Obj5', 'Obj6']
        resch=[off1,off2, off3, off4, off5, off6]
        re_sch=[off_1,off_2, off_3, off_4, off_5, off_6]
        visualise.obj_over_configs_rescheduling(x_names,resch,re_sch,"Objective Values","Perfomance of Rescheduling on the 6 fitness objectives")



if __name__ == "__main__":
    main()
