from collections import namedtuple
Resources = namedtuple("Resources", "used_cpu max_cpu used_mem max_mem")
class Node:
    def __init__(self, cpu_specified, memory_specified, id, max_power, idle_power):
        self.cpu_specified = cpu_specified
        self.memory_specified = memory_specified
        self.id=id
        self.remaining_cpu = cpu_specified
        self.remaining_memory = memory_specified
        self.containers_list = []
        self.max_power = max_power
        self.idle_power = idle_power
    def __repr__(self):
        return "Node_" + str(self.id)
    def get_containers_list(self):
        return self.containers_list
    def allocate(self, container):
        self.containers_list.append(container)
        assert (container.required_cpu <= self.remaining_cpu),"CPU limit exceeded"
        assert (container.required_memory <= self.remaining_memory),"Memory limit exceeded"
        self.remaining_cpu -= container.required_cpu
        self.remaining_memory -= container.required_memory
    def deallocate(self, container):
        self.containers_list.remove(container)
        self.remaining_cpu -= container.required_cpu
        self.remaining_memory -= container.required_memory
    def resources(self):
        return Resources(self.remaining_cpu, self.cpu_specified, self.remaining_memory, self.memory_specified)
class Container:
    def __init__(self, required_cpu, required_memory, task_type):
        self.required_cpu = required_cpu
        self.required_memory = required_memory #size in MB
        self.task_type = task_type
class Cluster:
    def __init__(self, nodes):
        self.nodes=nodes
    def add_node(self, node):
        self.nodes.append(node)
    def remove_node(self, node):
        self.nodes.remove(node)
    def allocate(self, index, container):
        self.nodes[index].allocate(container)
    def deallocate(self, index, container):
        self.nodes[index].deallocate(container)
    def resources(self):
        return [n.resources() for n in self.nodes]
def pod_fits_resources(nodes, container):
    fit_nodes = []
    for n in nodes:
        if n.cpu_specified >= n.remaining_cpu - container.required_cpu \
            and n.memory_specified >= n.remaining_memory - container.required_memory:
            fit_nodes.append(n)
    return fit_nodes
def least_request_priority(nodes, container):
    best_score = 0
    best_candidate = None
    fit_nodes = pod_fits_resources(nodes, container)
    for n in fit_nodes:
        free_cpu_frac = max(n.remaining_cpu - container.required_cpu, 0)/n.cpu_specified
        free_mem_frac = max(n.remaining_memory- container.required_memory, 0)/n.memory_specified
        score = (free_cpu_frac + free_mem_frac)/2
        if score > best_score:
            best_score = score
            best_candidate = n
    return best_candidate

# def main():
#     # n1 = Node(4,16000,1,"",[])
#     # n2 = Node(4,8000,2,"",[])
#     # n3 = Node(2,1024,3,"",[])
#     # n4 = Node(6,6512,4,"",[])
#     # c = []
#     # c.append(Container(1,128,1,""))
#     # c.append(Container(2,1000,2,""))
#     # c.append(Container(4,2000,3,""))
#     # 1
#     # [n.resources() for n in nodes]
#     # n1.used_cpu=4
#     # n2.used_mem=8000
#     # least_request_priority(nodes, c[1])
# if __name__ == "__main__":
#     main()