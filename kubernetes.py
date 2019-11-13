from collections import namedtuple
Resources = namedtuple("Resources", "used_cpu max_cpu used_mem max_mem")
class Node:
    def __init__(self, max_cpu, max_mem, id, status, containers=[]):
        self.max_cpu=max_cpu
        self.max_mem=max_mem
        self.id=id
        self.used_cpu = 0
        self.used_mem = 0
        self.status=status
        self.containers=containers
    def __repr__(self):
        return "Node_" + str(self.id)
    def allocate(self, container):
        self.containers.append(container)
        assert (container.cpu <= self.max_cpu - self.used_cpu),"CPU limit exceeded"
        assert (container.mem <= self.max_mem - self.used_mem),"Memory limit exceeded"
        self.used_cpu += container.cpu
        self.used_mem += container.mem
    def deallocate(self, container):
        self.containers.remove(container)
        self.used_cpu -= container.cpu
        self.used_mem -= container.mem
    def resources(self):
        return Resources(self.used_cpu, self.max_cpu, self.used_mem, self.max_mem)
class Container:
    def __init__(self, cpu, mem, id, status):
        self.cpu=cpu
        self.mem=mem
        self.id=id
        self.status=status
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
        if n.max_cpu >= n.used_cpu + container.cpu \
            and n.max_mem >= n.used_mem + container.mem:
            fit_nodes.append(n)
    return fit_nodes
def least_request_priority(nodes, container):
    best_score = 0
    best_canbdidate = None
    fit_nodes = pod_fits_resources(nodes, container)
    for n in fit_nodes:
        free_cpu_frac = max(n.max_cpu - n.used_cpu - container.cpu, 0)/n.max_cpu
        free_mem_frac = max(n.max_mem - n.used_mem - container.mem, 0)/n.max_mem
        score = (free_cpu_frac + free_mem_frac)/2
        if score > best_score:
            best_score = score
            best_candidate = n
    return n

def main():
    n1 = Node(4,16000,1,"",[])
    n2 = Node(4,8000,2,"",[])
    n3 = Node(2,1024,3,"",[])
    n4 = Node(6,6512,4,"",[])
    c = []
    c.append(Container(1,128,1,""))
    c.append(Container(2,1000,2,""))
    c.append(Container(4,2000,3,""))
    nodes = [n1,n2,n3,n4]
    [n.resources() for n in nodes]
    n1.used_cpu=4
    n2.used_mem=8000
    least_request_priority(nodes, c[1])
if __name__ == "__main__":
    main()