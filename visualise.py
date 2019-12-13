import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def obj_over_configs(x_config_names,y_kub,y_nsga,obj_name,title):
    x_idx = np.arange(len(x_config_names))
    width = 0.3
    plt.bar(x_idx-width/2,y_kub,width=width,label="Kubernetes")
    plt.bar(x_idx+width/2,y_nsga,width=width,label="NSGA-II")
    plt.xlabel('Fitness Objectives')
    plt.ylabel(obj_name)
    plt.title(title)
    plt.legend()
    plt.show()

def obj_over_configs_rescheduling(x_config_names,y_kub,y_nsga,obj_name,title):
    x_idx = np.arange(len(x_config_names))
    width = 0.3
    plt.bar(x_idx-width/2,y_kub,width=width,label="NSGA-II(Rescheduling)")
    plt.bar(x_idx+width/2,y_nsga,width=width,label="NSGA-II(Scheduling)")
    plt.xlabel('Fitness Objectives')
    plt.ylabel(obj_name)
    plt.title(title)
    plt.legend()
    plt.show()
    
"""
#filler data
a=['1req-1app','1req-1.5app','1.5req-1app','1.5req-1.5app','2req-2app']
kub=[]
nsga=[]
for i in range(5):
    kub.append(np.random.randint(10,50))
    nsga.append(np.random.randint(10,50))
obj_over_configs(a,kub,nsga,"network distance","250 machines")
"""
def optimal_front_at_gen(x_obj,y_obj,x_obj_name,y_obj_name,title):
    plt.scatter(x_obj,y_obj,s=8)
    plt.xlabel(x_obj_name)
    plt.ylabel(y_obj_name)
    plt.title(title)
    plt.show()
"""
#filler data
a=[]
b=[]
for i in range(10):
    a.append(np.random.randint(0,100))
    b.append(np.random.randint(0,100))
optimal_front_at_gen(a,b,"Balanced Cluster","Network distance","Generation 40")
"""