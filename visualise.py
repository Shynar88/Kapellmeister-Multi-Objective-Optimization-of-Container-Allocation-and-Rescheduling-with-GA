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
    plt.savefig('compare.png')
    plt.close()

def optimal_front_at_gen(x_obj,y_obj,x_obj_name,y_obj_name,title):
    plt.scatter(x_obj,y_obj,s=8)
    plt.xlabel(x_obj_name)
    plt.ylabel(y_obj_name)
    plt.title(title)
    plt.show()

def visualize_history(history):

    n_gen, pop_size, n_obj = history.shape

    fig, axs = plt.subplots(n_obj, 1)
    axs[-1].set_xlabel("Generations")

    for i in range(n_obj):
        x = np.arange(1,n_gen+1)
        axs[i].set_xlim(0, n_gen+1)
        axs[i].set_ylabel('obj_' + str(i+1))
        #for j in range(n_gen):
        #    x = np.ones(pop_size) * (j + 1)
        #    axs[i].plot(x, history[j,:,i], 'r.')
        max_obj = np.max(history[:,:,i], axis=1)
        min_obj = np.min(history[:,:,i], axis=1)
        axs[i].plot(x, max_obj, 'r-')
        axs[i].plot(x, min_obj, 'r-')
        axs[i].fill_between(x, max_obj, min_obj, where=max_obj>min_obj, facecolor='red', alpha=0.1)

        average = np.sum(history[:,:,i], axis=1) / pop_size
        x = np.arange(1, n_gen+1)
        axs[i].plot(x, average, 'b--')

    fig.set_size_inches(10,10)
    plt.savefig('plot.png')
    plt.close()
