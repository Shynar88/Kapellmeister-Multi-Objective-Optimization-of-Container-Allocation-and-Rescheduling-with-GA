import matplotlib.pyplot as plt
import numpy as np

def visualize_history(history, name):

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
    plt.savefig(name)
    plt.show()
    plt.close()

def visualize_kub_nsga(kub_scores,nsga_scores):
    width = 0.3
    x_idx = np.arange(1,len(nsga_scores)+1)
    plt.bar(x_idx-width/2,kub_scores,width=width,label="Kubernetes")
    plt.bar(x_idx+width/2,nsga_scores,width=width,label="NSGA-III")
    obj_names = ['obj_'+str(i) for i in range(1,len(nsga_scores)+1)]
    plt.xlabel('Fitness Objectives')
    plt.ylabel(obj_names)
    plt.title('Score comparison')
    plt.legend()
    plt.savefig('compare.png')
    plt.show()
    plt.close()
