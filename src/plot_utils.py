#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_decision_boundaries(result,size=(4,4)):
    
    l = np.zeros_like(result.l_p)
    l[result.l_m==1] = 1
    l[result.l_p==1] = 2
    l[(result.l_p==1) & (result.l_m==1)] = 3
    
    # Plot decision boundaries
    fig,ax = plt.subplots(figsize=size)
    im     = ax.imshow(l, aspect='auto')
    
    # Configure horizontal axis
    ax.set_xticks([i for i in range(0,result.N_steps)[::50]] + [result.N_steps])
    ax.set_xticklabels([np.round(result.t_grid[i],2) for i in range(0,result.N_steps)[::50]] + [np.round(result.t_grid[result.N_steps-1],2)])
    
    # Configure vertical axis
    ax.set_yticks(range(0,len(result.q_grid),2))
    #ax.set_yticklabels(result.q_grid[::2])
    ax.set_yticklabels(result.q_grid)
    
#    # Create labels
#    values  = np.unique(l.ravel())
#    colors  = [ im.cmap(im.norm(value)) for value in values]
#    labels  = {0:"Do nothing",1:"Buy",2:"Sell",3:"Buy and Sell"}
#    patches = [ mpatches.Patch(color=colors[i], label=labels[values[i]] ) for i in range(len(values)) ]
#    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
#    
#    
#    # Create axis labels
#    ax.set_xlabel('Trading time')
#    ax.set_ylabel('Inventory/Position')
#    ax.set_title('Optimal decisions')
    
    plt.tight_layout()
    plt.show()