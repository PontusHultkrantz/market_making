#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_decision_boundaries(l_m,l_p,q_grid):
    
    l = np.zeros_like(l_p)
    l[l_m==1] = 1
    l[l_p==1] = 2
    l[(l_p==1) & (l_m==1)] = 3
    
    # Plot decision boundaries
    fig,ax = plt.subplots(figsize=(4,4))
    im     = ax.imshow(l, aspect='auto')
    
    # Configure vertical axis
    ax.set_yticks(range(0,len(q_grid)))
    ax.set_yticklabels(q_grid)
    for label in ax.yaxis.get_ticklabels():
        label.set_visible(False)
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(True)
        
    # Configure horizontal axis
    N_steps = l_m.shape[1]
    t_grid  = np.linspace(0,1,N_steps)
    ax.set_xticks([i for i in range(0,N_steps)[::50]] + [N_steps])
    ax.set_xticklabels([np.round(t_grid[i],2) for i in range(0,N_steps)[::50]] + [np.round(t_grid[N_steps-1],2)])
    
    # Create labels
    values  = np.unique(l.ravel())
    colors  = [ im.cmap(im.norm(value)) for value in values]
    labels  = {0:"Do nothing",1:"Buy",2:"Sell",3:"Buy and Sell"}
    patches = [ mpatches.Patch(color=colors[i], label=labels[values[i]] ) for i in range(len(values)) ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.tight_layout()