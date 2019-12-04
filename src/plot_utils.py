#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_decision_boundaries(result,size=(4,4)):
    
    l = np.zeros_like(result.l_p)
    l[result.l_m==1] = 1
    l[result.l_p==1] = 2
    l[(result.l_p==1) & (result.l_m==1)] = 3
    
    fig,ax = plt.subplots(3,1,figsize=size)
    
    # Plot sell decisions
    im_1 = ax[0].imshow(result.l_p, aspect='auto')
    ax[0].set_xlabel('Trading time')
    ax[0].set_ylabel('Inventory/Position')    
    ax[0].set_title('Offer side')
    ax[0].set_yticks(range(0,len(result.q_grid),2))
    ax[0].set_yticklabels(result.q_grid[::2])
    # Create labels
    values  = np.unique(result.l_p.ravel())
    colors  = [ im_1.cmap(im_1.norm(value)) for value in values]
    labels  = {0:"Cancel / Do nothing",1:"Join Offer Queue"}
    patches = [ mpatches.Patch(color=colors[i], label=labels[values[i]] ) for i in range(len(values)) ]
    ax[0].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    
    # Plot buy decisions
    im_2 = ax[1].imshow(result.l_m, aspect='auto')
    ax[1].set_xlabel('Trading time')
    ax[1].set_ylabel('Inventory/Position')
    ax[1].set_title('Bid Side')
    ax[1].set_yticks(range(0,len(result.q_grid),2))
    ax[1].set_yticklabels(result.q_grid[::2])
    # Create labels
    values  = np.unique(result.l_m.ravel())
    colors  = [ im_2.cmap(im_2.norm(value)) for value in values]
    labels  = {0:"Cancel / Do nothing",1:"Join Bid Queue"}
    patches = [ mpatches.Patch(color=colors[i], label=labels[values[i]] ) for i in range(len(values)) ]
    ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    

    # Plot decision boundaries
    im_3 = ax[2].imshow(l, aspect='auto')
    
    # Configure horizontal axis
    ax[2].set_xticks([i for i in range(0,result.N_steps)[::50]] + [result.N_steps])
    ax[2].set_xticklabels([np.round(result.t_grid[i],2) for i in range(0,result.N_steps)[::50]] + [np.round(result.t_grid[result.N_steps-1],2)])
    
    # Configure vertical axis
    ax[2].set_yticks(range(0,len(result.q_grid),2))
    ax[2].set_yticklabels(result.q_grid[::2])
    #ax.set_yticklabels(result.q_grid)
    
    # Create labels
    values  = np.unique(l.ravel())
    colors  = [ im_3.cmap(im_3.norm(value)) for value in values]
    labels  = {0:"Cancel / Do nothing",1:"Join Bid Queue Only",2:"Join Offer Queue Only",3:"Join Both Queues"}
    patches = [ mpatches.Patch(color=colors[i], label=labels[values[i]] ) for i in range(len(values)) ]
    ax[2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    # Create axis labels
    ax[2].set_xlabel('Trading time')
    ax[2].set_ylabel('Inventory/Position')
    ax[2].set_title('Optimal decisions')
    
    plt.tight_layout()
    plt.show()