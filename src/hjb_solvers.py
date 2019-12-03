# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy

class MMATT_Model_Output:
    
    def __init__(self,l_p,l_m,h,q_lookup,q_grid,t_grid):
        
        self.m_l_p = l_m
        self.m_l_m = l_p
        self.m_h   = h
        
        self.m_q_lookup = q_lookup
        self.m_q_grid   = q_grid
        self.m_t_grid   = t_grid
        
    
    @property
    def h(self):
        """
        Returns a copy of the value function h at each (q,t) node.
        """
        return deepcopy(self.m_h)
    
    @property
    def l_p(self):
        """
        Returns a copy of the decision variables l^{+} at each (q,t) node
        """
        return deepcopy(self.m_l_p)
 
    @property
    def l_m(self):
        """
        Returns a copy of the decision variables l^{-} at each (q,t) node
        """
        return deepcopy(self.m_l_m)
    
    def get_l_plus(self,q,t):
        
        t_idx = filter(lambda x: x>=t, self.m_t_grid)[0]
        q_idx = self.m_q_lookup[q]
        
        return self.m_l_p[t_idx,q_idx]
        
        
    def get_l_minus(self,q,t):
        
        t_idx = filter(lambda x: x>=t, self.m_t_grid)[0]        
        q_idx = self.m_q_lookup[q]
        
        return self.m_l_p[t_idx,q_idx]
        

def solve_l(T,lambda_m,lambda_p,delta,alpha,phi,q_min,q_max,N_steps):

    if(not isinstance(T,(float,int))):
        raise TypeError('T has to be type of <float>')
    
    if(not isinstance(lambda_m,(float,int))):
        raise TypeError('lambda_m has to be type of <float> or <int>')
    
    if(lambda_m<0.0):
        raise ValueError('lambda_m has to be positive!')
    
    if(not isinstance(lambda_p,(float,int))):
        raise TypeError('lambda_p has to be type of <float> or <int>')    
    
    if(lambda_p<=0.0):
        raise ValueError('lambda_p has to be positive!')
    
    if(not isinstance(delta,float)):
        raise TypeError('delta has to be type of <float>')    

    if(delta<=0.0):
        raise ValueError('delta has to be positive!')

    if(not isinstance(alpha,float)):
        raise TypeError('alpha has to be type of <float>') 
    
    if(alpha<0.0):
        raise ValueError('alpha has to be positive!')
    
    if(not isinstance(phi,float)):
        raise TypeError('phi has to be type of <float>') 

    if(phi<0.0):
        raise ValueError('phi has to be positive!')

    if(not isinstance(q_min,int)):
        raise TypeError('q_min has to be type of <min>') 
    
    if(not isinstance(q_max,int)):
        raise TypeError('q_max has to be type of <min>')
        
    if(q_max<q_min):
        raise ValueError('q_max cannot be < q_min')
    
    if(not isinstance(N_steps,int)):
        raise TypeError('N_steps has to be type of <min>')
    
    if(N_steps<=0):
        raise ValueError('N_steps has to be positive')
    
    n        = q_max - q_min + 1 
    q_map    = dict( (q, i) for i, q in enumerate(range(q_max, q_min-1, -1)))
    q_lookup = lambda q : q_map[q] 
    
    # Time step for finite difference
    dt     = T / N_steps
    
    # Matrix for function h
    h       = np.zeros((n,N_steps))
    
    # Matrix for l^{+}
    l_p     = np.zeros((n,N_steps))
    
    # Matrix for l^{-}
    l_m     = np.zeros((n,N_steps))   
    
    # Terminal condition
    h[:,-1] = np.array([ - alpha*q**2 for q in range(q_max, q_min-1, -1)])    
    
    # Inventory levels
    q_grid  = [q for q in range(q_max, q_min-1, -1)]
    
    # Time points
    t_grid  = np.zeros(N_steps)
    t_grid[-1] = 1
    
    # Solve DPE/PDE using finite differences
    for idx in range(N_steps-1, 0, -1):
        
        # Posting indicators for current time step
        l_m_i = np.zeros(n)
        l_p_i = np.zeros(n)
        
        # h for previous time stemp
        h_i_p = np.zeros(n)
        
        # h for current time step
        h_i   = h[:,idx]
        
        for q in q_grid:
            
            l_p_ = 0.0
            l_m_ = 0.0
            
            if q == q_max: # 1_{q<q_max} == 0
                
                # Determine optimal l^{+}
                dh_p = 0.5*delta + h_i[q_lookup(q-1)] - h_i[q_lookup(q)]
                if(dh_p > 0): l_p_ = 1.0
                
                # Compute h(t-dt) using backward Euler
                h_i_p[q_lookup(q)] = h_i[q_lookup(q)] + ( -phi*(q**2)
                                                      + lambda_p*l_p_*dh_p )*dt
                l_m_i[q_lookup(q)] = 0.0
                l_p_i[q_lookup(q)] = l_p_
            
            elif q == q_min: # 1_{q>q_min} == 0
                
                # Determine optimal l^{-}
                dh_m = 0.5*delta + h_i[q_lookup(q+1)] - h_i[q_lookup(q)]
                if(dh_m > 0): l_m_ = 1.0

                # Compute h(t-dt) using backward Euler
                h_i_p[q_lookup(q)] = h_i[q_lookup(q)] + ( -phi*(q**2)
                                                      + lambda_m*l_m_*dh_m )*dt
                l_m_i[q_lookup(q)] = l_m_
                l_p_i[q_lookup(q)] = 0.0      
                
            else: # 1_{q>q_min} == 1 and 1_{q>q_min} == 1
                
                # Determine optimal l^{+}
                dh_p = 0.5*delta + h_i[q_lookup(q-1)] - h_i[q_lookup(q)]
                if(dh_p > 0): l_p_ = 1.0     
                
                # Determine optimal l^{-}
                dh_m = 0.5*delta + h_i[q_lookup(q+1)] - h_i[q_lookup(q)]
                if(dh_m > 0): l_m_ = 1.0                
                
                # Compute h(t-dt) using backward Euler
                h_i_p[q_lookup(q)] = h_i[q_lookup(q)] + ( -phi*(q**2)
                                                      + lambda_p*l_p_*dh_p 
                                                      + lambda_m*l_m_*dh_m )*dt
                l_m_i[q_lookup(q)] = l_m_
                l_p_i[q_lookup(q)] = l_p_                
                
        h[:,idx-1]   = h_i_p
        l_p[:,idx-1] = l_p_i
        l_m[:,idx-1] = l_m_i
        
        t_grid[idx] = t_grid[idx-1] - dt
    
    out = MMATT_Model_Output(l_p,l_m,h,q_lookup,q_grid,t_grid)
        
    return out


        
