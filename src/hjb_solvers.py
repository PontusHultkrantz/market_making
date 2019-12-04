# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy

__author__      = "Juha-Samuli Hellen"

class MMATT_Model_Parameters:
    """
    Object for encapsulating Market Making At-The-Touch (MMATT) model parameters.
    
    Member variables are implemented as properties to replicate C-style
    accessors to prevent/reduce risk of accidental mutation.
    
    """
    def __init__(self,lambda_m,lambda_p,delta,phi,alpha,q_min,q_max):
        
        if(not isinstance(lambda_m,(float,int,np.int32,np.int64))):
            raise TypeError(f'lambda_m has to be type of <float> or <int>, not {type(lambda_m)}')

        if(not isinstance(lambda_p,(float,int,np.int32,np.int64))):
            raise TypeError(f'lambda_p has to be type of <float> or <int>, not {type(lambda_m)}')       

        if(not isinstance(delta,(float,int,np.int32,np.int64))):
            raise TypeError('delta has to be type of <float> or <int>')

        if(not isinstance(phi,(float,int,np.int32,np.int64))):
            raise TypeError('phi has to be type of <float> or <int>')

        if(not isinstance(alpha,(float,int,np.int32,np.int64))):
            raise TypeError('alpha has to be type of <float> or <int>')            

        if(not isinstance(q_min,(int,np.int32,np.int64))):
            raise TypeError('q_min has to be type of <int>')  
            
        if(not isinstance(q_max,(int,np.int32,np.int64))):
            raise TypeError('q_max has to be type of <int>')  
        
        if(q_max <= q_min):
            raise ValueError('q_max has to be larger than q_min!')
        
        self.m_lambda_m = lambda_m # Order-flow at be bid
        self.m_lambda_p = lambda_p # Order flow at the offer
        
        self.m_delta = delta # average "edge" with respect to mid-price
        self.m_phi = phi   # running inventory penalty 
        self.m_alpha = alpha # terminal inventory penalty
        
        self.m_q_min = q_min 
        self.m_q_max = q_max
    
    @property
    def lambda_m(self):
        """
        Return copy of the order flow parameter at the bid
        """
        return deepcopy(self.m_lambda_m)
    
    @property
    def lambda_p(self):
        """
        Return copy of the order flow parameter at the ask
        """        
        return deepcopy(self.m_lambda_p)

    @property
    def delta(self):
        """
        Return copy of the average "edge".
        """        
        return deepcopy(self.m_delta)

    @property
    def phi(self):
        """
        Return copy of the running inventory penalty.
        """        
        return deepcopy(self.m_phi)

    @property
    def alpha(self):
        """
        Return copy of the terminal inventory penalty.
        """        
        return deepcopy(self.m_alpha)

    @property
    def q_min(self):
        """
        Return copy of the minimum inventory level.
        """        
        return deepcopy(self.m_q_min)    

    @property
    def q_max(self):
        """
        Return copy of the maximum inventory level.
        """        
        return deepcopy(self.m_q_max) 
    
    
    
class MMATT_Model_Output:
    """
    Object for encapsulating Market Making At-The-Touch (MMATT) output.
    
    Member variables are implemented as properties to replicate C-style
    accessors to prevent/reduce risk of accidental mutation.
    
    """    
    def __init__(self,l_p,l_m,h,q_lookup,q_grid,t_grid,N_steps):
        
        self.m_l_p = l_p
        self.m_l_m = l_m
        self.m_h = h
        
        self.m_q_lookup = q_lookup
        self.m_q_grid = q_grid
        self.m_t_grid = t_grid
        self.m_N_steps = N_steps
        
    
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
    
    @property
    def q_grid(self):
        """

        """
        return deepcopy(self.m_q_grid)
        
    @property
    def t_grid(self):
        """

        """
        return deepcopy(self.m_t_grid)
        
        
    @property
    def N_steps(self):
        
        return deepcopy(self.m_N_steps)
    
    def get_l_plus(self,q,t):
        """
        Returns SELL decision given inventory q and time remaining till end of
        trading day t.
        """
        t_idx = filter(lambda x: x>=t, self.m_t_grid)[0]
        
        if(q in self.m_q_lookup):
            q_idx = self.m_q_lookup[q]
        else:
            raise KeyError(f'Inventory level {q} exceeds inventory grid used in pre-computation.')
            
        return self.m_l_p[t_idx,q_idx]
        
        
    def get_l_minus(self,q,t):
        """
        Returns BUY decision given inventory q and time remaining till end of
        trading day t.
        """        
        t_idx = filter(lambda x: x>=t, self.m_t_grid)[0]        
        if(q in self.m_q_lookup):
            q_idx = self.m_q_lookup[q]
        else:
            raise KeyError(f'Inventory level {q} exceeds inventory grid used in pre-computation.')
        
        return self.m_l_m[t_idx,q_idx]
       
_EFF_ZERO = 1E-6

class MMATT_Finite_Difference_Solver:
    """
    
    """

    @staticmethod
    def solve(params, N_steps = 500):
        """
        Solves the optimal BUY and SELL decisions of the trading algorithm 
        using backward Euler finite difference scheme.
        """        
        
        n = params.q_max - params.q_min + 1 
        q_grid = [q for q in range(params.q_max, params.q_min-1, -1)]
        q_map = dict( (q, i) for i, q in enumerate(q_grid))
        q_lookup = lambda q : q_map[q] 
        
        # Time step for finite difference
        dt = 1.0 / N_steps
        
        # Matrix for function h
        h = np.zeros((n,N_steps))
        
        # Matrix for l^{+}
        l_p = np.zeros((n,N_steps))
        
        # Matrix for l^{-}
        l_m = np.zeros((n,N_steps))   
        
        # Terminal condition
        h[:,-1] = np.array([ -params.alpha*q**2 for q in q_grid])    
        
        # Time points
        t_grid = np.zeros(N_steps)
        t_grid[-1] = 1
        
        # Solve DPE/PDE using finite differences
        for idx in range(N_steps-1, 0, -1):
            
            t_grid[idx-1] = t_grid[idx] - dt
            
            # Posting indicators for current time step
            l_m_i = np.zeros(n)
            l_p_i = np.zeros(n)
            
            # h for previous time stemp
            h_i_p = np.zeros(n)
            
            # h for current time step
            h_i = h[:,idx]
            
            for q in q_grid:
                
                l_p_ = 0.0
                l_m_ = 0.0
                
                if q == params.q_max: # 1_{q<q_max} == 0
                    
                    # Determine optimal l^{+}
                    dh_p = 0.5*params.delta + h_i[q_lookup(q-1)] - h_i[q_lookup(q)]
                    if(dh_p > _EFF_ZERO): l_p_ = 1.0
                    
                    # Compute h(t-dt) using backward Euler
                    h_i_p[q_lookup(q)] = h_i[q_lookup(q)] + ( -params.phi*(q**2)
                                                          + params.lambda_p*l_p_*dh_p )*dt
                    l_m_i[q_lookup(q)] = 0.0
                    l_p_i[q_lookup(q)] = l_p_
                
                elif q == params.q_min: # 1_{q>q_min} == 0
                    
                    # Determine optimal l^{-}
                    dh_m = 0.5*params.delta + h_i[q_lookup(q+1)] - h_i[q_lookup(q)]
                    if(dh_m > _EFF_ZERO): l_m_ = 1.0
    
                    # Compute h(t-dt) using backward Euler
                    h_i_p[q_lookup(q)] = h_i[q_lookup(q)] + ( -params.phi*(q**2)
                                                          + params.lambda_m*l_m_*dh_m )*dt
                    l_m_i[q_lookup(q)] = l_m_
                    l_p_i[q_lookup(q)] = 0.0      
                    
                else: # 1_{q>q_min} == 1 and 1_{q>q_min} == 1
                    
                    # Determine optimal l^{+}
                    dh_p = 0.5*params.delta + h_i[q_lookup(q-1)] - h_i[q_lookup(q)]
                    if(dh_p > _EFF_ZERO): l_p_ = 1.0     
                    
                    # Determine optimal l^{-}
                    dh_m = 0.5*params.delta + h_i[q_lookup(q+1)] - h_i[q_lookup(q)]
                    if(dh_m > _EFF_ZERO): l_m_ = 1.0                
                     
                    # Compute h(t-dt) using backward Euler
                    h_i_p[q_lookup(q)] = h_i[q_lookup(q)] + ( -params.phi*(q**2)
                                                          + params.lambda_p*l_p_*dh_p 
                                                          + params.lambda_m*l_m_*dh_m )*dt
                    l_m_i[q_lookup(q)] = l_m_
                    l_p_i[q_lookup(q)] = l_p_                
                    
            h[:,idx-1] = h_i_p
            l_p[:,idx-1] = l_p_i
            l_m[:,idx-1] = l_m_i
            
            
        
        out = MMATT_Model_Output(l_p, l_m, h, q_lookup, q_grid, t_grid, N_steps)
            
        return out


        
