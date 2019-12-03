# -*- coding: utf-8 -*-
import numpy as np
import sys

log_float_max   = np.log(sys.float_info.max)
log_float_min   = np.log(sys.float_info.min)

s_log_float_max = log_float_max/float(1000)
s_log_float_min = log_float_min/float(1000)


def mm1_solve_h(T,lamda_m,lamda_p,kappa_m,kappa_p,alpha,phi,q_min,q_max,q_trg,N_steps):

    n      = q_max - q_min + 1 
    q_map  = dict( (q, i) for i, q in enumerate(range(q_max, q_min-1, -1)))
    lookup = lambda q : q_map[q] 
    
    C1 = lamda_p / (np.e * kappa_p)
    C2 = lamda_m / (np.e * kappa_m)
    
    dt  = T / N_steps
    
    h       = np.zeros((n,N_steps))
    h[:,-1] = np.array([ -alpha * (i-q_trg)**2 for i in range(q_max, q_min-1, -1)])    
    
    for idx in range(N_steps-1, 0, -1):
    
        h_i_prev = np.zeros(n)
        h_i      = h[:,idx]
        
        for q in range(q_max, q_min-1, -1):
            
            if q == q_max:
            
                exp_arg_1 = kappa_p*( h_i[lookup(q-1)] - h_i[lookup(q)] )
                exp_arg_1 = max([min([exp_arg_1, s_log_float_max]), s_log_float_min])
                
                h_i_prev[lookup(q)] = h_i[lookup(q)] + (-(phi*(q - q_trg)**2)
                                                     + C1 * np.exp(exp_arg_1))*dt
            
            elif q == q_min:
                
                exp_arg_2 = kappa_m*( h_i[lookup(q+1)] - h_i[lookup(q)] )
                exp_arg_2 = max([min([exp_arg_2, s_log_float_max]), s_log_float_min])
                
                h_i_prev[lookup(q)] = h_i[lookup(q)] + (-(phi*(q-q_trg)**2)
                                                     + C2 * np.exp(exp_arg_2))*dt
            
            else:
                
                exp_arg_1 = kappa_p*( h_i[lookup(q-1)] - h_i[lookup(q)] )
                exp_arg_2 = kappa_m*( h_i[lookup(q+1)] - h_i[lookup(q)] )
               
                exp_arg_1 = max([min([exp_arg_1, s_log_float_max]), s_log_float_min])
                exp_arg_2 = max([min([exp_arg_2, s_log_float_max]), s_log_float_min])
    
                h_i_prev[lookup(q)] = h_i[lookup(q)] + (-(phi*(q - q_trg)**2) 
                                                     + C1 * np.exp(exp_arg_1) 
                                                     + C2 * np.exp(exp_arg_2))*dt
   
        h[:,idx-1] = h_i_prev
        
    d_p = {}
    for q in range(q_max, q_min, -1):
        d_p[q] = (1.0/kappa_p) - (h[lookup(q-1)] - h[lookup(q)])  
        
    d_m = {}   
    for q in range(q_min, q_max ):
        d_m[q] = (1.0/kappa_m) - (h[lookup(q+1)] - h[lookup(q)]) 

    return h,d_m,d_p

def mm2_solve_h(T,lamda_m,lamda_p,delta,alpha,phi,q_min,q_max,N_steps):

    n      = q_max - q_min + 1 
    q_map  = dict( (q, i) for i, q in enumerate(range(q_max, q_min-1, -1)))
    lookup = lambda q : q_map[q] 
    
    dt     = T / N_steps
    
    h       = np.zeros((n,N_steps))
    l_p     = np.zeros((n,N_steps))
    l_m     = np.zeros((n,N_steps))   
    
    # Terminal condition
    h[:,-1] = np.array([ - alpha*q**2 for q in range(q_max, q_min-1, -1)])    
    
    q_grid  = [q for q in range(q_max, q_min-1, -1)]
     
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
            
            if q == q_max:
            
                # Compute (+) indicator argument
                h_diff_p = 0.5*delta + h_i[lookup(q-1)] - h_i[lookup(q)]
                
                # Determine optimal l_p
                if(h_diff_p > 0):
                    l_p_ = 1.0
                
                h_i_prev[lookup(q)] = h_i[lookup(q)] - (phi*(q**2))*dt + lamda_p*l_p_*h_diff_p*dt
                l_m_i[lookup(q)]    = 0.0
                l_p_i[lookup(q)]    = l_p_
                
            elif q == q_min:
                
                # Compute (-) indicator argument
                h_diff_m = 0.5*delta + h_i[lookup(q+1)] - h_i[lookup(q)]
                # Determine optimal l_m
                
                if(h_diff_m > 0):
                    l_m_ = 1.0

                h_i_p[lookup(q)] = h_i[lookup(q)] - (phi*(q**2))*dt + lamda_m*l_m_*h_diff_m*dt
                l_m_i[lookup(q)] = l_m_
                l_p_i[lookup(q)] = 0.0      
                
            else:
                
                # Compute (+) indicator argument
                h_diff_p = 0.5*delta + h_i[lookup(q-1)] - h_i[lookup(q)]
                
                # Determine optimal l_p
                if(h_diff_p > 0):
                    l_p_ = 1.0     
                
                # Compute (-) indicator argument
                h_diff_m = 0.5*delta + h_i[lookup(q+1)] - h_i[lookup(q)]
                
                # Determine optimal l_m
                if(h_diff_m > 0):
                    l_m_ = 1.0                
                
                h_i_p[lookup(q)] = h_i[lookup(q)] - (phi*(q**2))*dt + lamda_p*l_p_*h_diff_p*dt  + lamda_m*l_m_*h_diff_m*dt
                l_m_i[lookup(q)] = l_m_
                l_p_i[lookup(q)] = l_p_                
                
        h[:,idx-1]   = h_i_p
        l_p[:,idx-1] = l_p_i
        l_m[:,idx-1] = l_m_i
        

    return h,l_m,l_p,q_grid


        
