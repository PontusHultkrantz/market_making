
from src.hjb_solvers import MMATT_Model_Parameters
from src.hjb_solvers import MMATT_Finite_Difference_Solver
from src.plot_utils  import plot_decision_boundaries

lambda_m = 20      # Order-flow at the bid
lambda_p = 20      # Order-flow at the offer
alpha    = 0.001   # Terminal inventory penalty
phi      = 0.0001  # Running inventory penalty
q_min    = -20     # Maximum short inventory
q_max    =  20     # Maximum long inventory
delta    = 0.01    # Average edge level per trade

# Encapsulate parameters
params = MMATT_Model_Parameters(lambda_m,lambda_p,delta,phi,alpha,q_min,q_max)

# Solve
result = MMATT_Finite_Difference_Solver.solve(params)

# Plot the decision boundaries
#plot_decision_boundaries(result,(6,4))
import numpy as np

l = np.zeros_like(result.l_p)
l[result.l_m==1] = 1
l[result.l_p==1] = 2
l[(result.l_p==1) & (result.l_m==1)] = 3
    

#%%
import numpy as np

lambda_ms = np.arange(10,300,10)
lambda_ps = np.arange(10,300,10)

lambdas = []
for i in range(0,len(lambda_ms)):
    for j in range(0,len(lambda_ps)):
        lambdas.append((lambda_ms[i],lambda_ps[j]))
        
#%%
        
solutions = {}
for lam in lambdas:
    
    lambda_m = lam[0]
    lambda_p = lam[1]
    
    # Encapsulate parameters 
    params = MMATT_Model_Parameters(lambda_m, lambda_p, delta, phi, alpha, q_min, q_max)
    
    # Solve
    result = MMATT_Finite_Difference_Solver.solve(params)
    
    print(lam)
    solutions.update({lam:result})
    
#%%
    
# Plot the decision boundaries
result = solutions[(10,10)]
plot_decision_boundaries(result.m_l_m,result.m_l_p,result.m_q_grid,(6,4))

