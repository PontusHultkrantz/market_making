#%%
from src.hjb_solvers import MMATT_Model_Parameters
from src.hjb_solvers import MMATT_Finite_Difference_Solver
from src.plot_utils  import plot_decision_boundaries

lambda_m = 200     # Order-flow at the bid
lambda_p = 200     # Order-flow at the offer
alpha    = 0.001   # Terminal inventory penalty
phi      = 0.001   # Running inventory penalty
q_min    = -20     # Maximum short inventory
q_max    =  20     # Maximum long inventory
delta    = 0.01    # Average edge level per trade

# Encapsulate parameters
params = MMATT_Model_Parameters(lambda_m,lambda_p,delta,phi,alpha,q_min,q_max)

# Solve
result = MMATT_Finite_Difference_Solver.solve(params,500)

# Plot the decision boundaries
plot_decision_boundaries(result,(4.5,10))








