"""
Estimates income process parameters
"""
import numpy as np
import pandas as pd
from pathlib import Path
from model_moments import parameter_estimation
                            
#load empirical data 
moments_dir = Path("EmpiricalMoments") 
level_moments       = np.genfromtxt(Path(moments_dir,"levels_moments_v2.1_by1950_c_vector.csv"), delimiter=',')
level_omega         = np.genfromtxt(Path(moments_dir,"levels_moments_v2.1_by1950_omega.csv"), delimiter=',')
difference_moments  = np.genfromtxt(Path(moments_dir,"changes_moments_v2.1_by1950_c_vector.csv"), delimiter=',')
difference_omega    = np.genfromtxt(Path(moments_dir,"changes_moments_v2.1_by1950_omega.csv"), delimiter=',')

###############################################################################
# First estimate the standard model 
############################################################################### 
# set up initial guess and bounds
init_params = np.array([0.005, #var perm
                        0.03,  #var tran
                        0.3,  #theta
                        1.0, #rho
                        0.069]) #var_init  
bounds     = [(0.0,1.0)]*5 
# First levels
optimize_index = np.array([0,1,2,-1,4]) # exclude rho from estimation
estimates_levels_standard = parameter_estimation(level_moments, level_omega, init_params, \
                         model="Standard", levels_or_differences="levels", MD_weight_type="diagonal", \
                         optimize_index=optimize_index, bounds=bounds)
# Then differences
optimize_index = np.array([0,1,2,-1,-1])  #fix var_init - not identified with differences - and exclude rho
estimates_difference_standard = parameter_estimation(difference_moments, difference_omega, init_params, \
                         model="Standard", levels_or_differences="differences", MD_weight_type="diagonal", \
                         optimize_index=optimize_index, bounds=bounds)
   
###############################################################################
# Now estimate CHT model
############################################################################### 
# set up initial guess and bounds
bounds          = [(0.0,1.0), # var_perm
                   (0.0,1.0), # var_tran
                   (0.2,3.0), # omega is only one likely to be above 1.0
                   (0.0,1.0), # bonus
                   (0.0,0.2), # rho is less than 0.2
                   (0.0,1.0)] # var_init
init_params      = np.array([0.005,  #var perm
                             0.05, #var tran
                             0.5,                #omega
                             0.4,#0.3                #bonus
                             0.0, # rho perm
                             0.069]) # var init 
# First do levels
optimize_index = np.array([0,1,2,3,-1,5]) #exclude rho from estimation
estimates_levels_CHT = parameter_estimation(level_moments, level_omega, init_params, \
                         model="CHT", levels_or_differences="levels", MD_weight_type="diagonal", \
                         optimize_index=optimize_index, bounds=bounds)
# Then differences
optimize_index =  np.array([0,1,2,3,-1,-1])
estimates_differences_CHT = parameter_estimation(difference_moments, difference_omega, init_params, \
                         model="CHT", levels_or_differences="differences", MD_weight_type="diagonal", \
                         optimize_index=optimize_index, bounds=bounds)
###############################################################################
# Print all estimates
###############################################################################    
all_estimates = pd.DataFrame(([np.round(np.concatenate((estimates_levels_standard[0:3],[0],estimates_levels_standard[3:])),3),\
 np.round(np.concatenate((estimates_difference_standard[0:3],[0],estimates_difference_standard[3:])),3),\
 np.round(estimates_levels_CHT,3),\
 np.round(estimates_differences_CHT,3)]), \
['Standard, levels', 'Standard, differences', 'CHT, levels', 'CHT, differences'],\
['var_perm','var_tran','theta or omega', 'b', 'rho', 'var_init'])
print(all_estimates)



