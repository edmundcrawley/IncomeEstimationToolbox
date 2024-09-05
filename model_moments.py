# Minimum Distance calculations: model moments and optimization

import numpy as np
from scipy.optimize import minimize
from tools import vech, inv_vech

###############################################################################
# CHT levels covariance matricies
###############################################################################
def cov_bonusshk_levels(var_bonus, T):
  '''
  Calculates the covariance matrix for a bonus shock with NO persistence
  '''
  # Set up covariance matrix, initialized to zero
  cov_y  = np.zeros((T,T)) 
  for j in range(T):
    cov_y[j,j] = var_bonus
  cov_y_vec=vech(cov_y)
  return cov_y_vec

def cov_passingshk_levels(var, rho, T):
  '''
  Calculates the levels covariance matrix for the discrete-time version of passing shocks
  '''
  # Set up covariance matrix, initialized to zero
  cov_y  = np.zeros((T,T)) 
  for i in range(T):
      for j in range(i+1):
          cov_y[i,j] = var*rho**(i-j)
  cov_y_vec=vech(cov_y)
  return cov_y_vec


def cov_AR1_levels(var_AR1, rho_AR1, T):
  '''
  Calculates the levels covariance matrix for AR(1) shocks that arrive in discrete,
  annual periods. This is the persistent shock in the standard model.
  '''
  # Create covariance matrix
  cov_y  = np.zeros((T,T)) 
  for i in range(T):
      for j in range(i+1):
          cov_y[i,j] += np.sum(var_AR1*rho_AR1**(2*np.array(range(j+1))))*rho_AR1**(i-j)
  cov_y_vec=vech(cov_y)
  return cov_y_vec

    
def cov_CHT_model_levels(params,T):
    '''
    Crawley, Holm, Tretvoll model is a composite of there shock types (and in initial variance)
    '''
    [var_perm, var_tran, rho, bonus, rho_persistent, var_init] = params
    persistent_inc_cov = cov_AR1_levels(var_perm,rho_persistent,T)
    bonus_inc_cov = cov_bonusshk_levels(var_tran*bonus,T)
    passing_inc_cov = cov_passingshk_levels(var_tran*(1-bonus),rho,T)
    init_perm_inc_cov = vech(var_init*np.ones((T,T)))
       
    cov_CHT_levels = persistent_inc_cov + bonus_inc_cov + passing_inc_cov + init_perm_inc_cov
    return cov_CHT_levels
###############################################################################
# Standard model covariance matricies
###############################################################################
    
def cov_MA1_levels(var_tran, theta, T):
  '''
  Calculates the levels covariance matrix for MA(1) shocks that arrive in discrete
  annual periods
  '''
  shk_var = var_tran/(1.0+theta**2)
  # Create covariance matrix
  cov_y  = np.zeros((T,T)) 
  for i in range(T):
      cov_y[i, i] = shk_var*(1.0 + theta**2)
  for i in range(T-1):
      cov_y[i,i+1] = shk_var*theta
      cov_y[i+1,i] = cov_y[i,i+1]
  cov_y_vec = vech(cov_y)
  return cov_y_vec 

     
def cov_standard_model_levels(params,T):
    '''
    The covariance of the standard model is composed of persistent shocks and an MA(1) (+ initial variance)
    '''
    [var_perm, var_tran, theta, rho, var_init] = params
    perm_inc_cov = cov_AR1_levels(var_perm, rho,T)
    MA1_inc_cov = cov_MA1_levels(var_tran,theta,T)
    cov_standard_model = var_init + perm_inc_cov + MA1_inc_cov
    return cov_standard_model

def convert_cov_to_differences(levels_cov_vec):
    '''
    Given a levels covariance matrix, this function calculates the difference covariance matrix
    '''
    levels_cov = inv_vech(levels_cov_vec)
    difference_cov = levels_cov[1:,1:] - levels_cov[1:,:-1] - levels_cov[:-1,1:] + levels_cov[:-1,:-1]
    difference_cov_vec = vech(difference_cov)
    return difference_cov_vec


def model_covariance(params, T, model="CHT", levels_or_differences = "levels", every_2nd_year=False):
    '''
    Calculates the model covariance moments in levels or differences
    Can handle covariance matrices in which data is available every two years (e.g. PSID)
    '''
    if (levels_or_differences=="levels"):
        T_simulate = T
    else:
        T_simulate = T+1 # Need to calculate one more year, so when we take the difference we have T periods
    if(every_2nd_year):
        num_periods = T_simulate
        T_simulate=2*T_simulate

    # Income models
    if (model=="CHT"): 
        model_cov = cov_CHT_model_levels(params,T_simulate)
    if (model=="Standard"):
        model_cov = cov_standard_model_levels(params,T_simulate)
        
    if(every_2nd_year):
        full_model_cov = inv_vech(model_cov)
        model_cov = vech(full_model_cov[np.array(range(num_periods))*2+1,:][:,np.array(range(num_periods))*2+1])
        
    if levels_or_differences == "differences":
        model_cov = convert_cov_to_differences(model_cov)

    return model_cov

###############################################################################
def parameter_estimation(empirical_moments, Omega, init_params, \
                         model="CHT", levels_or_differences="levels", MD_weight_type="diagonal", \
                         every_2nd_year=False, optimize_index=None, \
                         bounds=None):
    '''
    Estimates the parameters of the income model
    
    Parameters
    ----------
    empirical_moments : np.array
        A vector of moments created by vech(cov_matrix)
    Omega : np.array
        A two dimensional array containing the covariance matrix of the emprirical moments (which confusingly are themselves covariances)
    init_params : np.array
        For CHT model, [var_perm, var_tran, omega, bonus, rho, var_init] 
            var_perm = persistent shock variance
            var_tran = variance of transitory component of income (bonus and passing)
            omega = decay parameter for passing shock
            b = fraction of transitory variance that is bonus type
            rho = decay parameter for persistent shock (=0 if permanent)
            var_init = initial cross-sectional variance
        for standard model, [var_perm, var_tran, theta, rho, var_init]
            var_perm = persistent shock variance
            var_tran = variance of transitory component of income (total MA(1) variance)
            theta = MA(1) parameter
            rho = AR(1) parameter for persistent shock (=1 if permanent)
            var_init = initial cross-sectional variance
    model : string, optional
        "CHT" or "Standard". The default is "CHT"
    levels_or_differences : string, optional
        "levels" or "differences". The default is "levels".
    MD_weight_type : string, optional
        "optimal", "diagonal", or "eye". The default is "diagonal".
    every_2nd_year : bool, optional
        Option to allow observations every 2nd year (e.g. PSID). The default is False.
    optimize_index : np.array, optional
        An array of length init_params. Each entry is equal to its position, unless the parmeter in that
        position is to be fixed at the initial value. e.g [0,1,2,3,-1,-1] for the 
        Standard model will fix rho and var_init to their initial values and estimate the first three parameters
        . The default is None (estimate all parameters except init_var for difference estimations).
    bounds : list of tuples, of length init_params, optional
        e.g [(0,1),(0,1),(0,1),(0,1),(0,1)] bounds all parameters between 0 and 1. The default is None.

    Returns
    -------
    np.array, np.array
        parameter estimates along with standard errors.

    '''
  #fix certain parameters if required
    if (optimize_index is None):
        optimize_index = np.array(range(len(init_params)))
        if levels_or_differences=="differences":
            optimize_index[-1] = -1 # fix init_var because difference moments do not identify this parameter

    optimize_params = init_params[np.equal(optimize_index,range(len(optimize_index)))] # parameters to be optimized are only those that have their own index in "optimize_index"
    fixed_params      = init_params[np.equal(optimize_index,-1)] # parameters to be fixed have -1 as an entry in the optimize_index
    if (bounds is not None): #cut out bounds for fixed parameters
          all_bounds = bounds
          bounds = []
          for i in range(len(all_bounds)):
              if (optimize_index[i]==i):
                  bounds += [all_bounds[i]]
    T = inv_vech(empirical_moments).shape[0]
    

    def model_cov_limited_params(optimize_params, T, optimize_index, fixed_params, model, levels_or_differences,every_2nd_year=False):
        fixed_index = np.equal(optimize_index,-1) # fixed parameters are indicated by -1 in optimize_index
        recover_index = np.array(range(len(optimize_index)))[np.equal(optimize_index,range(len(optimize_index)))] # index of each optimizing parameter in the original init_params vector
        params = np.zeros(len(optimize_index))
        params[recover_index] = optimize_params # Sets optimizing parameters equal to their entered value
        params[fixed_index]   = fixed_params # Sets fixed parameters equal to their entered value
        params[np.logical_not(fixed_index)] = params[optimize_index[np.logical_not(fixed_index)]] # Sets other parameters equal to the optimizing parameter chosen
        model_cov = model_covariance(params, T, model, levels_or_differences, every_2nd_year)
        return model_cov

    def objectiveFun(optimize_params, T, empirical_cov, weight_matrix, optimize_index, fixed_params,model, levels_or_differences,every_2nd_year):
        model_cov = model_cov_limited_params(optimize_params, T, optimize_index, fixed_params,model,levels_or_differences, every_2nd_year)
        distance = np.dot(np.dot((model_cov-empirical_cov), weight_matrix), (model_cov-empirical_cov))
        return distance
    
    # Define the weight matrix as Equal Weight Minimum Distance
    if MD_weight_type=="diagonal":
        weight_matrix = np.diag(np.diag(Omega)**(-1))
    elif MD_weight_type=="optimal":
        weight_matrix = np.linalg.inv(Omega)
    elif MD_weight_type=="eye":
        weight_matrix = np.eye(Omega.shape[0])*np.mean(np.diag(Omega)**(-1))

    # Do minimization
    solved_objective = minimize(objectiveFun, optimize_params, args=(T, empirical_moments, weight_matrix, optimize_index, fixed_params,model,levels_or_differences,every_2nd_year), method='L-BFGS-B', bounds=bounds, options= {'disp': 1})
    solved_params = solved_objective.x
    
    # Create output
    fixed_index = np.equal(optimize_index,-1) # fixed parameters are indicated by -1 in optimize_index
    recover_index = np.array(range(len(optimize_index)))[np.equal(optimize_index,range(len(optimize_index)))] # index of each optimizing parameter in the original init_params vector
    output_params = np.zeros(len(optimize_index))
    output_params[recover_index] = solved_params # Sets optimizing parameters equal to their entered value
    output_params[fixed_index]   = fixed_params # Sets fixed parameters equal to their entered value
    output_params[np.logical_not(fixed_index)] = output_params[optimize_index[np.logical_not(fixed_index)]] # Sets other parameters equal to the optimizing parameter chosen

    return output_params
