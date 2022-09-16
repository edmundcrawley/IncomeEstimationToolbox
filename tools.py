'''
Tools that are used to calculate covariance matrices of time aggregated income processes
'''
import numpy as np

def vech_indices(N):
    '''
    Returns the indices of the lower trianglular elements
    of an NxN matrix
    '''
    rows = [];
    columns = []
    for i in range(N):
        rows += range(i,N)
        columns += [i]*(N-i)
    return (np.array(rows), np.array(columns))

def vech(A):
    '''
    Returns the lower trianglular elements
    of an NxN matrix as a vector
    '''
    N = A.shape[0]
    indicies = vech_indices(N)
    return A[indicies]

def inv_vech(V):
    '''
    Inverse of vech. Returns a symetric matrix
    '''
    N = np.floor((len(V)*2)**0.5 ).astype(int)
    indicies = vech_indices(N)
    A = np.zeros((N,N))
    A[indicies] = V
    A[(indicies[1],indicies[0])] = V
    return A

def cov_passing_levels(omega):
  '''
    Calculates the levels variance matrix for a passing income
    process in which shocks arrive uniformly and persist as a constant flow
    for a random time (distributed as an exponential random variable with
    parameter omega)
  '''
  expm1 = np.exp(-omega)
  int01_x2fx = 2.0/omega**2 - expm1*(1 + 2/omega + 2/omega**2)  # integral between 0 and 1 of x^2 on exponential density
  int01_xfx  = 1.0/omega - expm1*(1+1/omega)                      # integral between 0 and 1 of x on exponential density
  int01_fx   = 1 - expm1                                              # integral between 0 and 1 of exponential density
  # var_0_T0: component of variance from shocks that occur between T and T-1
  var_0_T0  = 2/omega**2 * (1 - int01_xfx - 1/omega*int01_fx)
  # var_0_T1: component of variance from shocks that occur between T-1 and T-2
  var_0_T1_a = expm1/omega*int01_fx
  var_0_T1_b = 1/omega*int01_x2fx*int01_fx
  var_0_T1 = var_0_T1_a + var_0_T1_b
  # var_0_Tinf: component of variance from shocks that occur between T-2 and -inf
  var_0_Tinf = var_0_T1/(1-expm1)
  # var_0: total variance
  var_0 = var_0_T0 + var_0_Tinf
  # cov_1_T0: component of cov(y_T, y_{T-1}) from shocks that occur between T-1 and T-2
  cov_1_T0_a = expm1/omega*int01_xfx
  cov_1_T0_b = 1/omega*int01_xfx**2
  cov_1_T0 = cov_1_T0_a + cov_1_T0_b
  # cov_1_T1: component of cov(y_T, y_{T-1}) from shocks that occur between T-2 and T-3
  cov_1_T1_a = 1/omega*expm1**2*int01_fx
  cov_1_T1_b = 1/omega*expm1*int01_fx*int01_xfx
  cov_1_T1 = cov_1_T1_a + cov_1_T1_b
  # cov_1_T1: component of cov(y_T, y_{T-1}) from shocks that occur between T-3 and -inf
  cov_1_Tinf = cov_1_T1/(1-expm1)
  # cov_1: total cov(y_T, y_{T-1})
  cov_1 = cov_1_T0 + cov_1_Tinf
  # Returning "components" allows calculation of time varying processes. Each component refers to shocks that originate in a specific year, so they could have different variances and or omegas
  components = np.array([[cov_1_T0,cov_1_T1,cov_1_Tinf],
                         [var_0_T0,var_0_T1,var_0_Tinf ],
                         [cov_1_T0,cov_1_T1,cov_1_Tinf]])
  return np.array([cov_1, var_0, cov_1]) , components

def expm1mx(x):
    '''
    Calculates exp(x) -1 - x to high precision
    '''
    if abs(x)>0.95:
        ret = np.expm1(x)-x
    else:
        shx2 = np.sinh(x/2.0)
        sh2x2 = shx2**2
        ret = 2.0 * sh2x2 + (2.0 * shx2 * (1 + sh2x2)**0.5 - x)
    return ret

def expm1mxm05x2(x):
    '''
    Calculates exp(x) -1 - x- 0.5x**2 to high precision
    '''
    if abs(x)>0.2:
        ret = expm1mx(x)-x**2/2.0
    else:
        ret = 0.0
        N=10
        for i in range(N):
            n = N-i+2
            n_factorial = 1
            for j in range(n):
                n_factorial *= (j+1)
            ret += x**n/n_factorial
    return ret

def cov_persistent_levels(rho):
  '''
    Calculates the covariance of the level of a process, starting at zero,
    in which shocks arrive and then decay exponentially
  '''
 
  if (rho==0.0): # permanent shocks (random walk)
      cov_m1 = -9999
      cov_0    = -9999
      cov_1  = -9999
      components = np.array([[1.0/2.0,1.0,-9999],
                             [1.0/3.0, 1.0 ,-9999],
                             [1.0/2.0, 1.0,-9999]])
  else:
        expm1_om   = np.expm1(-rho)
        expm1mx_om   = expm1mx(-rho)
        expm1mxm05x2_om   = expm1mxm05x2(-rho)
        expm1_2om   = np.expm1(-2*rho)
        expm1mx_2om = expm1mx(-2*rho)
        expm1mxm05x2_2om = expm1mxm05x2(-2*rho)
 
        #cov_0_T0: component of variance due to shocks that occur between T and T-1
        cov_0_T0 = 1.0/(expm1_om*expm1_om)*(              expm1mxm05x2_om/rho +              expm1mxm05x2_om/rho -                           expm1mxm05x2_2om/(2*rho))
        #cov_0_Tinf: component of variance due to shocks that occur between T-1 and -inf (assuming shocks go back to infinity - we will truncate at T=0 and not use this component)
        cov_0_Tinf = 1.0/(2*rho)
        #cov_0: variance of process (assuming shocks go back to infinity - we will truncate at T=0 and not use this component)
        cov_0 = cov_0_T0 + cov_0_Tinf
        # cov_1_T0: compoment of cov(y_T, y_{T+1}) due to shocks that occur between T and T-1
        cov_1_T0 = -1.0/(expm1_om)*(-expm1mx_om/rho + expm1mx_2om/(2*rho))
        cov_1_Tinf = np.exp(-rho)/(2*rho)
        cov_1 = cov_1_T0 + cov_1_Tinf
        
        # cov_m1_T0: compoment of cov(y_T, y_{T-1}) due to shocks that occur between T and T-1
        cov_m1_T0 = -1.0/(expm1_om)*(-expm1mx_om/rho + expm1mx_2om/(2*rho))
        # cov_m1_Tinf: compoment of cov(y_T, y_{T-1}) due to shocks that occur between T-1 and -inf
        cov_m1_Tinf = np.exp(-rho)/(2*rho)
        cov_m1 = cov_m1_T0 + cov_m1_Tinf
        # components columns: component due to shocks between i) T and T-1, ii) T-1 and T-2, and ii) T-1 and -inf
        components = np.array([[cov_m1_T0,-cov_m1_Tinf*expm1_2om,cov_m1_Tinf],
                               [cov_0_T0, -cov_0_Tinf*expm1_2om ,cov_0_Tinf ],
                               [cov_1_T0, -cov_1_Tinf*expm1_2om ,cov_1_Tinf ]])
  return np.array([cov_m1, cov_0, cov_1]) , components
  
 
