from model import *
from generic_mbg import stukel_invlogit, invlogit
import pymc as pm
import numpy as np
import flikelihood
from scipy import special
from scipy.special import gammaln

# Indicate to generic_mbg which nuggets and observations correspond to
# the Gaussian process submodel sp_sub.
nugget_labels = {'sp_sub': 'V'}
obs_labels = {'sp_sub': 'eps_p_f'}

# Indicate to generic_mbg which columns of the datafile do not indicate
# to position, time or linear covariates.
non_cov_columns = {'pos': 'float', 'neg': 'float'}

def check_data(input):
    """
    This function should take a dataset, represented as a NumPy record
    array, and raise an error if it contains problems.
    """
    if np.any(input.pos+input.neg)==0:
        raise ValueError, 'Some sample sizes are zero.'
    if np.any(np.isnan(input.pos)) or np.any(np.isnan(input.neg)):
        raise ValueError, 'Some NaNs in input'
    if np.any(input.pos<0) or np.any(input.neg<0):
        raise ValueError, 'Some negative values in pos and neg'

def example_map(sp_sub):
    """
    This function should convert sp_sub to the quantity to be mapped, 
    in this case the inverse-logit of f.
    """
    f = sp_sub.copy('F')
    p = invlogit(f)
    return p

# Indicate to generic_mbg that example_map above is the function that 
# converts Gaussian random field evaluations to maps.
map_postproc = [example_map]

def simdata_postproc(sp_sub, survey_plan):
    """
    This function should take a value for the Gaussian random field in the submodel 
    sp_sub, evaluated at the survey plan locations, and return a simulated dataset.
    """
    p = pm.invlogit(sp_sub)
    n = survey_plan.n
    return pm.rbinomial(n, p)
    
def survey_likelihood(sp_sub, survey_plan, data, i):
    """
    This function should return the log of the likelihood of data[i]
    given row i of survey_plan and each element of sp_sub. It must
    be normalized, because the evidence will be used to avoid difficult
    computations at low importance weights.
    """
    # The function 'binomial' is implemented in the Fortran extension flikelihood.f for speed.
    l = flikelihood.binomial(data[i], survey_plan.n[i], sp_sub)
    # Add the normalizing constant and return.
    return l + gammaln(survey_plan.n[i]+1)-gammaln(data[i]+1)-gammaln(survey_plan.n[i]-data[i]+1)

def mcmc_init(M):
    """
    This function should take a PyMC MCMC object and make any needed
    modifications to its jumping strategy. See
    
    Patil, A., Huard, D. and Fonnesbeck, C. "PyMC 2: Bayesian Stochastic Modeling in Python"
    Journal of Statistical Software 35(4).
    """
    # Use Haario, Saksman and Tamminen's adaptive Metropolis algorithm to handle
    # the scalar covariance parameters and the overall mean m. See
    # CITE
    M.use_step_method(pm.gp.GPParentAdaptiveMetropolis, [M.amp, M.scale, M.diff_degree, M.m])
                    
metadata_keys = ['fi','ti','ui']