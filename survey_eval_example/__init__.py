# from mcmc import *
from model import *
from generic_mbg import stukel_invlogit, invlogit
import pymc as pm
import numpy as np
import flikelihood

nugget_labels = {'sp_sub': 'V'}
obs_labels = {'sp_sub': 'eps_p_f'}
non_cov_columns = {'pos': 'float', 'neg': 'float'}

def check_data(input):
    if np.any(input.pos+input.neg)==0:
        raise ValueError, 'Some sample sizes are zero.'
    if np.any(np.isnan(input.pos)) or np.any(np.isnan(input.neg)):
        raise ValueError, 'Some NaNs in input'
    if np.any(input.pos<0) or np.any(input.neg<0):
        raise ValueError, 'Some negative values in pos and neg'

def example_map(sp_sub):
    itn = sp_sub.copy('F')
    itn = invlogit(itn)
    return itn
    
map_postproc = [example_map]

def simdata_postproc(sp_sub, survey_plan):
    p = pm.invlogit(sp_sub)
    n = survey_plan.n
    return pm.rbinomial(n, p)
    
def survey_likelihood(sp_sub, survey_plan, data, i):
    return flikelihood.binomial(data[i], survey_plan.n[i], sp_sub)

def areal_diff(gc): 
    "Difference in areal mean between some areas" 

    def h(Ghana, Togo, V):
        "The V is in there just to test"
        return Ghana - Togo

    g = dict([(k, lambda sp_sub, x, a=v['geom'].area: invlogit(sp_sub(x))*a) for k,v in gc.iteritems()])

    return h, g

areal_postproc = [areal_diff]

def mcmc_init(M):
    M.use_step_method(pm.gp.GPParentAdaptiveMetropolis, [M.amp, M.scale, M.diff_degree, M.m])
                    
metadata_keys = ['fi','ti','ui']