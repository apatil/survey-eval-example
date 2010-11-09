# Author: Anand Patil
# Date: 3 June 2010
# License: Gnu GPL
#####################


import numpy as np
import pymc as pm
from generic_mbg import *
import generic_mbg

__all__ = ['make_model']

def mean_fn(x, m):
    return np.zeros(x.shape[:-1])+m
    
def make_model(lon,lat,input_data,covariate_keys,pos,neg,cpus=1):
    """
    This function is required by the generic MBG code.
    """
    # Uniquify data locations    
    data_mesh, logp_mesh, fi, ui, ti = uniquify(lon,lat)
    
    # Create the mean & its evaluation at the data locations.
    m = pm.Uninformative('m',0)

    @pm.deterministic
    def M(m=m):
        return pm.gp.Mean(mean_fn,m=m)

    # The partial sill.
    amp = pm.Exponential('amp', .1, value=1.)

    # The range parameter.
    scale = pm.Exponential('scale', .1, value=.08)

    # This parameter controls the degree of differentiability of the field.
    diff_degree = pm.Uniform('diff_degree', .01, 3)

    # The nugget variance.
    V = pm.Gamma('V', 10, 100, value=.1)
    tau = 1./V
    
    # Create the covariance & its evaluation at the data locations.
    @pm.deterministic(trace=True)
    def C(amp=amp, scale=scale, diff_degree=diff_degree):
        """A covariance function created from the current parameter values."""
        return pm.gp.FullRankCovariance(pm.gp.cov_funs.matern.euclidean, amp=amp, scale=scale, diff_degree=diff_degree)
    
    # The Gaussian process submodel
    sp_sub = pm.gp.GPSubmodel('sp_sub',M,C,logp_mesh)
    
    # Add the nugget process
    eps_p_f = pm.Normal('eps_p_f', sp_sub.f_eval[fi], tau, value=pm.logit((pos+1.)/(pos+neg+2.)))

    # Probability of 'success'
    p = pm.Lambda('s',lambda lt=eps_p_f: invlogit(lt), trace=False)
    
    # The data have the 'observed' flag set to True.
    d = pm.Binomial('d', pos+neg, p, value=pos, observed=True)
        
    return locals()
