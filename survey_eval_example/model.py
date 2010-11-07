# Author: Anand Patil
# Date: 3 June 2010
# License: Gnu GPL
#####################


import numpy as np
import pymc as pm
import gc
from map_utils import *
from generic_mbg import *
import generic_mbg
from zib import zib

__all__ = ['make_model']

def mean_fn(x):
    return np.zeros(x.shape[:-1])    
    
    
def make_model(lon,lat,input_data,covariate_keys,pos,neg,cpus=1):
    """
    This function is required by the generic MBG code.
    """
    ra = csv2rec(input_data)
    
    if np.any(pos+neg==0):
        where_zero = np.where(pos+neg==0)[0]
        raise ValueError, 'Pos+neg = 0 in the rows (starting from zero):\n %s'%where_zero
    
    
    # How many nuggeted field points to handle with each step method
    grainsize = 10
        
    # Non-unique data locations
    data_mesh = combine_spatial_inputs(lon, lat)
    
    s_hat = (pos+1.)/(pos+neg+2.)
    
    # Uniquify the data locations.
    locs = [(lon[0], lat[0])]
    fi = [0]
    ui = [0]
    for i in xrange(1,len(lon)):

        # If repeat location, add observation
        loc = (lon[i], lat[i])
        if loc in locs:
            fi.append(locs.index(loc))

        # Otherwise, new obs
        else:
            locs.append(loc)
            fi.append(max(fi)+1)
            ui.append(i)
    fi = np.array(fi)
    ti = [np.where(fi == i)[0] for i in xrange(max(fi)+1)]
    ui = np.asarray(ui)

    lon = np.array(locs)[:,0]
    lat = np.array(locs)[:,1]

    # Unique data locations
    logp_mesh = combine_spatial_inputs(lon,lat)
    
    # Create the mean & its evaluation at the data locations.
    @pm.deterministic
    def M():
        return pm.gp.Mean(mean_fn)
        
    @pm.deterministic
    def M_eval(M=M):
        return M(logp_mesh)

    init_OK = False
    while not init_OK:
        try:        
            # The partial sill.
            amp = pm.Exponential('amp', .1, value=1.)

            a1 = pm.Uninformative('a1',0,observed=True)
            a2 = pm.Uninformative('a2',0,observed=True)
            
            # The range parameters. Units are RADIANS. 
            # 1 radian = the radius of the earth, about 6378.1 km
            scale = pm.Exponential('scale', .1, value=.08)
            scale_in_km = scale*6378.1

            # This parameter controls the degree of differentiability of the field.
            diff_degree = pm.Uniform('diff_degree', .01, 3)

            # The nugget variance.
            V = pm.Exponential('V', .1, value=.1, observed=True)
            tau = 1./V
            
            # Create the covariance & its evaluation at the data locations.
            @pm.deterministic(trace=True)
            def C(amp=amp, scale=scale, diff_degree=diff_degree):
                """A covariance function created from the current parameter values."""
                eval_fun = CovarianceWithCovariates(pm.gp.cov_funs.matern.geo_rad, input_data, covariate_keys, ui, fac=1.e4, ra=ra)
                return pm.gp.FullRankCovariance(eval_fun, amp=amp, scale=scale, diff_degree=diff_degree)
            
            sp_sub = pm.gp.GPSubmodel('sp_sub',M,C,logp_mesh)
            sp_sub.f_eval.value
        
            # Loop over data clusters
            eps_p_f_d = []
            s_d = []
            data_d = []

            for i in xrange(len(pos)/grainsize+1):
                sl = slice(i*grainsize,(i+1)*grainsize,None)
                # Nuggeted field in this cluster
                this_f = sp_sub.f_eval[fi[sl]]
                if len(this_f.value)>0:
                    eps_p_f_d.append(pm.Normal('eps_p_f_%i'%i, this_f, tau, value=pm.stukel_logit(s_hat[sl],a1.value,a2.value),trace=False))

                    # The allele frequency
                    s_d.append(pm.Lambda('s_%i'%i,lambda lt=eps_p_f_d[-1]: stukel_invlogit(lt,a1,a2),trace=False))

                    # The observed allele frequencies
                    @pm.stochastic(name='data_%i'%i, observed=True)
                    def d_now(value=pos[sl], n=pos[sl]+neg[sl], p=s_d[-1]):
                        return pm.binomial_like(value,n,p)
                        
                    data_d.append(d_now)
            
            # The field plus the nugget
            @pm.deterministic
            def eps_p_f(eps_p_fd = eps_p_f_d):
                """Concatenated version of eps_p_f, for postprocessing & Gibbs sampling purposes"""
                return np.concatenate(eps_p_fd)
            
            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()
        
    return locals()
